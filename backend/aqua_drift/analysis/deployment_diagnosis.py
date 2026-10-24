"""Why do the automatic drops lose the target? Per-drop diagnosis of the closed loop.

Runs the in-process scenario (estimator + automatic deployment + layer) for the maneuver
scenarios of maneuver_deployment and records, for every drop task, when it was planned, when it
was laid, how far the truth passed from it and whether its observer ever detected; and over time
the number of detecting observers, the estimator status, the layer's mode and when the estimator
noticed each maneuver:

    python -m aqua_drift.analysis.deployment_diagnosis --scenarios S1 --range 500 --seeds 1 \
        --seconds 2400 --out diag.json

Uses the truth only to score the drops afterwards, never in the loop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from aqua_drift.analysis.maneuver_deployment import SCENARIOS

YD_TO_M = 0.9144


def run_one(scenario: str, seed: int, range_yd: float, seconds: int, particles: int,
            forward: dict | None = None) -> dict:
    from aqua_drift.models import ScenarioConfig
    from aqua_drift.physics import local_offset_m
    from aqua_drift.scenario import ScenarioRun

    config = ScenarioConfig()
    config.max_slant_range_yd = range_yd
    if range_yd < 3000:
        config.deployment.surround_radius_yd = 0.8 * range_yd
    config.estimator.particle_count = particles
    config.estimator.random_seed = 7 + 101 * seed
    config.bearing.random_seed = 11 + 101 * seed
    config.lloyd.random_seed = 23 + 101 * seed
    config.layer.random_seed = 31 + 101 * seed
    for key, value in (forward or {}).items():
        setattr(config.forward, key, value)
    run = ScenarioRun(config, 4, forward=True)
    events = {tick: rest for tick, *rest in SCENARIOS[scenario]}

    drops: dict[int, dict] = {}  # task id -> record
    observer_of: dict[str, int] = {}  # observer id -> task id
    first_seen: dict[int, int] = {}
    min_dist: dict[int, float] = {}
    series: dict[str, list] = {k: [] for k in ("t", "det", "status", "layer", "maneuver_tick", "err")}
    started = time.time()
    for tick in range(1, seconds + 1):
        if tick in events:
            hdg, speed, z = events[tick]
            if hdg is not None:
                run.config.target.desired_hdg_deg = hdg
            if speed is not None:
                run.config.target.desired_through_water_speed_kt = speed
            if z is not None:
                run.config.target.desired_depth_ft = z
        known = {o.observer_id for o in run.observers}
        batch = run.step()
        for task in run.tasks:
            rec = drops.get(task.task_id)
            if rec is None:
                east, north, _ = local_offset_m(run.target.position, task.position)
                rec = drops[task.task_id] = {
                    "id": task.task_id, "source": task.source, "created": task.created_tick,
                    "planned": task.planned_tick, "revision": task.revision, "reason": task.reason,
                    "truth_dist_at_plan_yd": math.hypot(east, north) / YD_TO_M,
                }
            rec["status"] = task.status
            if task.status == "DONE" and "done" not in rec:
                rec["done"] = task.done_tick
                east, north, _ = local_offset_m(run.target.position, task.position)
                rec["truth_dist_at_splash_yd"] = math.hypot(east, north) / YD_TO_M
                rec["depth_ft"] = task.position.depth_ft
                fresh = [o for o in run.observers if o.observer_id not in known]
                if fresh:  # the observer laid this tick (one per drop)
                    observer_of[fresh[-1].observer_id] = task.task_id
                    known.add(fresh[-1].observer_id)
        for obs in batch.observations:
            task_id = observer_of.get(obs.observer_id)
            if task_id is None:
                continue
            east, north, _ = local_offset_m(run.target.position, obs.observer_position)
            d = math.hypot(east, north) / YD_TO_M
            min_dist[task_id] = min(min_dist.get(task_id, math.inf), d)
            if obs.detected and task_id not in first_seen:
                first_seen[task_id] = tick
        if tick % 10:
            continue
        est = run.engine.output().estimates[0]
        err = run.error(run.engine.output())
        series["t"].append(tick)
        series["det"].append(sum(1 for o in batch.observations if o.detected))
        series["status"].append(est.observability_status)
        series["layer"].append(run.layer_state.mode if run.layer_state is not None else None)
        series["maneuver_tick"].append((est.metadata or {}).get("maneuver_detected_tick"))
        series["err"].append(err.get("horizontal_error_yd"))
    for task_id, rec in drops.items():
        rec["first_detect"] = first_seen.get(task_id)
        rec["min_truth_dist_yd"] = min_dist.get(task_id)
    return {
        "scenario": scenario, "seed": seed, "range_yd": range_yd, "events": SCENARIOS[scenario],
        "drops": list(drops.values()), "series": series, "wall_s": time.time() - started,
    }


def summarize(result: dict) -> dict:
    r_yd = result["range_yd"]
    drops = [d for d in result["drops"] if d["status"] == "DONE"]
    s = result["series"]
    t, det = np.array(s["t"]), np.array(s["det"])
    latency = [d["done"] - d["created"] for d in drops]
    lateness = [d["done"] - d["planned"] for d in drops if d["planned"] is not None]
    useful = [d for d in drops if d["first_detect"] is not None]
    layer = [m for m in s["layer"] if m is not None]
    out = {
        "case": f"{result['scenario']} R={r_yd:.0f} seed {result['seed']}",
        "laid": len(drops), "useful": len(useful),
        "useful_pct": 100.0 * len(useful) / max(len(drops), 1),
        "plan_to_splash_med_s": float(np.median(latency)) if latency else math.nan,
        "late_vs_planned_med_s": float(np.median(lateness)) if lateness else math.nan,
        "splash_dist_med_R": float(np.median([d["truth_dist_at_splash_yd"] for d in drops])) / r_yd if drops else math.nan,
        "pass_dist_med_R": float(np.median([d["min_truth_dist_yd"] for d in drops
                                           if d["min_truth_dist_yd"] is not None])) / r_yd if drops else math.nan,
        "no_detection_pct": 100.0 * float(np.mean(det == 0)),
        "layer_transit_pct": 100.0 * sum(m == "TRANSIT" for m in layer) / max(len(layer), 1),
        "replaced": sum(d["status"] == "REPLACED" for d in result["drops"]),
    }
    noticed = []
    for event_tick, *_ in result["events"]:
        hit = next((tt for tt, m in zip(t, s["maneuver_tick"]) if m is not None and m > event_tick), None)
        noticed.append(None if hit is None else int(hit - event_tick))
        after = [d for d in useful if d["created"] > event_tick]
        out[f"event {event_tick}"] = {
            "noticed_after_s": noticed[-1],
            "first_useful_new_drop_after_s": min((d["first_detect"] - event_tick for d in after), default=None),
            "no_detection_pct_next_600s": 100.0 * float(np.mean(det[(t > event_tick) & (t <= event_tick + 600)] == 0)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", default="S0,S1,S2,S3,S4")
    parser.add_argument("--range", type=float, default=500.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seconds", type=int, default=2400)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--set", action="append", default=[], help="forward setting key=json value")
    parser.add_argument("--out", default=None, help="results as JSON lines (resumes from it)")
    args = parser.parse_args()
    forward = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
    jobs = [(sc, seed) for sc in args.scenarios.split(",") for seed in range(args.seeds)]
    results: list[dict] = []
    if args.out and os.path.exists(args.out):
        with open(args.out) as handle:
            results = [json.loads(line) for line in handle if line.strip()]
    done = {(r["scenario"], r["seed"]) for r in results}

    def keep(result: dict) -> None:
        results.append(result)
        if args.out:
            with open(args.out, "a") as handle:
                handle.write(json.dumps(result) + "\n")

    todo = [j for j in jobs if j not in done]
    if args.jobs > 1 and todo:
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = [pool.submit(run_one, sc, seed, args.range, args.seconds, args.particles, forward)
                       for sc, seed in todo]
            for future in as_completed(futures):
                keep(future.result())
    else:
        for sc, seed in todo:
            keep(run_one(sc, seed, args.range, args.seconds, args.particles, forward))
    for result in sorted(results, key=lambda r: (r["scenario"], r["seed"])):
        print(json.dumps(summarize(result), ensure_ascii=False))


if __name__ == "__main__":
    main()
