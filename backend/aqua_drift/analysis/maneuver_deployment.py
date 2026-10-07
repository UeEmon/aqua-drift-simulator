"""Automatic deployment against target maneuvers: closed-loop comparison study.

Runs the in-process scenario (estimator + automatic optimal deployment + layer) for target
motions with course, speed and depth changes, with and without bearings, over several seeds,
and reports tracking accuracy, recovery after maneuvers and loss of contact:

    python -m aqua_drift.analysis.maneuver_deployment --seeds 4 --seconds 3000 --jobs 4 \
        --out results.json

Scenarios (target 8 kt, HDG 090, 500 Ft at the start):
  S0 straight (regression check)
  S1 90 deg course change at 1200 s
  S2 speed 8 -> 4 kt at 1200 s, 4 -> 14 kt at 2400 s
  S3 depth 500 -> 1300 Ft from 900 s
  S4 zigzag +-60 deg every 600 s with speed and depth changes
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# (tick, desired HDG, desired speed kt, desired depth Ft); None keeps the value
SCENARIOS: dict[str, list[tuple[int, float | None, float | None, float | None]]] = {
    "S0": [],
    "S1": [(1200, 180.0, None, None)],
    "S2": [(1200, None, 4.0, None), (2400, None, 14.0, None)],
    "S3": [(900, None, None, 1300.0)],
    "S4": [
        (600, 30.0, 10.0, None), (1200, 150.0, 6.0, 900.0), (1800, 90.0, 12.0, None),
        (2400, 30.0, 8.0, 300.0),
    ],
}


def run_one(scenario: str, seed: int, bearing: bool, seconds: int, particles: int) -> dict:
    from aqua_drift.models import ScenarioConfig
    from aqua_drift.scenario import ScenarioRun

    config = ScenarioConfig()
    config.estimator.particle_count = particles
    config.estimator.use_bearing = bearing
    config.estimator.random_seed = 7 + 101 * seed
    config.bearing.random_seed = 11 + 101 * seed
    config.lloyd.random_seed = 23 + 101 * seed
    config.layer.random_seed = 31 + 101 * seed
    events = {tick: rest for tick, *rest in SCENARIOS[scenario]}
    run = ScenarioRun(config, 4, forward=True)
    horizontal, depth, detecting, sigma, times = [], [], [], [], []
    started = time.time()
    plan_seconds: list[float] = []
    original = run._forward_deploy

    def timed() -> None:
        t0 = time.perf_counter()
        original()
        plan_seconds.append(time.perf_counter() - t0)

    run._forward_deploy = timed
    for tick in range(1, seconds + 1):
        if tick in events:
            hdg, speed, z = events[tick]
            if hdg is not None:
                run.config.target.desired_hdg_deg = hdg
            if speed is not None:
                run.config.target.desired_through_water_speed_kt = speed
            if z is not None:
                run.config.target.desired_depth_ft = z
        batch = run.step()
        if tick % 10:
            continue
        detecting.append(sum(1 for o in batch.observations if o.detected))
        output = run.engine.output()
        err = run.error(output)
        times.append(tick)
        if not err:
            horizontal.append(None)
            depth.append(None)
            sigma.append(None)
            continue
        horizontal.append(err["horizontal_error_yd"])
        depth.append(err["depth_error_ft"])
        sigma.append(output.estimates[0].uncertainty.horizontal_major_yd)
    return {
        "scenario": scenario, "seed": seed, "bearing": bearing, "times": times,
        "horizontal_yd": horizontal, "depth_ft": depth, "detecting": detecting, "sigma_yd": sigma,
        "drops": sum(n for _, n in run.deployments) if not config.layer.enabled
        else sum(1 for t in run.tasks if t.status == "DONE"),
        "plans": len(plan_seconds), "plan_max_s": max(plan_seconds, default=0.0),
        "plan_mean_s": float(np.mean(plan_seconds)) if plan_seconds else 0.0,
        "wall_s": time.time() - started,
    }


def summarize(results: list[dict], settle_s: int = 600) -> dict:
    """Metrics per (scenario, bearing): RMS / 95 % horizontal error and RMS depth error after
    settle_s, recovery time after each maneuver, fraction of time with < 2 detecting observers."""
    table: dict[str, dict] = {}
    for key in sorted({(r["scenario"], r["bearing"]) for r in results}):
        rows = [r for r in results if (r["scenario"], r["bearing"]) == key]
        h, d, lost, recovery = [], [], [], []
        for r in rows:
            for t, e, z, n in zip(r["times"], r["horizontal_yd"], r["depth_ft"], r["detecting"]):
                if t < settle_s:
                    continue
                h.append(e if e is not None else 1e4)
                d.append(z if z is not None else 1e4)
                lost.append(n < 2)
            for tick, *_ in SCENARIOS[r["scenario"]]:
                rec = None
                for t, e, s in zip(r["times"], r["horizontal_yd"], r["sigma_yd"]):
                    if t <= tick + 60 or e is None:
                        continue
                    if e <= max(2.0 * s, 50.0):
                        rec = t - tick
                        break
                    rec = None
                recovery.append(rec if rec is not None else float("nan"))
        h_arr = np.array(h)
        table[f"{key[0]} {'bearing' if key[1] else 'doppler'}"] = {
            "runs": len(rows),
            "horizontal_rms_yd": float(np.sqrt(np.mean(h_arr**2))) if h else math.nan,
            "horizontal_p95_yd": float(np.percentile(h_arr, 95)) if h else math.nan,
            "depth_rms_ft": float(np.sqrt(np.mean(np.array(d) ** 2))) if d else math.nan,
            "under2_detecting": float(np.mean(lost)) if lost else math.nan,
            "recovery_s": float(np.nanmean(recovery)) if recovery and not all(map(math.isnan, recovery)) else math.nan,
            "drops": float(np.mean([r["drops"] for r in rows])),
            "plan_max_s": float(max(r["plan_max_s"] for r in rows)),
        }
    return table


def print_table(table: dict) -> None:
    print("| case | runs | horiz RMS YD | horiz p95 YD | depth RMS Ft | <2 detecting | recovery s | drops | plan max s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in table.items():
        print(
            f"| {name} | {m['runs']} | {m['horizontal_rms_yd']:.0f} | {m['horizontal_p95_yd']:.0f} | "
            f"{m['depth_rms_ft']:.0f} | {100 * m['under2_detecting']:.1f} % | {m['recovery_s']:.0f} | "
            f"{m['drops']:.1f} | {m['plan_max_s']:.2f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--bearing", choices=["both", "on", "off"], default="both")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None, help="per-run series as JSON lines (resumes from it)")
    args = parser.parse_args()
    modes = {"both": [True, False], "on": [True], "off": [False]}[args.bearing]
    jobs = [(s, seed, b, args.seconds, args.particles)
            for s in args.scenarios.split(",") for b in modes for seed in range(args.seeds)]
    results: list[dict] = []
    done: set[tuple] = set()
    if args.out and os.path.exists(args.out):  # resume: one JSON line per finished run
        with open(args.out) as handle:
            results = [json.loads(line) for line in handle if line.strip()]
        done = {(r["scenario"], r["seed"], r["bearing"]) for r in results}
    todo = [job for job in jobs if job[0:3] not in done]

    def keep(result: dict) -> None:
        results.append(result)
        if args.out:
            with open(args.out, "a") as handle:
                handle.write(json.dumps(result) + "\n")

    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            for result in pool.map(run_one, *zip(*todo)) if todo else []:
                keep(result)
    else:
        for job in todo:
            keep(run_one(*job))
    print_table(summarize(results))


if __name__ == "__main__":
    main()
