"""Lost contact (失探知): closed-loop study of what happens when no observer detects the target.

Runs the in-process scenario (estimator + automatic optimal deployment + layer) and records,
every 10 s, the number of detecting observers, the estimator status, the horizontal error and
1 sigma, and whether the truth lies inside the presence region:

    python -m aqua_drift.analysis.lost_contact --seeds 4 --seconds 3000 --jobs 4

Scenarios (target 8 kt, HDG 090, 500 Ft at the start; R = max slant range):
  L0 straight                           L1 90 deg course change at 1200 s
  L2 speed 8 -> 16 kt at 1200 s         L3 180 deg course change at 1200 s
Ranges: "6000" (default field) and "500" (initial square shrunk to 400 YD).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SCENARIOS: dict[str, list[tuple[int, float | None, float | None]]] = {
    "L0": [],
    "L1": [(1200, 180.0, None)],
    "L2": [(1200, None, 16.0)],
    "L3": [(1200, 270.0, None)],
}


def run_one(scenario: str, seed: int, range_yd: float, seconds: int, particles: int,
            variant: str = "current") -> dict:
    from aqua_drift.models import ScenarioConfig
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
    run = ScenarioRun(config, 4, forward=True)
    if variant != "current":
        from aqua_drift.analysis import lost_contact_proto
        lost_contact_proto.install(run, variant)
    events = {tick: rest for tick, *rest in SCENARIOS[scenario]}
    rec: dict[str, list] = {k: [] for k in ("t", "det", "status", "err", "sigma", "inside")}
    started = time.time()
    for tick in range(1, seconds + 1):
        if tick in events:
            hdg, speed = events[tick]
            if hdg is not None:
                run.config.target.desired_hdg_deg = hdg
            if speed is not None:
                run.config.target.desired_through_water_speed_kt = speed
        batch = run.step()
        if tick % 10:
            continue
        output = run.engine.output()
        est = output.estimates[0]
        err = run.error(output)
        rec["t"].append(tick)
        rec["det"].append(sum(1 for o in batch.observations if o.detected))
        rec["status"].append(est.observability_status)
        rec["err"].append(err.get("horizontal_error_yd"))
        rec["sigma"].append(est.uncertainty.horizontal_major_yd if est.uncertainty else None)
        rec["inside"].append(_inside(run, est))
    return {
        "scenario": scenario, "seed": seed, "range_yd": range_yd, "variant": variant, **rec,
        "reinit": run.engine.reinitializations,
        "drops": sum(1 for t in run.tasks if t.status == "DONE"),
        "tasks": len(run.tasks),
        "wall_s": time.time() - started,
    }


def _inside(run, est) -> bool | None:
    """Truth horizontally inside the presence region (any component's outline)."""
    region = est.presence_region
    if est.current_position is None or region is None or not region.components:
        return None
    lon, lat = run.target.position.longitude, run.target.position.latitude
    return any(_in_polygon(lon, lat, c.polygon) for c in region.components)


def _in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def summarize(results: list[dict]) -> dict:
    table = {}
    for key in sorted({(r["variant"], r["range_yd"], r["scenario"]) for r in results}):
        rows = [r for r in results if (r["variant"], r["range_yd"], r["scenario"]) == key]
        lost, coast, err, gaps, longest, inside, drops, reinit, never = [], [], [], [], [], [], [], [], 0
        for r in rows:
            det = np.array(r["det"])
            t = np.array(r["t"])
            started = np.argmax(det > 0) if (det > 0).any() else len(det)
            det, t = det[started:], t[started:]
            lost.extend(det == 0)
            coast.extend(s.startswith("COASTING") for s in r["status"][started:])
            err.extend(e if e is not None else 1e4 for e in r["err"][started:])
            inside.extend(bool(v) for v in r["inside"][started:] if v is not None)
            run_len = 0
            gap_max = 0
            for d in det:
                if d == 0:
                    run_len += 10
                    gap_max = max(gap_max, run_len)
                else:
                    if run_len:
                        gaps.append(run_len)
                    run_len = 0
            if run_len:
                gaps.append(run_len)
                never += 1  # still lost at the end
            longest.append(gap_max)
            drops.append(r["drops"])
            reinit.append(r["reinit"])
        e = np.array(err)
        table[f"{key[0]} R={key[1]:.0f} {key[2]}"] = {
            "runs": len(rows),
            "lost_frac": float(np.mean(lost)) if lost else math.nan,
            "coast_frac": float(np.mean(coast)) if coast else math.nan,
            "gaps": len(gaps), "gap_mean_s": float(np.mean(gaps)) if gaps else 0.0,
            "longest_s": float(np.mean(longest)),
            "lost_at_end": never,
            "err_median": float(np.median(e)) if len(e) else math.nan,
            "err_p95": float(np.percentile(e, 95)) if len(e) else math.nan,
            "inside": float(np.mean(inside)) if inside else math.nan,
            "drops": float(np.mean(drops)), "reinit": float(np.mean(reinit)),
        }
    return table


def print_table(table: dict) -> None:
    print("| case | runs | lost % | COASTING % | gaps | mean gap s | longest s | lost at end | err med YD | err p95 YD | truth in cloud % | drops | reinit |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in table.items():
        print(f"| {name} | {m['runs']} | {100*m['lost_frac']:.1f} | {100*m['coast_frac']:.1f} | {m['gaps']} | "
              f"{m['gap_mean_s']:.0f} | {m['longest_s']:.0f} | {m['lost_at_end']} | {m['err_median']:.0f} | "
              f"{m['err_p95']:.0f} | {100*m['inside']:.0f} | {m['drops']:.1f} | {m['reinit']:.1f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--ranges", default="6000,500")
    parser.add_argument("--variants", default="current")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None, help="per-run series as JSON lines (resumes from it)")
    args = parser.parse_args()
    jobs = [(s, seed, float(r), args.seconds, args.particles, v)
            for v in args.variants.split(",") for r in args.ranges.split(",")
            for s in args.scenarios.split(",") for seed in range(args.seeds)]
    results: list[dict] = []
    if args.out and os.path.exists(args.out):
        with open(args.out) as handle:
            results = [json.loads(line) for line in handle if line.strip()]
    done = {(r["scenario"], r["seed"], r["range_yd"], r["variant"]) for r in results}
    todo = [j for j in jobs if (j[0], j[1], j[2], j[5]) not in done]

    def keep(result: dict) -> None:
        results.append(result)
        if args.out:
            with open(args.out, "a") as handle:
                handle.write(json.dumps(result) + "\n")

    if args.jobs > 1 and todo:
        with ProcessPoolExecutor(args.jobs) as pool:
            for result in pool.map(run_one, *zip(*todo)):
                keep(result)
    else:
        for job in todo:
            keep(run_one(*job))
    print_table(summarize(results))


if __name__ == "__main__":
    main()
