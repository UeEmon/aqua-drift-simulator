"""Observation-mode comparison study (analysis tool; the runtime estimator uses Doppler only).

Compares, on the same simulated geometry (same target, same drifting observers, same common
maximum slant range), the accuracy achievable with:

    POS      : target position observed directly
    RB       : slant range + horizontal bearing
    BRG      : horizontal bearing only (sigma 15 deg, every 15 s, independent errors)
    DOP      : Doppler frequency only (1 s, common unknown frequency bias)
    DOP+GATE : Doppler + detection/non-detection at the known common max slant range
               (no missed detections inside the range, so each detection start/end is a
               slant-range = R_max measurement) -> the observation set of the runtime system
    BRG+DOP  : bearing + Doppler
    RB+DOP   : range/bearing + Doppler
    BRG+DOP+GATE : bearing + Doppler + detection gate -> current runtime observation set

Two numbers are reported per mode:
* CRLB  - Cramer-Rao lower bound of the constant-velocity batch estimate at the final time
          (Fisher information evaluated at the truth); the best any unbiased method can do.
* MC    - RMSE of a Gauss-Newton batch least-squares estimate over Monte-Carlo noise draws,
          started from a perturbed truth (local accuracy; global ambiguity is handled by the
          particle filter in the runtime system, not here).

Unspecified noise levels are parameters (defaults are stated assumptions):
    position sigma 50 YD (horizontal) / 30 Ft (depth), range sigma 2 % of range,
    Doppler resolution 0.03 Hz.

    python -m aqua_drift.analysis.compare_modes --observers 4 --runs 30
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from aqua_drift.models import ScenarioConfig
from aqua_drift.physics import local_offset_m
from aqua_drift.scenario import ScenarioRun

YD = 0.9144
FT = 0.3048
KT = 0.5144444444444445


@dataclass
class Geometry:
    ticks: np.ndarray  # (T,)
    obs_pos: np.ndarray  # (T,R,3) local metres
    obs_vel: np.ndarray  # (T,R,3)
    det: np.ndarray  # (T,R)
    truth: np.ndarray  # (6,) [p, v] at final tick (ground velocity)
    max_range: float
    transitions: list[tuple[float, int]]  # (mid tick, observer index) of gate crossings
    f_rec: float
    f_true: float
    c: float


@dataclass
class Noise:
    pos_h_m: float = 50 * YD
    pos_v_m: float = 30 * FT
    range_frac: float = 0.02
    bearing_rad: float = math.radians(15.0)
    bearing_interval_s: int = 15
    doppler_hz: float = 0.03
    gate_m: float = 3.0  # timing of a crossing within 1 s -> ~ relative speed x 0.3 s
    depth_prior_mean_ft: float = 750.0
    depth_prior_sigma_ft: float = 433.0  # uniform 0..1500 Ft


def build_geometry(observers: int, seconds: int, pattern: str = "grid") -> Geometry:
    config = ScenarioConfig()
    config.deployment.pattern = pattern
    config.estimator.particle_count = 500  # engine unused here
    run = ScenarioRun(config, observers)
    run.engine.process = lambda batch: None  # geometry only; no filtering needed here
    origin = run.observers[0].position
    ticks, pos, vel, det = [], [], [], []
    truth_track = []
    for _ in range(seconds):
        batch = run.step()
        ticks.append(run.tick)
        pos.append([local_offset_m(origin, o.position) for o in run.observers])
        vel.append([
            (o.ground_velocity.east_kt * KT, o.ground_velocity.north_kt * KT,
             o.ground_velocity.vertical_fps * FT) for o in run.observers
        ])
        det.append([o.detected for o in batch.observations])
        truth_track.append(local_offset_m(origin, run.target.position))
    target = run.target
    truth = np.array([
        *truth_track[-1],
        target.ground_velocity.east_kt * KT,
        target.ground_velocity.north_kt * KT,
        target.ground_velocity.vertical_fps * FT,
    ])
    det_arr = np.array(det, dtype=bool)
    transitions = [
        (ticks[i] - 0.5, j)
        for i in range(1, len(ticks))
        for j in range(det_arr.shape[1])
        if det_arr[i, j] != det_arr[i - 1, j]
    ]
    return Geometry(
        max_range=config.max_slant_range_yd * YD,
        transitions=transitions,
        ticks=np.array(ticks, dtype=float),
        obs_pos=np.array(pos),
        obs_vel=np.array(vel),
        det=np.array(det, dtype=bool),
        truth=truth,
        f_rec=config.source.source_frequency_hz + config.source.shared_recognition_bias_hz,
        f_true=config.source.source_frequency_hz,
        c=config.source.sound_speed_mps,
    )


def measurements(theta: np.ndarray, geo: Geometry, kinds: set[str], noise: Noise) -> np.ndarray:
    """Whitened model predictions (divide by sigma). theta = [p(3), v(3), bias]."""
    t_end = geo.ticks[-1]
    p = theta[0:3][None, :] - theta[3:6][None, :] * (t_end - geo.ticks)[:, None]  # (T,3)
    rel = p[:, None, :] - geo.obs_pos  # (T,R,3)
    out = []
    first_det = geo.det.any(axis=1)
    if "POS" in kinds:
        rows = first_det
        out += [p[rows, 0] / noise.pos_h_m, p[rows, 1] / noise.pos_h_m, p[rows, 2] / noise.pos_v_m]
    sampled = (geo.ticks.astype(int) % noise.bearing_interval_s == 0)[:, None] & geo.det
    if "RB" in kinds:
        rng = np.linalg.norm(rel, axis=2)
        out.append(np.log(np.maximum(rng[sampled], 1.0)) / noise.range_frac)
    if "RB" in kinds or "BRG" in kinds:
        brg = np.arctan2(rel[..., 0], rel[..., 1])
        out.append(brg[sampled] / noise.bearing_rad)
    if "DOP" in kinds:
        rng = np.linalg.norm(rel, axis=2)
        rdot = np.sum((theta[3:6][None, None, :] - geo.obs_vel) * rel, axis=2) / np.maximum(rng, 1e-6)
        f = (geo.f_rec - theta[6]) * (1.0 - rdot / geo.c)
        out.append(f[geo.det] / noise.doppler_hz)
    if "GATE" in kinds and geo.transitions:
        gate = []
        for t_mid, j in geo.transitions:
            i = int(np.searchsorted(geo.ticks, t_mid))
            o = 0.5 * (geo.obs_pos[i - 1, j] + geo.obs_pos[i, j])
            q = theta[0:3] - theta[3:6] * (t_end - t_mid)
            gate.append(np.linalg.norm(q - o) / noise.gate_m)
        out.append(np.array(gate))
    out.append(np.array([theta[2] / (noise.depth_prior_sigma_ft * FT)]))
    return np.concatenate([np.atleast_1d(x) for x in out]) if out else np.zeros(0)


def _angle_scale(geo: Geometry, kinds: set[str], noise: Noise) -> np.ndarray:
    """Per-entry period (2 pi / sigma) for bearing entries, 0 for linear entries."""
    sizes = []
    first_det = geo.det.any(axis=1)
    sampled = (geo.ticks.astype(int) % noise.bearing_interval_s == 0)[:, None] & geo.det
    if "POS" in kinds:
        sizes.append((3 * int(first_det.sum()), 0.0))
    if "RB" in kinds:
        sizes.append((int(sampled.sum()), 0.0))
    if "RB" in kinds or "BRG" in kinds:
        sizes.append((int(sampled.sum()), 2 * np.pi / noise.bearing_rad))
    if "DOP" in kinds:
        sizes.append((int(geo.det.sum()), 0.0))
    if "GATE" in kinds and geo.transitions:
        sizes.append((len(geo.transitions), 0.0))
    sizes.append((1, 0.0))  # depth prior
    return np.concatenate([np.full(n, period) for n, period in sizes])


def wrapped_diff(a: np.ndarray, b: np.ndarray, period: np.ndarray) -> np.ndarray:
    d = a - b
    ang = period > 0
    d[ang] = (d[ang] + period[ang] / 2) % period[ang] - period[ang] / 2
    return d


def _jacobian(theta: np.ndarray, geo: Geometry, kinds: set[str], noise: Noise) -> np.ndarray:
    steps = np.array([1.0, 1.0, 0.3, 0.01, 0.01, 0.003, 0.001])
    base = measurements(theta, geo, kinds, noise)
    period = _angle_scale(geo, kinds, noise)
    cols = []
    for i, h in enumerate(steps):
        d = theta.copy()
        d[i] += h
        cols.append(wrapped_diff(measurements(d, geo, kinds, noise), base, period) / h)
    return np.stack(cols, axis=1)


def crlb(geo: Geometry, kinds: set[str], noise: Noise) -> np.ndarray:
    theta = np.append(geo.truth, geo.f_rec - geo.f_true)
    jac = _jacobian(theta, geo, kinds, noise)
    fisher = jac.T @ jac
    if "DOP" not in kinds:
        fisher[6, 6] += 1.0  # bias not involved
    fisher += np.diag([1e-12] * 6 + [1.0 / 0.5**2])  # weak prior: bias sigma 0.5 Hz
    return np.linalg.pinv(fisher)


def monte_carlo(geo: Geometry, kinds: set[str], noise: Noise, runs: int, rng: np.random.Generator):
    theta_true = np.append(geo.truth, geo.f_rec - geo.f_true)
    clean = measurements(theta_true, geo, kinds, noise)
    prior_value = noise.depth_prior_mean_ft / noise.depth_prior_sigma_ft
    period = _angle_scale(geo, kinds, noise)
    assert len(period) == len(clean), "angle mask out of sync with measurements"
    errors = []
    for _ in range(runs):
        z = clean + rng.normal(size=clean.shape)
        z[-1] = prior_value  # depth prior is a fixed pseudo-measurement, not truth-centred
        x0 = theta_true + rng.normal(0, 1, 7) * np.array([300, 300, 100, 0.5, 0.5, 0.05, 0.05])

        def resid(theta: np.ndarray, z: np.ndarray = z) -> np.ndarray:
            r = wrapped_diff(measurements(theta, geo, kinds, noise), z, period)
            return np.append(r, theta[6] / 0.5)

        sol = least_squares(resid, x0, x_scale=np.array([300, 300, 30, 0.5, 0.5, 0.05, 0.05]))
        errors.append(sol.x - theta_true)
    return np.array(errors)


MODES = {
    "POS": {"POS"},
    "RB": {"RB"},
    "BRG": {"BRG"},
    "DOP": {"DOP"},
    "DOP+GATE": {"DOP", "GATE"},
    "BRG+DOP": {"BRG", "DOP"},
    "RB+DOP": {"RB", "DOP"},
    "BRG+DOP+GATE": {"BRG", "DOP", "GATE"},
}


def run_study(
    observers: int, seconds: int, runs: int, seed: int = 3, pattern: str = "grid"
) -> list[dict[str, float]]:
    geo = build_geometry(observers, seconds, pattern)
    noise = Noise()
    rng = np.random.default_rng(seed)
    rows = []
    for name, kinds in MODES.items():
        cov = crlb(geo, kinds, noise)
        err = monte_carlo(geo, kinds, noise, runs, rng)
        rows.append({
            "mode": name,
            "crlb_pos_yd": math.sqrt(max(cov[0, 0] + cov[1, 1], 0)) / YD,
            "crlb_depth_ft": math.sqrt(max(cov[2, 2], 0)) / FT,
            "crlb_vel_kt": math.sqrt(max(cov[3, 3] + cov[4, 4], 0)) / KT,
            "mc_pos_yd": float(np.sqrt(np.mean(err[:, 0] ** 2 + err[:, 1] ** 2))) / YD,
            "mc_depth_ft": float(np.sqrt(np.mean(err[:, 2] ** 2))) / FT,
            "mc_vel_kt": float(np.sqrt(np.mean(err[:, 3] ** 2 + err[:, 4] ** 2))) / KT,
        })
    detections = int(geo.det.sum())
    for row in rows:
        row["detections"] = detections
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--observers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=1500)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--pattern", default="grid", help="observer layout (grid, surround, ...)")
    args = parser.parse_args()
    rows = run_study(args.observers, args.seconds, args.runs, pattern=args.pattern)
    print(f"observers={args.observers} seconds={args.seconds} runs={args.runs}")
    print("| mode | CRLB pos (YD) | MC pos (YD) | CRLB depth (Ft) | MC depth (Ft) | CRLB vel (kt) | MC vel (kt) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['crlb_pos_yd']:.1f} | {r['mc_pos_yd']:.1f} | {r['crlb_depth_ft']:.1f} | "
            f"{r['mc_depth_ft']:.1f} | {r['crlb_vel_kt']:.3f} | {r['mc_vel_kt']:.3f} |"
        )


if __name__ == "__main__":
    main()
