"""Optimal automatic observer deployment for Doppler tracking (positions, number and depths).

Decides from the ESTIMATE ONLY (never the simulator truth). Geometry is planned in the water
frame: observers drift with the water and the target is carried by the same water, so relative
to the observers the target moves with its estimated through-water velocity and a dropped
observer stays where it is dropped.

Criterion: the Fisher information that the Doppler measurements (1 s, sigma_f, common
frequency bias, detection only inside the common maximum slant range R_max) would give about
the target state theta = [p0, v, b] over the planning horizon. For a constant-velocity pass

    f(t) = f0 (1 - rdot(t)/c),  rdot = u . v,  u = (p(t) - o) / r,  p(t) = p0 + v t
    d rdot / d p0 = (v - rdot u) / r,   d rdot / d v = t (v - rdot u) / r + u

so the information of one candidate observer o is J_o = sum_t g g^T / sigma_f^2 over the
times it detects (slant range <= R_max). The planning cost is the predicted error of the
target position (horizontal + weighted depth) at the end of the lead time and at the end of
the horizon, from the prior (current estimate uncertainty) + existing / pending observers
+ the new ones, averaged over heading hypotheses (estimate +-1 sigma) for robustness.

Candidates: a grid ahead of the estimate along the predicted track and across it (within the
detection range, in lanes parallel to the track) x a set of observer depths. Greedy selection:
add the candidate that reduces the cost most; stop as soon as the predicted error reaches
target_error_yd, when the next one would improve the cost by less than min_relative_gain, or at
max_per_drop / the free observer slots. That gives the number of observers; their depths come
out of the same criterion (the depth information grows with the vertical share of the slant
range: observers on the track and at depths far from the target depth). One observer per lane:
observers in one line give a mirror ambiguity about that line that the local (Fisher)
criterion cannot see.

Trigger: deploy when the predicted position after lead_time_s is not covered by enough
observers, or when the predicted error exceeds target_error_yd and the best new observer would
improve it by at least trigger_gain (the existing field will not track the target well ahead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position, TrackEstimate
from aqua_drift.physics import local_offset_m

YD_TO_M = 0.9144
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
SAMPLE_S = 10.0  # Fisher information sampled every 10 s (each sample stands for 10 x 1 s)


@dataclass
class PlanReport:
    count: int
    depths_ft: list[float]
    cost_before_yd: float  # predicted position error (1 sigma, YD, horizontal + weighted depth)
    cost_after_yd: float
    first_gain: float
    candidates: int
    horizontal_before_yd: float = 0.0
    horizontal_after_yd: float = 0.0
    depth_before_ft: float = 0.0
    depth_after_ft: float = 0.0


def _fisher(
    obs: np.ndarray,  # (M,3) observer positions, water frame (m, depth positive down)
    p0: np.ndarray,  # (3,) target position now
    v: np.ndarray,  # (3,) target through-water velocity
    ticks: np.ndarray,  # (T,) seconds from now
    f0: float,
    c: float,
    sigma_f: float,
    max_range: float,
) -> np.ndarray:
    """(M,7,7) Doppler Fisher information of each observer about theta = [p0, v, b]."""
    pos = p0[None, :] + v[None, :] * ticks[:, None]  # (T,3)
    rel = pos[None, :, :] - obs[:, None, :]  # (M,T,3)
    r = np.linalg.norm(rel, axis=2)
    u = rel / np.maximum(r, 1.0)[..., None]
    rdot = np.sum(u * v[None, None, :], axis=2)
    drdp = (v[None, None, :] - rdot[..., None] * u) / np.maximum(r, 1.0)[..., None]
    scale = -f0 / c
    g = np.concatenate(
        [scale * drdp, scale * (ticks[None, :, None] * drdp + u), -np.ones(r.shape + (1,))], axis=2
    )  # (M,T,7)
    detect = (r <= max_range).astype(float) * (SAMPLE_S / sigma_f**2)
    return np.einsum("mt,mti,mtj->mij", detect, g, g)


def _components(info: np.ndarray, eval_ticks: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Predicted horizontal and depth error variances (m^2), averaged over the evaluation
    times, for each information matrix (...,7,7)."""
    cov = np.linalg.inv(info)
    horizontal = np.zeros(info.shape[:-2])
    depth = np.zeros(info.shape[:-2])
    for t in eval_ticks:
        phi = np.zeros((3, 7))
        phi[:, 0:3] = np.eye(3)
        phi[:, 3:6] = np.eye(3) * t
        p = np.einsum("ij,...jk,lk->...il", phi, cov, phi)
        horizontal = horizontal + p[..., 0, 0] + p[..., 1, 1]
        depth = depth + p[..., 2, 2]
    return horizontal / len(eval_ticks), depth / len(eval_ticks)


def _cost(info: np.ndarray, eval_ticks: list[float], depth_weight: float) -> np.ndarray:
    """Planning cost: horizontal + depth_weight x depth error variance (m^2)."""
    horizontal, depth = _components(info, eval_ticks)
    return horizontal + depth_weight * depth


def _prior(estimate: TrackEstimate) -> np.ndarray:
    u = estimate.uncertainty
    sh = max(u.horizontal_major_yd * YD_TO_M, 5.0)
    sz = max(u.depth_sigma_ft * FT_TO_M, 5.0)
    sv = max(u.through_water_speed_sigma_kt * KNOT_TO_MPS, 0.05)
    sigmas = np.array([sh, sh, sz, sv, sv, 0.1, max(u.bias_sigma_hz, 0.01)])
    return np.diag(1.0 / sigmas**2)


def plan_optimal_deployment(
    estimate: TrackEstimate,
    observers: list[Position],
    pending: list[Position],
    config: ForwardDeploymentConfig,
    max_slant_range_yd: float,
    free_slots: int,
    source_frequency_hz: float = 400.0,
    sound_speed_mps: float = 1500.0,
    frequency_sigma_hz: float = 0.03,
    coverage_short: bool = False,
    force: bool = False,
) -> tuple[list[Position], str, PlanReport | None]:
    r_max = max_slant_range_yd * YD_TO_M
    origin = estimate.current_position
    depth_est = (estimate.depth_ft if estimate.depth_ft is not None else origin.depth_ft) * FT_TO_M
    speed = (estimate.through_water_speed_kt or 0.0) * KNOT_TO_MPS
    heading = math.radians(estimate.hdg_deg)
    limit = min(config.max_per_drop, max(free_slots, 0))
    if limit <= 0:
        return [], "no free observer slot", None
    horizon = float(config.horizon_s)
    ticks = np.arange(0.0, horizon + 1e-9, SAMPLE_S)
    eval_ticks = [float(min(config.lead_time_s, horizon)), horizon]
    p0 = np.array([0.0, 0.0, depth_est])
    existing = np.array(
        [[*local_offset_m(origin, p)[0:2], p.depth_ft * FT_TO_M] for p in [*observers, *pending]]
    ).reshape(-1, 3)

    # candidate grid (water frame): ahead along the predicted track and across it, x depths
    ux, uy = math.sin(heading), math.cos(heading)
    travel = speed * horizon
    along = np.arange(0.25 * r_max, max(travel, 0.5 * r_max) + 1e-9, 0.25 * r_max)
    across = np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]) * r_max
    depths = np.array(sorted(set(config.depth_options_ft))) * FT_TO_M
    grid = np.array([
        [ux * a + uy * x, uy * a - ux * x, z] for a in along for x in across for z in depths
    ])

    hdg_sigma = math.radians(min(max(estimate.uncertainty.hdg_sigma_deg, 3.0), 30.0))
    hypotheses = [heading, heading - hdg_sigma, heading + hdg_sigma]
    base_info, cand_info = [], []
    prior = _prior(estimate)
    for h in hypotheses:
        v = np.array([math.sin(h) * speed, math.cos(h) * speed, 0.0])
        info = prior.copy()
        if len(existing):
            info += _fisher(existing, p0, v, ticks, source_frequency_hz, sound_speed_mps,
                            frequency_sigma_hz, r_max).sum(axis=0)
        base_info.append(info)
        cand_info.append(_fisher(grid, p0, v, ticks, source_frequency_hz, sound_speed_mps,
                                 frequency_sigma_hz, r_max))
    base_info = np.array(base_info)  # (H,7,7)
    cand_info = np.array(cand_info)  # (H,M,7,7)

    def mean_cost(info_h: np.ndarray) -> np.ndarray:  # (H,...,7,7) -> (...)
        return np.mean(_cost(info_h, eval_ticks, config.depth_weight), axis=0)

    current = base_info.copy()
    cost_before = float(mean_cost(current))
    cost_now = cost_before
    chosen: list[int] = []
    available = np.ones(len(grid), dtype=bool)
    first_gain = 0.0
    target = (config.target_error_yd * YD_TO_M) ** 2
    lanes = np.round(np.array([
        -uy * g[0] + ux * g[1] for g in grid
    ]) / (0.25 * r_max)).astype(int)  # cross-track lane index (left positive)
    for _ in range(limit):
        if chosen and cost_now <= target:
            break
        if not chosen and not (force or coverage_short) and cost_now <= target:
            break  # the existing field already tracks well enough over the horizon
        trial = mean_cost(current[:, None, :, :] + cand_info)  # (M,)
        trial = np.where(available, trial, np.inf)
        best = int(np.argmin(trial))
        gain = (cost_now - float(trial[best])) / max(cost_now, 1e-12)
        if not chosen:
            first_gain = gain
            if not (force or coverage_short) and gain < config.trigger_gain:
                break
        elif gain < config.min_relative_gain:
            break
        chosen.append(best)
        current = current + cand_info[:, best]
        cost_now = float(trial[best])
        # one observer per lane (no collinear line of new observers, no stacking)
        available &= lanes != lanes[best]

    h0, d0 = (float(np.mean(x)) for x in _components(base_info, eval_ticks))
    h1, d1 = (float(np.mean(x)) for x in _components(current, eval_ticks))
    report = PlanReport(
        horizontal_before_yd=math.sqrt(h0) / YD_TO_M, horizontal_after_yd=math.sqrt(h1) / YD_TO_M,
        depth_before_ft=math.sqrt(d0) / FT_TO_M, depth_after_ft=math.sqrt(d1) / FT_TO_M,
        count=len(chosen),
        depths_ft=[round(grid[i, 2] / FT_TO_M) for i in chosen],
        cost_before_yd=math.sqrt(cost_before) / YD_TO_M,
        cost_after_yd=math.sqrt(cost_now) / YD_TO_M,
        first_gain=first_gain,
        candidates=len(grid),
    )
    if not chosen:
        if cost_before <= target:
            return [], (
                f"optimal: predicted error (horizontal {report.horizontal_before_yd:.0f} YD, depth "
                f"{report.depth_before_ft:.0f} Ft) already within {config.target_error_yd:.0f} YD"
            ), report
        return [], (
            f"optimal: best new observer would improve the predicted error by only "
            f"{first_gain * 100:.0f} % (< {config.trigger_gain * 100:.0f} %)"
        ), report
    positions = [_offset(origin, float(grid[i, 0]), float(grid[i, 1]), float(grid[i, 2] / FT_TO_M)) for i in chosen]
    why = "operator request" if force else ("coverage" if coverage_short else "information gain")
    reason = (
        f"optimal ({why}): {len(chosen)} observers, depths {report.depths_ft} Ft; predicted error "
        f"horizontal {report.horizontal_before_yd:.0f} -> {report.horizontal_after_yd:.0f} YD, "
        f"depth {report.depth_before_ft:.0f} -> {report.depth_after_ft:.0f} Ft"
    )
    return positions, reason, report
