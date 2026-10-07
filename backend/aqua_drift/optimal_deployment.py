"""Optimal automatic observer deployment (positions, number, depths and drop times), robust
to target maneuvers and using every measurement the tracker uses.

Decides from the ESTIMATE ONLY (never the simulator truth). Geometry is planned in the water
frame: observers drift with the water and the target is carried by the same water, so relative
to the observers the target moves with its through-water velocity and a dropped observer stays
where it is dropped.

Motion hypotheses. The target may change course, speed or depth within the planning horizon,
so the plan is evaluated over a weighted set of piecewise constant-velocity trajectories: the
estimated track (and its heading +-1 sigma) and, with a total weight `maneuver_weight`, course
changes (+-45 / +-90 deg), speed changes (x0.5 / x1.5) and depth changes (+-500 Ft) starting
`maneuver_time_s` from now (planning rates 1 deg/s, 0.1 kt/s, 2 Ft/s).

Information. Along each trajectory the posterior Cramer-Rao bound of the state
theta = [p, v, b] (position, through-water velocity, common frequency bias) is propagated
recursively every STEP_S seconds (Tichavsky et al.):

    J(k) = (F J(k-1)^-1 F^T + Q)^-1 + sum_o I_o(k)

with the white-acceleration process noise Q (`process_velocity_sigma_kt` /
`process_depth_rate_sigma_fps` of velocity drift per 10 min), so old measurements lose their
value and a field that keeps observing the target after a maneuver is preferred to one that
only fixed the past. Per observer and step (only while it is in the water and detecting,
slant range <= R_max):

    Doppler   f = f0 (1 - rdot/c), d rdot/d p = (v - rdot u)/r, d rdot/d v = u, d f/d b = -1
              (1 s samples, sigma_f)
    bearing   beta = atan2(dx, dy), d beta/d p = [dy, -dx, 0] / rho^2
              (every bearing_interval_s, sigma_beta; when the tracker uses bearings)
    gate      at the start / end of detection the slant range equals R_max: d r/d p = u
              (sigma = the tracker's range-gate softness)

Cost (m^2). For each hypothesis: the mean predicted position error variance (horizontal +
depth_weight x depth) over the steps from lead_time_s to the horizon, plus coverage_loss_yd^2
times the fraction of those steps with fewer than min_coverage detecting observers. Over the
hypotheses: the weighted mean + risk_weight x the weighted mean of the worst risk_quantile
(CVaR), so a plan that only works if the target holds its course is penalised. Plus the mirror
term: observers close to one line leave a mirror image of the track about that line that the
local information cannot see; the expected squared error P(confuse) x |p - p_mirror|^2 is
added, with P(confuse) = Phi(-sqrt(D)/2) and D the discrimination (Doppler with the common
bias removed, and bearings) between the track and its mirror.

Candidates: a grid in the track frame ahead of the estimate and to both sides (wide enough for
the turn hypotheses), x the observer depth options, kept only where some hypothesis comes
within detection range. Greedy selection: add the candidate that lowers the cost most; stop
when the predicted error reaches target_error_yd, when the next one would improve the cost by
less than min_relative_gain, or at max_per_drop / the free observer slots. New observers keep
0.3 R_max apart.

Drop time: an observer measures only once it is in the water, so each candidate's information
counts from the earliest time the layer (設標者) can be there (its free time, flight distance and
turn). For every chosen observer the optimal drop time is as late as possible without losing
information -- just before the target (earliest over the hypotheses, less a lead time and a
position-uncertainty margin) comes within the detection range -- and never before the layer can
be there. A later drop keeps the observer's 3 h observing time and its slot for when they are
useful.

Trigger: deploy when the predicted position after lead_time_s is not covered by enough
observers, or when the predicted error exceeds target_error_yd and the best new observer would
improve it by at least trigger_gain (the existing field will not track the target well ahead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position, TrackEstimate
from aqua_drift.physics import local_offset_m

YD_TO_M = 0.9144
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
FINE_S = 10.0  # trajectory resolution
STEP_S = 60.0  # information recursion step
TURN_RATE = math.radians(1.0)  # planning assumptions for the maneuver hypotheses
SPEED_RATE = 0.1 * KNOT_TO_MPS
DEPTH_RATE = 2.0 * FT_TO_M
MIN_DEPTH_M = 50.0 * FT_TO_M
MIRROR_DOPPLER_SIGMA_HZ = 0.1  # model-error floor for the mirror discrimination (one per step)
MIN_SEPARATION = 0.3  # x R_max between new observers


@dataclass
class PlanReport:
    count: int
    depths_ft: list[float]
    cost_before_yd: float  # predicted error (1 sigma equivalent, YD) of the full cost
    cost_after_yd: float
    first_gain: float
    candidates: int
    horizontal_before_yd: float = 0.0
    horizontal_after_yd: float = 0.0
    depth_before_ft: float = 0.0
    depth_after_ft: float = 0.0
    drop_times_s: list[float] = field(default_factory=list)  # optimal drop time (s from now)
    coverage_loss_before: float = 0.0  # weighted fraction of the horizon with < min_coverage
    coverage_loss_after: float = 0.0
    hypotheses: int = 0


@dataclass
class SensorModel:
    """What the tracker measures (the planner values exactly these measurements)."""

    source_frequency_hz: float = 400.0
    sound_speed_mps: float = 1500.0
    frequency_sigma_hz: float = 0.03
    use_bearing: bool = True
    bearing_sigma_deg: float = 15.0
    bearing_interval_s: float = 15.0
    use_gate: bool = True
    gate_sigma_yd: float = 15.0


@dataclass
class Hypothesis:
    weight: float
    pos: np.ndarray  # (T,3) every FINE_S, water frame (m, depth positive down)
    vel: np.ndarray  # (T,3)
    name: str = ""


def _trajectory(
    p0: np.ndarray, heading: float, speed: float, vz: float, horizon: float, max_depth: float,
    change_s: float = math.inf, turn: float = 0.0, speed_to: float | None = None,
    depth_to: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ticks = np.arange(0.0, horizon + 1e-9, FINE_S)
    pos = np.zeros((len(ticks), 3))
    vel = np.zeros((len(ticks), 3))
    p = p0.astype(float).copy()
    h, s = heading, speed
    goal_h = heading + turn
    for i, t in enumerate(ticks):
        w = vz
        if t >= change_s:
            h += float(np.clip(goal_h - h, -TURN_RATE * FINE_S, TURN_RATE * FINE_S))
            if speed_to is not None:
                s += float(np.clip(speed_to - s, -SPEED_RATE * FINE_S, SPEED_RATE * FINE_S))
            w = 0.0
            if depth_to is not None:
                w = float(np.clip((depth_to - p[2]) / FINE_S, -DEPTH_RATE, DEPTH_RATE))
        if (p[2] <= MIN_DEPTH_M and w < 0) or (p[2] >= max_depth and w > 0):
            w = 0.0
        pos[i] = p
        vel[i] = (math.sin(h) * s, math.cos(h) * s, w)
        p = p + vel[i] * FINE_S
    return pos, vel


def hypotheses(
    p0: np.ndarray, heading: float, speed: float, vz: float, heading_sigma: float,
    config: ForwardDeploymentConfig, horizon: float, max_depth: float,
) -> list[Hypothesis]:
    """Weighted motion hypotheses: the estimated track (heading and +-1 sigma) and, with a
    total weight config.maneuver_weight, course / speed / depth changes."""
    def make(weight: float, name: str, h: float = heading, **change) -> Hypothesis:
        pos, vel = _trajectory(p0, h, speed, vz, horizon, max_depth, **change)
        return Hypothesis(weight, pos, vel, name)

    mw = config.maneuver_weight if config.maneuver_time_s < horizon else 0.0
    straight = 1.0 - mw
    out = [
        make(0.5 * straight, "track"),
        make(0.25 * straight, "track-1s", heading - heading_sigma),
        make(0.25 * straight, "track+1s", heading + heading_sigma),
    ]
    if mw > 0:
        at = float(config.maneuver_time_s)
        dz = config.maneuver_depth_change_ft * FT_TO_M
        maneuvers = [
            ("turn-90", {"turn": -math.pi / 2}), ("turn-45", {"turn": -math.pi / 4}),
            ("turn+45", {"turn": math.pi / 4}), ("turn+90", {"turn": math.pi / 2}),
            ("slow", {"speed_to": 0.5 * speed}), ("fast", {"speed_to": 1.5 * speed}),
            ("up", {"depth_to": max(p0[2] - dz, MIN_DEPTH_M)}),
            ("down", {"depth_to": min(p0[2] + dz, max_depth)}),
        ]
        out += [make(mw / len(maneuvers), name, change_s=at, **change) for name, change in maneuvers]
    return out


def _measurement_info(
    obs: np.ndarray,  # (M,3)
    pos: np.ndarray,  # (H,K,3) target at the recursion steps
    vel: np.ndarray,  # (H,K,3)
    steps: np.ndarray,  # (K,) s from now
    start: np.ndarray,  # (M,) s from now when each observer is in the water
    sensor: SensorModel,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(H,M,K,7,7) information of each observer at each step and (H,M,K) detection flags."""
    rel = pos[:, None, :, :] - obs[None, :, None, :]  # (H,M,K,3)
    r = np.maximum(np.linalg.norm(rel, axis=3), 1.0)
    u = rel / r[..., None]
    v = vel[:, None, :, :]
    rdot = np.sum(u * v, axis=3)
    present = steps[None, :] >= start[:, None]  # (M,K)
    inside = r <= max_range
    det = inside & present[None]
    scale = -sensor.source_frequency_hz / sensor.sound_speed_mps
    drdp = (v - rdot[..., None] * u) / r[..., None]
    g = np.concatenate([scale * drdp, scale * u, -np.ones(r.shape + (1,))], axis=3)
    w = det * (STEP_S / sensor.frequency_sigma_hz**2)
    info = np.einsum("hmk,hmki,hmkj->hmkij", w, g, g)
    if sensor.use_bearing:
        rho2 = np.maximum(rel[..., 0] ** 2 + rel[..., 1] ** 2, 1.0)
        gb = np.zeros(r.shape + (7,))
        gb[..., 0] = rel[..., 1] / rho2
        gb[..., 1] = -rel[..., 0] / rho2
        wb = det * (STEP_S / sensor.bearing_interval_s / math.radians(sensor.bearing_sigma_deg) ** 2)
        info += np.einsum("hmk,hmki,hmkj->hmkij", wb, gb, gb)
    if sensor.use_gate:
        both = present[:, 1:] & present[:, :-1]  # (M,K-1)
        cross = np.zeros_like(det)
        cross[..., 1:] = (inside[..., 1:] != inside[..., :-1]) & both[None]
        gg = np.zeros(r.shape + (7,))
        gg[..., 0:3] = u
        wg = cross / (sensor.gate_sigma_yd * YD_TO_M) ** 2
        info += np.einsum("hmk,hmki,hmkj->hmkij", wg, gg, gg)
    return info, det


def _process(config: ForwardDeploymentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity transition F and white-acceleration noise Q for one STEP_S."""
    dt = STEP_S
    F = np.eye(7)
    F[0:3, 3:6] = np.eye(3) * dt
    qh = (config.process_velocity_sigma_kt * KNOT_TO_MPS) ** 2 / 600.0
    qv = (config.process_depth_rate_sigma_fps * FT_TO_M) ** 2 / 600.0
    Q = np.zeros((7, 7))
    for axis, q in ((0, qh), (1, qh), (2, qv)):
        Q[axis, axis] = q * dt**3 / 3
        Q[axis, axis + 3] = Q[axis + 3, axis] = q * dt**2 / 2
        Q[axis + 3, axis + 3] = q * dt
    Q[6, 6] = 1e-10
    return F, Q


def _recursion(
    prior: np.ndarray, meas: np.ndarray, F: np.ndarray, Q: np.ndarray, evaluate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior CRLB along the steps. prior (7,7), meas (...,K,7,7). Returns the horizontal and
    depth error variances (m^2) averaged over the evaluated steps, shape (...)."""
    horizontal = np.zeros(meas.shape[:-3])
    depth = np.zeros(meas.shape[:-3])
    P = None
    for k in range(meas.shape[-3]):
        if P is None:
            J = prior + meas[..., k, :, :]
        else:
            J = np.linalg.inv(F @ P @ F.T + Q) + meas[..., k, :, :]
        P = np.linalg.inv(J)
        if evaluate[k]:
            horizontal += P[..., 0, 0] + P[..., 1, 1]
            depth += P[..., 2, 2]
    n = max(int(evaluate.sum()), 1)
    return horizontal / n, depth / n


def _combine(cost: np.ndarray, weights: np.ndarray, risk_weight: float, quantile: float) -> np.ndarray:
    """Weighted mean over the hypotheses (axis 0) + risk_weight x CVaR of the worst quantile."""
    mean = np.tensordot(weights, cost, axes=(0, 0))
    if risk_weight <= 0:
        return mean
    order = np.argsort(-cost, axis=0)
    c = np.take_along_axis(cost, order, axis=0)
    w = weights[order]
    before = np.cumsum(w, axis=0) - w
    part = np.clip(quantile - before, 0.0, w)
    return mean + risk_weight * np.sum(part * c, axis=0) / quantile


def _mirror_cost(
    sets: np.ndarray,  # (M,N,3) observer set per candidate (padded with NaN)
    start: np.ndarray,  # (M,N)
    pos: np.ndarray, vel: np.ndarray,  # (K,3) central hypothesis
    steps: np.ndarray, evaluate: np.ndarray, sensor: SensorModel, max_range: float,
) -> np.ndarray:
    """Expected squared error (m^2) from the mirror image of the track about the principal
    axis of each observer set (only observers that detect the track count)."""
    M = sets.shape[0]
    valid = np.isfinite(sets[..., 0])
    xy = np.where(valid[..., None], sets[..., 0:2], 0.0)
    rel = pos[None, None, :, :] - np.where(valid[..., None], sets, 0.0)[:, :, None, :]  # (M,N,K,3)
    r = np.maximum(np.linalg.norm(rel, axis=3), 1.0)
    det = (r <= max_range) & valid[..., None] & (steps[None, None, :] >= start[..., None])
    used = det.any(axis=2)  # (M,N)
    count = used.sum(axis=1)
    centre = np.sum(xy * used[..., None], axis=1) / np.maximum(count, 1)[:, None]
    d = (xy - centre[:, None, :]) * used[..., None]
    cov = np.einsum("mni,mnj->mij", d, d)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, :, 1]  # (M,2) principal direction
    R = 2.0 * axis[:, :, None] * axis[:, None, :] - np.eye(2)[None]
    mp = pos[None, :, 0:2] - centre[:, None, :]
    mirror_xy = centre[:, None, :] + np.einsum("mij,mkj->mki", R, mp)
    mirror_v = np.einsum("mij,kj->mki", R, vel[:, 0:2])
    mpos = np.concatenate([mirror_xy, np.broadcast_to(pos[None, :, 2:3], (M, len(pos), 1))], axis=2)
    mvel = np.concatenate([mirror_v, np.broadcast_to(vel[None, :, 2:3], (M, len(vel), 1))], axis=2)
    mrel = mpos[:, None, :, :] - np.where(valid[..., None], sets, 0.0)[:, :, None, :]
    mr = np.maximum(np.linalg.norm(mrel, axis=3), 1.0)
    k = sensor.source_frequency_hz / sensor.sound_speed_mps
    f_true = -k * np.sum(rel / r[..., None] * vel[None, None], axis=3)
    f_mirror = -k * np.sum(mrel / mr[..., None] * mvel[:, None], axis=3)
    diff = np.where(det, f_true - f_mirror, 0.0)
    n_det = np.maximum(det.sum(axis=(1, 2)), 1)
    diff = np.where(det, diff - (diff.sum(axis=(1, 2)) / n_det)[:, None, None], 0.0)  # bias absorbs
    D = np.sum(diff**2, axis=(1, 2)) / MIRROR_DOPPLER_SIGMA_HZ**2
    if sensor.use_bearing:
        b_true = np.arctan2(rel[..., 0], rel[..., 1])
        b_mirror = np.arctan2(mrel[..., 0], mrel[..., 1])
        db = (b_true - b_mirror + np.pi) % (2 * np.pi) - np.pi
        D += np.sum(np.where(det, db, 0.0) ** 2, axis=(1, 2)) * (
            STEP_S / sensor.bearing_interval_s / math.radians(sensor.bearing_sigma_deg) ** 2)
    confuse = 0.5 * erfc(np.sqrt(D) / 2.0 / math.sqrt(2.0))
    gap = np.sum((pos[None, :, 0:2] - mpos[:, :, 0:2]) ** 2, axis=2)  # (M,K)
    gap = np.mean(gap[:, evaluate], axis=1) if evaluate.any() else np.zeros(M)
    return np.where(count >= 2, confuse * gap, 0.0)


@dataclass
class LayerAvailability:
    """When / where the layer (設標者) is free for a new drop and how it flies."""

    ready_s: float  # seconds from now until the layer is free (after its open tasks)
    position: Position  # where it is then
    speed_kt: float
    max_bank_deg: float
    heading_deg: float | None = None  # its heading then (None: unknown -> an average half turn)

    def earliest_s(self, origin: Position, points: np.ndarray) -> np.ndarray:
        """Earliest drop time (s from now) at each point (water-frame metres from origin):
        free time + straight flight + the turn towards the point at the bank limit (a full
        loop more when the point is close behind)."""
        start = np.array(local_offset_m(origin, self.position)[0:2])
        v = self.speed_kt * KNOT_TO_MPS
        omega = 9.80665 * math.tan(math.radians(self.max_bank_deg)) / v
        radius = v / omega
        delta = points[:, 0:2] - start[None, :]
        distance = np.linalg.norm(delta, axis=1)
        if self.heading_deg is None:
            turn = np.full(len(points), 0.5 * math.pi / omega)
        else:
            bearing = np.arctan2(delta[:, 0], delta[:, 1])
            error = np.abs((bearing - math.radians(self.heading_deg) + math.pi) % (2 * math.pi) - math.pi)
            turn = error / omega + np.where((distance < 2 * radius) & (error > 0.5 * math.pi), math.pi / omega, 0.0)
        return self.ready_s + distance / v + turn


def drop_times(
    chosen_points: np.ndarray,  # (K,3)
    earliest: np.ndarray,  # (K,)
    trajectories: list[np.ndarray],  # (T,3) every FINE_S
    max_range: float,
    lead_s: float,
    position_sigma_m: float,
    speed: float,
) -> np.ndarray:
    """Optimal drop time of each chosen observer (s from now): as late as possible without
    losing information -- just before the target (earliest over the hypotheses, less a
    position-uncertainty margin) comes within the detection range -- but not before the layer
    can be there. A later drop keeps the observer's 3 h observing time and its slot for when
    they are useful and lets the plan use the latest estimate."""
    enter = np.full(len(chosen_points), np.inf)
    for pos in trajectories:
        r = np.linalg.norm(pos[None, :, :] - chosen_points[:, None, :], axis=2)  # (K,T)
        inside = r <= max_range
        first = np.where(inside.any(axis=1), inside.argmax(axis=1) * FINE_S, np.inf)
        enter = np.minimum(enter, first)
    margin = lead_s + 2.0 * position_sigma_m / max(speed, 0.1)
    return np.maximum(earliest, np.where(np.isfinite(enter), enter - margin, earliest))


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
    layer: LayerAvailability | None = None,
    sensor: SensorModel | None = None,
    max_depth_ft: float = 1500.0,
) -> tuple[list[Position], str, PlanReport | None]:
    """Plan positions, number and depths (and, in report.drop_times_s, the optimal drop times)."""
    sensor = sensor or SensorModel()
    sensor.source_frequency_hz = source_frequency_hz
    sensor.sound_speed_mps = sound_speed_mps
    sensor.frequency_sigma_hz = frequency_sigma_hz
    sensor.use_gate = sensor.use_gate and config.use_detection_gate
    r_max = max_slant_range_yd * YD_TO_M
    origin = estimate.current_position
    depth_est = (estimate.depth_ft if estimate.depth_ft is not None else origin.depth_ft) * FT_TO_M
    speed = (estimate.through_water_speed_kt or 0.0) * KNOT_TO_MPS
    vz = (estimate.vertical_rate_fps or 0.0) * FT_TO_M
    heading = math.radians(estimate.hdg_deg)
    limit = min(config.max_per_drop, max(free_slots, 0))
    if limit <= 0:
        return [], "no free observer slot", None
    horizon = float(config.horizon_s)
    steps = np.arange(0.0, horizon + 1e-9, STEP_S)
    every = round(STEP_S / FINE_S)
    evaluate = steps >= min(float(config.lead_time_s), steps[-1])
    p0 = np.array([0.0, 0.0, depth_est])
    existing = np.array(
        [[*local_offset_m(origin, p)[0:2], p.depth_ft * FT_TO_M] for p in [*observers, *pending]]
    ).reshape(-1, 3)

    hdg_sigma = math.radians(min(max(estimate.uncertainty.hdg_sigma_deg, 3.0), 30.0))
    hyps = hypotheses(p0, heading, speed, vz, hdg_sigma, config, horizon, max_depth_ft * FT_TO_M)
    weights = np.array([h.weight for h in hyps])
    pos = np.array([h.pos[::every] for h in hyps])  # (H,K,3)
    vel = np.array([h.vel[::every] for h in hyps])

    # candidate grid in the track frame, wide enough for the turn hypotheses, x depths
    ux, uy = math.sin(heading), math.cos(heading)
    travel = speed * horizon
    along = np.arange(0.25 * r_max, max(travel, 0.5 * r_max) + 1e-9, 0.25 * r_max)
    half = max(0.75 * r_max, 0.6 * travel) if config.maneuver_weight > 0 else 0.75 * r_max
    across = np.arange(-half, half + 1e-9, 0.25 * r_max)
    depths = np.array(sorted(set(config.depth_options_ft))) * FT_TO_M
    spots = np.array([[ux * a + uy * x, uy * a - ux * x] for a in along for x in across])
    fine = np.concatenate([h.pos[:, 0:2] for h in hyps])
    near = np.array([np.min(np.linalg.norm(fine - s, axis=1)) for s in spots]) <= r_max
    spots = spots[near]
    grid = np.array([[s[0], s[1], z] for s in spots for z in depths]).reshape(-1, 3)
    if not len(grid):
        return [], "optimal: no candidate within detection range of the predicted track", None

    # an observer only measures once it is in the water: from the layer's earliest arrival
    earliest = layer.earliest_s(origin, grid) if layer is not None else np.zeros(len(grid))
    F, Q = _process(config)
    prior = _prior(estimate)
    if len(existing):
        base_meas, base_det = _measurement_info(existing, pos, vel, steps, np.zeros(len(existing)), sensor, r_max)
        base_meas, base_count = base_meas.sum(axis=1), base_det.sum(axis=1)  # (H,K,7,7), (H,K)
    else:
        base_meas = np.zeros((len(hyps), len(steps), 7, 7))
        base_count = np.zeros((len(hyps), len(steps)))
    cand_meas, cand_det = _measurement_info(grid, pos, vel, steps, earliest, sensor, r_max)
    useful = cand_det.any(axis=(0, 2))
    grid, earliest = grid[useful], earliest[useful]
    cand_meas, cand_det = cand_meas[:, useful], cand_det[:, useful]
    if not len(grid):
        return [], "optimal: the layer cannot lay an observer before the target passes", None
    loss_m2 = (config.coverage_loss_yd * YD_TO_M) ** 2

    def evaluate_cost(meas: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, ...]:
        """meas (H,...,K,7,7), count (H,...,K) -> combined cost, horizontal, depth, loss (...)."""
        h, d = _recursion(prior, meas, F, Q, evaluate)
        loss = np.mean(count[..., evaluate] < config.min_coverage, axis=-1)
        per_h = h + config.depth_weight * d + loss_m2 * loss
        total = _combine(per_h, weights, config.risk_weight, config.risk_quantile)
        return (total, np.tensordot(weights, h, axes=(0, 0)), np.tensordot(weights, d, axes=(0, 0)),
                np.tensordot(weights, loss, axes=(0, 0)))

    central = 0  # the "track" hypothesis
    detecting_existing = existing  # _mirror_cost keeps the observers that detect the track

    def mirror(chosen_idx: list[int], extra: np.ndarray | None) -> np.ndarray:
        """Mirror cost of existing + chosen (+ each extra candidate)."""
        fixed = np.vstack([detecting_existing, grid[chosen_idx]]) if chosen_idx else detecting_existing
        fixed_start = np.concatenate([np.zeros(len(detecting_existing)), earliest[chosen_idx]])
        if extra is None:
            sets, start = fixed[None], fixed_start[None]
        else:
            m = len(extra)
            sets = np.concatenate([np.broadcast_to(fixed, (m, *fixed.shape)), grid[extra][:, None, :]], axis=1)
            start = np.concatenate([np.broadcast_to(fixed_start, (m, len(fixed_start))), earliest[extra][:, None]], axis=1)
        if sets.shape[1] < 2:
            return np.zeros(sets.shape[0])
        return _mirror_cost(sets, start, pos[central], vel[central], steps, evaluate, sensor, r_max)

    current_meas, current_count = base_meas, base_count
    cost_before, h0, d0, loss0 = (float(x) for x in evaluate_cost(base_meas, base_count))
    mirror_now = float(mirror([], None)[0])
    cost_before += mirror_now
    cost_now = cost_before
    h1, d1, loss1 = h0, d0, loss0
    chosen: list[int] = []
    available = np.ones(len(grid), dtype=bool)
    first_gain = 0.0
    target = (config.target_error_yd * YD_TO_M) ** 2
    for _ in range(limit):
        if chosen and cost_now <= target:
            break
        if not chosen and not (force or coverage_short) and cost_now <= target:
            break  # the existing field already tracks well enough over the horizon
        idx = np.flatnonzero(available)
        if not len(idx):
            break
        trial, th, td, tl = evaluate_cost(
            current_meas[:, None] + cand_meas[:, idx], current_count[:, None] + cand_det[:, idx])
        trial = trial + mirror(chosen, idx)
        pick = int(np.argmin(trial))
        best = int(idx[pick])
        gain = (cost_now - float(trial[pick])) / max(cost_now, 1e-12)
        if not chosen:
            first_gain = gain
            if not (force or coverage_short) and gain < config.trigger_gain:
                break
        elif gain < config.min_relative_gain:
            break
        chosen.append(best)
        current_meas = current_meas + cand_meas[:, best]
        current_count = current_count + cand_det[:, best]
        cost_now = float(trial[pick])
        h1, d1, loss1 = float(th[pick]), float(td[pick]), float(tl[pick])
        # new observers keep apart (no stacking at one spot)
        available &= np.linalg.norm(grid[:, 0:2] - grid[best, 0:2], axis=1) >= MIN_SEPARATION * r_max

    report = PlanReport(
        horizontal_before_yd=math.sqrt(h0) / YD_TO_M, horizontal_after_yd=math.sqrt(h1) / YD_TO_M,
        depth_before_ft=math.sqrt(d0) / FT_TO_M, depth_after_ft=math.sqrt(d1) / FT_TO_M,
        count=len(chosen),
        depths_ft=[round(grid[i, 2] / FT_TO_M) for i in chosen],
        cost_before_yd=math.sqrt(cost_before) / YD_TO_M,
        cost_after_yd=math.sqrt(cost_now) / YD_TO_M,
        first_gain=first_gain,
        candidates=len(grid),
        coverage_loss_before=loss0,
        coverage_loss_after=loss1,
        hypotheses=len(hyps),
    )
    if not chosen:
        if cost_before <= target:
            return [], (
                f"optimal: predicted error {report.cost_before_yd:.0f} YD (horizontal "
                f"{report.horizontal_before_yd:.0f} YD, depth {report.depth_before_ft:.0f} Ft) "
                f"already within {config.target_error_yd:.0f} YD"
            ), report
        return [], (
            f"optimal: best new observer would improve the predicted error by only "
            f"{first_gain * 100:.0f} % (< {config.trigger_gain * 100:.0f} %)"
        ), report
    positions = [_offset(origin, float(grid[i, 0]), float(grid[i, 1]), float(grid[i, 2] / FT_TO_M)) for i in chosen]
    if config.schedule_drops:
        times = drop_times(
            grid[chosen], earliest[chosen], [h.pos for h in hyps], r_max, float(config.drop_lead_s),
            estimate.uncertainty.horizontal_major_yd * YD_TO_M, speed,
        )
    else:
        times = earliest[chosen]
    report.drop_times_s = [float(t) for t in times]
    why = "operator request" if force else ("coverage" if coverage_short else "information gain")
    reason = (
        f"optimal ({why}): {len(chosen)} observers, depths {report.depths_ft} Ft; predicted error "
        f"{report.cost_before_yd:.0f} -> {report.cost_after_yd:.0f} YD (horizontal "
        f"{report.horizontal_before_yd:.0f} -> {report.horizontal_after_yd:.0f} YD, "
        f"depth {report.depth_before_ft:.0f} -> {report.depth_after_ft:.0f} Ft, "
        f"coverage gaps {100 * loss0:.0f} -> {100 * loss1:.0f} %); "
        f"drop in {[round(t) for t in report.drop_times_s]} s"
    )
    return positions, reason, report


def availability_from_feed(feed) -> LayerAvailability | None:
    """LayerAvailability from a DeploymentFeed (None without the layer: drops are immediate)."""
    if not getattr(feed, "layer_enabled", False) or feed.layer_ready_position is None:
        return None
    return LayerAvailability(
        ready_s=max(float((feed.layer_ready_tick or feed.tick) - feed.tick), 0.0),
        position=feed.layer_ready_position,
        speed_kt=feed.layer_speed_kt,
        max_bank_deg=feed.layer_max_bank_deg,
        heading_deg=feed.layer_ready_heading_deg,
    )


def sensor_from_feed(feed) -> SensorModel:
    """SensorModel from a DeploymentFeed (what the tracker measures)."""
    return SensorModel(
        source_frequency_hz=feed.source_frequency_hz,
        sound_speed_mps=feed.sound_speed_mps,
        frequency_sigma_hz=feed.frequency_sigma_hz,
        use_bearing=feed.use_bearing,
        bearing_sigma_deg=feed.bearing_sigma_deg,
        bearing_interval_s=feed.bearing_interval_s,
        gate_sigma_yd=feed.range_gate_sigma_yd,
    )
