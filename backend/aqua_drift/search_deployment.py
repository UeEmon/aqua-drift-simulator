"""Re-detection (再探知) search drops after a lost contact (失探知).

While no observer detects the target the estimator coasts its particle cloud (non-detection
pushes the particles out of every detection circle), so the mean and the track-based forward
planner (aqua_drift.forward_deployment) are of no use: the estimate is LOST. The search plan
instead covers the cloud itself: drop points are chosen greedily so that detection discs of
radius search_radius_fraction x R_max centred on them together contain the largest share of the
particle mass at the time the observers will be in the water, until search_pd of the mass is
covered or search_max_drops drops are planned.

With the layer's availability (optimal_deployment.LayerAvailability) the drops are planned along
its route (search_points_routed): each next drop is chosen among points the layer can reach from
the previous one, the cloud predicted at the time it gets there, and planned for that time. All
drops at one time, as before, were laid one after another several minutes apart (a loop at the
turn limit each), up to ~500 s late and where the cloud had been, ~2 km off.

Geometry is in the water frame, like the forward planner: observers drift with the water and
the target is moved by the same water, so only the through-water velocity moves a particle
relative to a drop point. The cloud comes from the estimator (metadata["search_cloud"], relative
to the estimate's position) -- the estimate only, never the simulator truth.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position, TrackEstimate
from aqua_drift.optimal_deployment import LayerAvailability
from aqua_drift.physics import local_offset_m

YD_TO_M = 0.9144
FT_TO_M = 0.3048
MIN_DEPTH_FT = 50.0
MAX_DEPTH_FT = 1500.0
MIN_GAIN = 1e-3  # a disc that adds less probability mass than this is not worth a drop
CANDIDATES = 400  # cloud particles tried as disc centres


def cloud_from_particles(x: np.ndarray, w: np.ndarray, centre: np.ndarray, count: int,
                         rng: np.random.Generator) -> dict[str, list[float]]:
    """The estimator's side: `count` equally weighted samples of the cloud (positions relative to
    centre, through-water velocity), rounded to keep the estimate small."""
    index = rng.choice(len(w), size=min(count, len(w)), p=w / w.sum())
    return {
        "east_m": np.round(x[index, 0] - centre[0], 1).tolist(),
        "north_m": np.round(x[index, 1] - centre[1], 1).tolist(),
        "vel_east_mps": np.round(x[index, 3], 3).tolist(),
        "vel_north_mps": np.round(x[index, 4], 3).tolist(),
    }


def predicted_positions(cloud: dict[str, list[float]], t_drop_s: float, maneuver_fraction: float,
                        gap_s: float, rng: np.random.Generator) -> np.ndarray:
    base, rate = _prediction(cloud, maneuver_fraction, gap_s, rng)
    return base + rate * t_drop_s


def _prediction(cloud: dict[str, list[float]], maneuver_fraction: float, gap_s: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Where the cloud's samples are expected at the time of the drop (east/north, m).

    A sample goes on with its through-water velocity v: position + v x t_drop. The coasted cloud
    keeps (nearly) the course held before the contact was lost, so it is wrong when the target
    maneuvered meanwhile. A share maneuver_fraction of the samples therefore is a maneuver
    hypothesis: at a time u x gap_s after the last detection the target turned by a random angle
    and changed its speed by a random factor (0.5 .. 1.5), so it moved v' instead of v for
    (1 - u) x gap_s + t_drop_s: position + (v' - v) x that time."""
    east = np.asarray(cloud["east_m"], dtype=float)
    north = np.asarray(cloud["north_m"], dtype=float)
    ve = np.asarray(cloud["vel_east_mps"], dtype=float)
    vn = np.asarray(cloud["vel_north_mps"], dtype=float)
    n = len(east)
    moved = np.zeros(n, dtype=bool)
    if maneuver_fraction > 0.0:
        moved = rng.random(n) < maneuver_fraction
    turn = rng.uniform(-np.pi, np.pi, n)
    factor = rng.uniform(0.5, 1.5, n)
    ve2 = factor * (ve * np.cos(turn) + vn * np.sin(turn))
    vn2 = factor * (-ve * np.sin(turn) + vn * np.cos(turn))
    before = (1.0 - rng.random(n)) * gap_s  # the maneuver's share of the time up to the last detection
    dve = np.where(moved, ve2 - ve, 0.0)
    dvn = np.where(moved, vn2 - vn, 0.0)
    # (base, rate): the position at t s from now is base + rate x t (linear in t)
    base = np.stack([east + dve * before, north + dvn * before], axis=1)
    rate = np.stack([ve + dve, vn + dvn], axis=1)
    return base, rate


def search_points(cloud: dict[str, list[float]], t_drop_s: float, radius_m: float, pd: float,
                  max_drops: int, rng: np.random.Generator, maneuver_fraction: float = 0.0,
                  gap_s: float = 0.0) -> tuple[np.ndarray, float]:
    """Greedy disc cover. Returns (drop points east/north relative to the cloud's centre, the
    share of the cloud's mass they cover). A drop point is where the target is expected at the
    time of the drop: cloud position + through-water velocity x t_drop, with a share
    maneuver_fraction of the samples being maneuver hypotheses (predicted_positions)."""
    predicted = predicted_positions(cloud, t_drop_s, maneuver_fraction, gap_s, rng)
    n = len(predicted)
    if n == 0 or max_drops <= 0:
        return np.empty((0, 2)), 0.0
    left = np.full(n, 1.0 / n)
    candidates = predicted[rng.choice(n, size=min(CANDIDATES, n), replace=False)]
    chosen: list[np.ndarray] = []
    covered = 0.0
    for _ in range(max_drops):
        inside = np.linalg.norm(predicted[None, :, :] - candidates[:, None, :], axis=2) <= radius_m
        gain = inside @ left
        best = int(np.argmax(gain))
        if gain[best] <= MIN_GAIN:
            break
        chosen.append(candidates[best])
        covered += float(gain[best])
        left[inside[best]] = 0.0
        if covered >= pd:
            break
    return (np.array(chosen) if chosen else np.empty((0, 2))), covered


def search_points_routed(cloud: dict[str, list[float]], layer: LayerAvailability, origin: Position,
                         fall_s: float, radius_m: float, pd: float, max_drops: int, rng: np.random.Generator,
                         maneuver_fraction: float = 0.0, gap_s: float = 0.0,
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Greedy disc cover along the layer's route. Returns (drop points east/north relative to
    origin, the cloud's centre; their drop times s from now; the share of the cloud covered).

    From where and when the layer is free, each candidate (a cloud sample where it will be when the
    layer can get there, two fixed-point steps) is reached at layer.earliest_s; the cover is counted
    with the cloud predicted at that time plus the fall fall_s. The best becomes the next drop and
    the layer goes on from it (heading: the leg flown to it)."""
    base, rate = _prediction(cloud, maneuver_fraction, gap_s, rng)
    n = len(base)
    if n == 0 or max_drops <= 0:
        return np.empty((0, 2)), np.empty(0), 0.0
    left = np.full(n, 1.0 / n)
    pick = rng.choice(n, size=min(CANDIDATES, n), replace=False)
    here = layer
    previous = np.array(local_offset_m(origin, layer.position)[0:2])
    chosen: list[np.ndarray] = []
    times: list[float] = []
    covered = 0.0
    for _ in range(max_drops):
        t = np.full(len(pick), here.ready_s)
        for _ in range(2):  # where the sample will be when the layer can be there
            points = base[pick] + rate[pick] * (t + fall_s)[:, None]
            t = here.earliest_s(origin, points)
        points = base[pick] + rate[pick] * (t + fall_s)[:, None]
        at = base[None, :, :] + rate[None, :, :] * (t + fall_s)[:, None, None]  # (C,N,2)
        inside = np.linalg.norm(at - points[:, None, :], axis=2) <= radius_m
        gain = inside @ left
        best = int(np.argmax(gain))
        if gain[best] <= MIN_GAIN or not math.isfinite(t[best]):
            break
        chosen.append(points[best])
        times.append(float(t[best]))
        covered += float(gain[best])
        left[inside[best]] = 0.0
        if covered >= pd:
            break
        leg = points[best] - previous
        heading = math.degrees(math.atan2(leg[0], leg[1])) % 360.0 if np.hypot(*leg) > 1.0 else here.heading_deg
        here = replace(here, ready_s=float(t[best]), position=_offset(origin, float(points[best][0]),
                                                                      float(points[best][1]), 0.0),
                       heading_deg=heading, queue=[(here.position, float(t[best]))])
        previous = points[best]
    return (np.array(chosen) if chosen else np.empty((0, 2))), np.array(times), covered


def plan_search_drops(estimate: TrackEstimate | None, config: ForwardDeploymentConfig,
                      max_slant_range_yd: float, count: int, ready_in_s: float,
                      layer: LayerAvailability | None = None,
                      ) -> tuple[list[Position], str, list[float]]:
    """Return (drop positions, reason, drop times s from now); an empty list means no search drops.
    `count` is the number of drops that may be planned (free slots), `ready_in_s` the time until
    the layer is free. With `layer` the drops are routed (search_points_routed) and timed; without
    it they are planned for one time, ready_in_s + search_transit_s + search_fall_s (no times)."""
    if not (config.enabled and config.search_enabled):
        return [], "search disabled", []
    if estimate is None or estimate.current_position is None:
        return [], "no estimate", []
    if estimate.observability_status != "LOST":
        return [], f"contact not lost ({estimate.observability_status})", []
    cloud = (estimate.metadata or {}).get("search_cloud")
    if not cloud or not cloud.get("east_m"):
        return [], "no particle cloud", []
    count = min(count, config.search_max_drops)
    if count <= 0:
        return [], "no free observer", []
    rng = np.random.default_rng(estimate.tick)
    gap = float((estimate.metadata or {}).get("seconds_since_detection") or 0.0)
    radius = config.search_radius_fraction * max_slant_range_yd * YD_TO_M
    times: list[float] = []
    if layer is not None:
        points, drop_s, covered = search_points_routed(
            cloud, layer, estimate.current_position, config.search_fall_s, radius, config.search_pd, count,
            rng, config.search_maneuver_fraction, gap,
        )
        times = [float(x) for x in drop_s]
    else:
        t_drop = ready_in_s + config.search_transit_s + config.search_fall_s
        points, covered = search_points(cloud, t_drop, radius, config.search_pd, count, rng,
                                        config.search_maneuver_fraction, gap)
    if not len(points):
        return [], "no point covers the cloud", []
    depth = float(np.clip(estimate.depth_ft if estimate.depth_ft is not None else 200.0,
                          MIN_DEPTH_FT, MAX_DEPTH_FT))
    positions = [_offset(estimate.current_position, float(e), float(n), depth) for e, n in points]
    reason = f"search after lost contact ({100 * covered:.0f} % of the cloud, {len(positions)} drops)"
    return positions, reason, times
