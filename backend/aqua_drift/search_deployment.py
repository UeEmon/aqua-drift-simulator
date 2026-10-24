"""Re-detection (再探知) search drops after a lost contact (失探知).

While no observer detects the target the estimator coasts its particle cloud (non-detection
pushes the particles out of every detection circle), so the mean and the track-based forward
planner (aqua_drift.forward_deployment) are of no use: the estimate is LOST. The search plan
instead covers the cloud itself: drop points are chosen greedily so that detection discs of
radius search_radius_fraction x R_max centred on them together contain the largest share of the
particle mass at the time the observers will be in the water, until search_pd of the mass is
covered or search_max_drops drops are planned.

Geometry is in the water frame, like the forward planner: observers drift with the water and
the target is moved by the same water, so only the through-water velocity moves a particle
relative to a drop point. The cloud comes from the estimator (metadata["search_cloud"], relative
to the estimate's position) -- the estimate only, never the simulator truth.
"""
from __future__ import annotations

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position, TrackEstimate

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


def search_points(cloud: dict[str, list[float]], t_drop_s: float, radius_m: float, pd: float,
                  max_drops: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Greedy disc cover. Returns (drop points east/north relative to the cloud's centre, the
    share of the cloud's mass they cover). A drop point is where the target is expected at the
    time of the drop: cloud position + through-water velocity x t_drop."""
    east = np.asarray(cloud["east_m"], dtype=float)
    north = np.asarray(cloud["north_m"], dtype=float)
    predicted = np.stack([east + np.asarray(cloud["vel_east_mps"]) * t_drop_s,
                          north + np.asarray(cloud["vel_north_mps"]) * t_drop_s], axis=1)
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


def plan_search_drops(estimate: TrackEstimate | None, config: ForwardDeploymentConfig,
                      max_slant_range_yd: float, count: int, ready_in_s: float,
                      ) -> tuple[list[Position], str]:
    """Return (drop positions, reason); an empty list means no search drops. `count` is the number
    of drops that may be planned (free slots), `ready_in_s` the time until the layer is free."""
    if not (config.enabled and config.search_enabled):
        return [], "search disabled"
    if estimate is None or estimate.current_position is None:
        return [], "no estimate"
    if estimate.observability_status != "LOST":
        return [], f"contact not lost ({estimate.observability_status})"
    cloud = (estimate.metadata or {}).get("search_cloud")
    if not cloud or not cloud.get("east_m"):
        return [], "no particle cloud"
    count = min(count, config.search_max_drops)
    if count <= 0:
        return [], "no free observer"
    t_drop = ready_in_s + config.search_transit_s + config.search_fall_s
    rng = np.random.default_rng(estimate.tick)
    points, covered = search_points(
        cloud, t_drop, config.search_radius_fraction * max_slant_range_yd * YD_TO_M,
        config.search_pd, count, rng,
    )
    if not len(points):
        return [], "no point covers the cloud"
    depth = float(np.clip(estimate.depth_ft if estimate.depth_ft is not None else 200.0,
                          MIN_DEPTH_FT, MAX_DEPTH_FT))
    positions = [_offset(estimate.current_position, float(e), float(n), depth) for e, n in points]
    reason = f"search after lost contact ({100 * covered:.0f} % of the cloud, {len(positions)} drops)"
    return positions, reason
