"""Forward (前程) observer deployment planner.

Decides from the ESTIMATE ONLY (never the simulator truth) whether and where to place new
observers ahead of the target.

Geometry is planned in the water frame. Observers drift with the water and the target is moved
by the same water, so relative to the observers the target advances with its through-water
velocity u (estimated HDG and through-water speed). After lead time T the target's position
relative to today's observer field is p_est + u T. If fewer than `min_coverage` observers
(active or already pending) lie within coverage_fraction x R_max (horizontal) of that point,
`observers_per_drop` observers are placed `ahead_distance_yd` ahead of the estimate on the
estimated heading, `lateral_offset_yd` either side of the track, so the target passes between
them (Doppler CPA on both sides, non-collinear with the existing field).
"""
from __future__ import annotations

import math

from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position, TrackEstimate
from aqua_drift.optimal_deployment import LayerAvailability, plan_optimal_deployment
from aqua_drift.physics import local_offset_m

YD_TO_M = 0.9144
KNOT_TO_MPS = 0.5144444444444445
MIN_DEPTH_FT = 50.0
MAX_DEPTH_FT = 1500.0


def plan_forward_deployment(*args, **kwargs) -> tuple[list[Position], str]:
    """Return (positions to deploy, reason); see plan_forward_deployment_scheduled."""
    positions, reason, _ = plan_forward_deployment_scheduled(*args, **kwargs)
    return positions, reason


def plan_forward_deployment_scheduled(
    tick: int,
    estimate: TrackEstimate | None,
    observers: list[Position],
    pending: list[Position],
    config: ForwardDeploymentConfig,
    max_slant_range_yd: float,
    last_deploy_tick: int | None,
    depth_step_ft: float = 150.0,
    force: bool = False,
    free_slots: int = 99,
    source_frequency_hz: float = 400.0,
    sound_speed_mps: float = 1500.0,
    frequency_sigma_hz: float = 0.03,
    layer: LayerAvailability | None = None,
) -> tuple[list[Position], str, list[int] | None]:
    """Return (positions to deploy, reason, planned drop ticks). An empty list means no
    deployment now. The planned ticks (optimal strategy) are when the layer should lay each
    observer; None = as soon as possible.
    `force` (operator request) skips the coverage and cooldown checks but still requires a
    usable estimate. With config.strategy == "optimal" the positions, number and depths are
    chosen by aqua_drift.optimal_deployment; "fixed" uses the two-sided pattern."""
    if not config.enabled and not force:
        return [], "disabled", None
    if estimate is None or estimate.current_position is None or estimate.uncertainty is None:
        return [], "no estimate", None
    if not estimate.observability_status.startswith("TRACKING"):
        return [], f"estimate not usable ({estimate.observability_status})", None
    r_max = max_slant_range_yd * YD_TO_M
    if estimate.uncertainty.horizontal_major_yd * YD_TO_M > config.max_sigma_fraction * r_max:
        return [], "estimate too uncertain", None
    if not force and last_deploy_tick is not None and tick - last_deploy_tick < config.cooldown_s:
        return [], "cooldown", None
    speed_kt = estimate.through_water_speed_kt or 0.0
    if speed_kt < config.min_speed_kt or estimate.hdg_deg is None:
        return [], "target (nearly) stationary in the water", None

    heading = math.radians(estimate.hdg_deg)
    ux, uy = math.sin(heading), math.cos(heading)
    speed = speed_kt * KNOT_TO_MPS
    origin = estimate.current_position
    lead = speed * config.lead_time_s
    predicted = (ux * lead, uy * lead)  # metres east/north of the estimate (water frame)
    radius = config.coverage_fraction * r_max
    covered = 0
    for position in [*observers, *pending]:
        east, north, _ = local_offset_m(origin, position)
        if math.hypot(east - predicted[0], north - predicted[1]) <= radius:
            covered += 1
    if config.strategy == "optimal":
        if not force and estimate.uncertainty.horizontal_major_yd * YD_TO_M > config.optimal_max_sigma_fraction * r_max:
            return [], "estimate not yet converged enough for optimal placement", None
        positions, reason, report = plan_optimal_deployment(
            estimate, observers, pending, config, max_slant_range_yd, free_slots,
            source_frequency_hz, sound_speed_mps, frequency_sigma_hz,
            coverage_short=covered < config.min_coverage, force=force, layer=layer,
        )
        if not positions and covered >= config.min_coverage:
            reason = f"covered ({covered} observers near predicted position); {reason}"
        planned = [tick + round(t) for t in report.drop_times_s] if positions and report else None
        return positions, reason, planned
    if covered >= config.min_coverage and not force:
        return [], f"covered ({covered} observers near predicted position)", None

    ahead = config.ahead_distance_yd * YD_TO_M
    lateral = config.lateral_offset_yd * YD_TO_M
    count = config.observers_per_drop
    positions: list[Position] = []
    depth0 = estimate.depth_ft if estimate.depth_ft is not None else origin.depth_ft
    for index in range(count):
        if count == 1:
            side = 0.0
        else:
            side = -1.0 + 2.0 * index / (count - 1)  # spread from left to right of the track
        # small along-track stagger so successive drops never line up exactly
        along = ahead + (0.15 * lateral if index % 2 else -0.15 * lateral)
        east = ux * along + uy * side * lateral
        north = uy * along - ux * side * lateral
        depth = depth0 + (depth_step_ft if index % 2 else -depth_step_ft)
        depth = min(max(depth, MIN_DEPTH_FT), MAX_DEPTH_FT)
        positions.append(_offset(origin, east, north, depth))
    reason = "operator request" if force else (
        f"predicted position in {config.lead_time_s} s covered by {covered} < "
        f"{config.min_coverage} observers"
    )
    return positions, reason, None
