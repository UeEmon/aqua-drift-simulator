"""Replanning of the open automatic drops as the estimate is updated, without stalling the laying.

The forward planner (aqua_drift.forward_deployment / optimal_deployment) only ever adds drops: an
open drop task keeps the point it was planned on even when the estimate has moved since. This
module replaces the open automatic drops with a fresh plan when the estimate has clearly moved
away from what they were planned on and the fresh plan is clearly better -- while making sure
that continual updates never keep the layer (設標者) from laying anything:

Kept (never replaced):
    * a drop the layer is flying to or holding at (it would otherwise turn back),
    * a drop due within replan_freeze_s (its approach is under way),
    * a drop replaced replan_max_revisions times already (each slot settles after a few revisions),
    * operator / manual drops and drops without a plan basis,
    * with manual approval, drops the operator has approved.
  So the head of the queue is always laid, whatever the estimate does.

Trigger (hysteresis 1): every automatic drop carries the estimate it was planned on (PlanBasis:
the drop point relative to the estimated target, water frame, and the estimated heading and
speed). Drop point and target drift with the same water, so the drop's expected offset from the
target now is basis offset - basis velocity x elapsed time; the estimate has moved by the
difference to the actual offset. A replan starts when for some revisable drop
    shift > max(replan_shift_sigma x 1 sigma major, replan_shift_fraction x R_max), or
    |heading change| > max(replan_heading_deg, 2 sigma), or |speed change| > max(replan_speed_kt, 2 sigma),
    or a maneuver was detected after the drop was planned,
and at most every replan_min_interval_s (a newly detected maneuver may start one at once, once).

Decision (hysteresis 2): on the same, current estimate the cost (optimal_deployment's predicted
error with the risk, coverage and mirror terms) of keeping the drops is compared with that of a
fresh plan made around the kept drops only; the drops are replaced only when the fresh plan
lowers it by at least replan_min_improvement. Both are evaluated with the not-yet-laid points
observing from now, so the comparison is symmetric.

The plan itself (candidates, cost, drop times, flight order) is optimal_deployment's; the
layer's free time / position with only the kept drops comes from layer.ready_pose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aqua_drift import layer as layerlib
from aqua_drift.forward_deployment import YD_TO_M, covered_count
from aqua_drift.models import (
    DropTask,
    ForwardDeploymentConfig,
    LayerState,
    PlanBasis,
    Position,
    ReplanFeed,
    TrackEstimate,
)
from aqua_drift.optimal_deployment import (
    FINE_S,
    FT_TO_M,
    LayerAvailability,
    SensorModel,
    hypotheses,
    plan_optimal_deployment,
)
from aqua_drift.physics import local_offset_m

KNOT_TO_MPS = 0.5144444444444445
FLYING_MODES = ("TRANSIT", "HOLD")


@dataclass
class ReplanDecision:
    replaces: list[int]  # open drop tasks to replace
    positions: list[Position]  # the new drops
    planned_ticks: list[int] | None
    bases: list[PlanBasis]
    revision: int
    reason: str


def flying_task_id(state: LayerState | None) -> int | None:
    """The drop the layer is flying to or holding at (None: orbiting)."""
    if state is None or state.mode not in FLYING_MODES:
        return None
    return state.task_id


def due_tick(task: DropTask, tick: int) -> float | None:
    """When the drop is expected: its planned time, or later when the layer gets there later."""
    eta = None if task.eta_s is None or not math.isfinite(task.eta_s) else tick + task.eta_s
    if task.planned_tick is None:
        return eta
    return task.planned_tick if eta is None else max(task.planned_tick, eta)


def is_revisable(task: DropTask, tick: int, config: ForwardDeploymentConfig, flying: int | None,
                 auto_approval: bool = True) -> bool:
    """May a fresh plan replace this open drop? (see the module docstring: what is kept)
    With manual approval only proposed drops: an approved one stays as the operator approved it
    (its replacement would wait for approval again)."""
    if task.status not in ("PROPOSED", "APPROVED") or task.source != "forward" or task.basis is None:
        return False
    if task.status == "APPROVED" and not auto_approval:
        return False
    if task.task_id == flying or task.revision >= config.replan_max_revisions:
        return False
    due = due_tick(task, tick)
    return due is not None and due - tick > config.replan_freeze_s


def basis_for(estimate: TrackEstimate, position: Position) -> PlanBasis:
    """The estimate a drop at position is planned on (relative offset at the estimate's tick)."""
    east, north, _ = local_offset_m(estimate.current_position, position)
    return PlanBasis(tick=estimate.tick, east_m=east, north_m=north, hdg_deg=estimate.hdg_deg or 0.0,
                     speed_kt=estimate.through_water_speed_kt or 0.0)


def _angle_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def estimate_change(estimate: TrackEstimate, drops: list[DropTask],
                    config: ForwardDeploymentConfig, r_max_m: float) -> str | None:
    """Why the estimate no longer matches what these drops were planned on (None: it does)."""
    u = estimate.uncertainty
    shift_limit = max(config.replan_shift_sigma * u.horizontal_major_yd * YD_TO_M,
                      config.replan_shift_fraction * r_max_m)
    heading_limit = max(config.replan_heading_deg, 2.0 * u.hdg_sigma_deg)
    speed_limit = max(config.replan_speed_kt, 2.0 * u.through_water_speed_sigma_kt)
    maneuver = (estimate.metadata or {}).get("maneuver_detected_tick")
    hdg, speed = estimate.hdg_deg or 0.0, estimate.through_water_speed_kt or 0.0
    for task in drops:
        b = task.basis
        if maneuver is not None and maneuver > b.tick:
            return f"maneuver detected at tick {maneuver} after drop {task.task_id} was planned"
        if _angle_deg(hdg, b.hdg_deg) > heading_limit:
            return f"heading {b.hdg_deg:.0f} -> {hdg:.0f} deg since drop {task.task_id} was planned"
        if abs(speed - b.speed_kt) > speed_limit:
            return f"speed {b.speed_kt:.1f} -> {speed:.1f} kt since drop {task.task_id} was planned"
        dt = estimate.tick - b.tick
        v = b.speed_kt * KNOT_TO_MPS
        h = math.radians(b.hdg_deg)
        expected = (b.east_m - v * math.sin(h) * dt, b.north_m - v * math.cos(h) * dt)
        east, north, _ = local_offset_m(estimate.current_position, task.position)
        shift = math.hypot(east - expected[0], north - expected[1])
        if shift > shift_limit:
            return (f"estimate moved {shift / YD_TO_M:.0f} YD (> {shift_limit / YD_TO_M:.0f}) "
                    f"since drop {task.task_id} was planned")
    return None


def layer_availability(replan: ReplanFeed, kept: list[DropTask], tick: int) -> LayerAvailability:
    """When / where the layer is free for new drops with only the kept drops in its queue."""
    queue = sorted(kept, key=lambda t: t.flight_key())
    ready, where, heading = layerlib.ready_pose(replan.layer_state, replan.layer, queue, tick, replan.datum)
    state = replan.layer_state
    return LayerAvailability(
        ready_s=float(max(ready - tick, 0)), position=where, speed_kt=replan.layer.speed_kt,
        max_bank_deg=replan.layer.max_bank_deg, heading_deg=heading,
        now_position=state.position if state is not None else None,
        now_heading_deg=state.heading_deg if state is not None else None,
        queue=[(t.position, None if t.planned_tick is None else float(t.planned_tick - tick)) for t in queue],
    )


def plan_cost_yd(estimate: TrackEstimate, observers: list[Position], pending: list[Position],
                 config: ForwardDeploymentConfig, max_slant_range_yd: float, sensor: SensorModel,
                 max_depth_ft: float) -> float | None:
    """optimal_deployment's cost (predicted error, YD) of a field without adding to it."""
    probe = config.model_copy(update={"max_per_drop": 1, "trigger_gain": 1.0})
    _, _, report = plan_optimal_deployment(
        estimate, observers, pending, probe, max_slant_range_yd, 1,
        sensor.source_frequency_hz, sensor.sound_speed_mps, sensor.frequency_sigma_hz,
        sensor=sensor, max_depth_ft=max_depth_ft,
    )
    return None if report is None else report.cost_before_yd


def plan_replacement(
    tick: int,
    estimate: TrackEstimate | None,
    observers: list[Position],
    other_pending: list[Position],
    replan: ReplanFeed,
    config: ForwardDeploymentConfig,
    max_slant_range_yd: float,
    free_slots: int,
    sensor: SensorModel,
    max_depth_ft: float = 1500.0,
) -> tuple[ReplanDecision | None, str]:
    """Replace the revisable open drops with a fresh plan? (decision or None, reason)

    other_pending: observers on their way into the water that are not open drops (queued
    placements, observers in free fall); free_slots: as for the forward planner (the open drops
    hold slots; the replaced ones give theirs back)."""
    if not (config.enabled and config.replan_enabled and config.strategy == "optimal"):
        return None, "replanning disabled"
    flying = flying_task_id(replan.layer_state)
    open_tasks = [t for t in replan.tasks if t.status in ("PROPOSED", "APPROVED")]
    auto = replan.layer.approval == "auto"
    revisable = [t for t in open_tasks if is_revisable(t, tick, config, flying, auto)]
    if not revisable:
        return None, "no revisable drop"
    if estimate is None or estimate.current_position is None or estimate.uncertainty is None:
        return None, "no estimate"
    if not estimate.observability_status.startswith("TRACKING") or estimate.hdg_deg is None:
        return None, f"estimate not usable ({estimate.observability_status})"
    r_max = max_slant_range_yd * YD_TO_M
    if estimate.uncertainty.horizontal_major_yd * YD_TO_M > config.optimal_max_sigma_fraction * r_max:
        return None, "estimate not converged enough to replan"
    if (estimate.through_water_speed_kt or 0.0) < config.min_speed_kt:
        return None, "target (nearly) stationary in the water"
    why = estimate_change(estimate, revisable, config, r_max)
    if why is None:
        return None, "open drops still match the estimate"
    maneuver = (estimate.metadata or {}).get("maneuver_detected_tick")
    last = replan.last_replan_tick
    new_maneuver = maneuver is not None and (last is None or maneuver > last)
    if last is not None and tick - last < config.replan_min_interval_s and not new_maneuver:
        return None, f"{why}; replanned {tick - last} s ago (< {config.replan_min_interval_s} s)"

    ids = {t.task_id for t in revisable}
    kept = [t for t in open_tasks if t.task_id not in ids]
    kept_pending = [*other_pending, *(t.position for t in kept)]
    keep_cost = plan_cost_yd(estimate, observers, [*kept_pending, *(t.position for t in revisable)],
                             config, max_slant_range_yd, sensor, max_depth_ft)
    if keep_cost is None:
        return None, f"{why}; the open drops cannot be evaluated"
    covered = covered_count(estimate, [*observers, *kept_pending], config, r_max)
    positions, plan_reason, report = plan_optimal_deployment(
        estimate, observers, kept_pending, config, max_slant_range_yd, free_slots + len(revisable),
        sensor.source_frequency_hz, sensor.sound_speed_mps, sensor.frequency_sigma_hz,
        coverage_short=covered < config.min_coverage, layer=layer_availability(replan, kept, tick),
        sensor=sensor, max_depth_ft=max_depth_ft,
    )
    if not positions:
        return None, f"{why}; kept the open drops (a fresh plan adds none: {plan_reason})"
    new_cost = plan_cost_yd(estimate, observers, [*kept_pending, *positions],
                            config, max_slant_range_yd, sensor, max_depth_ft)
    if new_cost is None or new_cost > (1.0 - config.replan_min_improvement) * keep_cost:
        return None, (f"{why}; kept the open drops (fresh plan {new_cost or float('nan'):.0f} YD vs "
                      f"{keep_cost:.0f} YD: < {config.replan_min_improvement * 100:.0f} % better)")
    replaced = sorted(ids)
    reason = (f"replan: {why}; drops {replaced} replaced, predicted error {keep_cost:.0f} -> "
              f"{new_cost:.0f} YD; {plan_reason}")
    planned = [tick + round(t) for t in report.drop_times_s] if report else None
    return ReplanDecision(
        replaces=replaced, positions=positions, planned_ticks=planned,
        bases=[basis_for(estimate, p) for p in positions],
        revision=1 + max(t.revision for t in revisable), reason=reason,
    ), reason


def detection_weight(estimate: TrackEstimate, point: Position, drop_s: float, config: ForwardDeploymentConfig,
                     max_slant_range_yd: float, max_depth_ft: float = 1500.0) -> float:
    """Weight of the motion hypotheses (optimal_deployment.hypotheses: the estimated track +-1 sigma
    and the maneuvers) in which the target comes within detection range of an observer laid at
    point drop_s from now, within the planning horizon. The range is R_max plus 2 sigma of the
    horizontal position uncertainty, so a drop counts as useless only when it clearly is."""
    origin = estimate.current_position
    u = estimate.uncertainty
    depth = (estimate.depth_ft if estimate.depth_ft is not None else origin.depth_ft) * FT_TO_M
    speed = (estimate.through_water_speed_kt or 0.0) * KNOT_TO_MPS
    vz = (estimate.vertical_rate_fps or 0.0) * FT_TO_M
    hdg_sigma = math.radians(min(max(u.hdg_sigma_deg, 3.0), 30.0))
    horizon = float(config.horizon_s)
    hyps = hypotheses(np.array([0.0, 0.0, depth]), math.radians(estimate.hdg_deg or 0.0), speed, vz, hdg_sigma,
                      config, horizon, max_depth_ft * FT_TO_M)
    east, north, _ = local_offset_m(origin, point)
    at = np.array([east, north, point.depth_ft * FT_TO_M])
    reach = max_slant_range_yd * YD_TO_M + 2.0 * u.horizontal_major_yd * YD_TO_M
    weight = 0.0
    for h in hyps:
        after = np.arange(len(h.pos)) * FINE_S >= drop_s
        if after.any() and np.min(np.linalg.norm(h.pos[after] - at, axis=1)) <= reach:
            weight += h.weight
    return weight / sum(h.weight for h in hyps)


def cancel_suggestions(tick: int, estimate: TrackEstimate | None, replan: ReplanFeed,
                       config: ForwardDeploymentConfig, max_slant_range_yd: float,
                       max_depth_ft: float = 1500.0) -> dict[int, str | None]:
    """Which approved drops the replanner keeps (flown to, due soon, revised to the limit, operator
    drops) should be proposed to the operator for cancellation, because the target will no longer
    come within detection range of them (task id -> reason, None: no longer proposed). Replaceable
    drops are not listed: the replanner replaces them itself. Nothing is cancelled here: the layer
    keeps flying a proposed drop until the operator cancels it. Hysteresis: proposed below
    cancel_suggest_weight, withdrawn only above 2.5 x it."""
    if not (config.enabled and config.replan_enabled) or estimate is None or estimate.uncertainty is None \
            or estimate.current_position is None or estimate.hdg_deg is None \
            or not estimate.observability_status.startswith("TRACKING"):
        return {}
    r_max = max_slant_range_yd * YD_TO_M
    if estimate.uncertainty.horizontal_major_yd * YD_TO_M > config.optimal_max_sigma_fraction * r_max:
        return {}  # not converged: no reason to doubt the drops yet
    flying = flying_task_id(replan.layer_state)
    auto = replan.layer.approval == "auto"
    out: dict[int, str | None] = {}
    for task in replan.tasks:
        if task.status != "APPROVED" or is_revisable(task, tick, config, flying, auto):
            continue
        due = due_tick(task, tick)
        drop_s = max((due if due is not None else tick) - estimate.tick, 0.0)
        weight = detection_weight(estimate, task.position, drop_s, config, max_slant_range_yd, max_depth_ft)
        limit = config.cancel_suggest_weight * (2.5 if task.cancel_suggestion else 1.0)
        if weight < limit:
            # shown to the operator as it is
            out[task.task_id] = (f"探知に寄与しない見込み（投入後に目標が探知距離に入る運動仮説は "
                                 f"{100 * weight:.0f} % のみ）")
        elif task.cancel_suggestion:
            out[task.task_id] = None
    return out
