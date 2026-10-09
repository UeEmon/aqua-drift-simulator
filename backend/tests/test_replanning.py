import math

import pytest

from aqua_drift.deployment import _offset
from aqua_drift.models import (
    DeploymentRequest,
    DropTask,
    EstimateMode,
    ForwardDeploymentConfig,
    LayerConfig,
    LayerState,
    Position,
    PresenceRegion,
    ReplanFeed,
    ScenarioConfig,
    TrackEstimate,
    Uncertainty,
)
from aqua_drift.optimal_deployment import SensorModel
from aqua_drift.physics import local_offset_m
from aqua_drift.replanning import basis_for, estimate_change, is_revisable, plan_replacement
from aqua_drift.state import ReplanRejected, SimulationState

ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=400.0)
R_MAX_YD = 6000
TICK = 1000


def estimate(hdg: float = 90.0, stw: float = 8.0, at: Position = ORIGIN, tick: int = TICK) -> TrackEstimate:
    return TrackEstimate(
        mode=EstimateMode.ONLINE, tick=tick, observability_status="TRACKING", current_position=at,
        depth_ft=400.0, hdg_deg=hdg, through_water_speed_kt=stw,
        uncertainty=Uncertainty(
            horizontal_major_yd=50, horizontal_minor_yd=20, horizontal_major_axis_deg=0,
            depth_sigma_ft=100, ground_speed_sigma_kt=0.2, through_water_speed_sigma_kt=0.2,
            cog_sigma_deg=1, hdg_sigma_deg=1, bias_sigma_hz=0.01,
        ),
        presence_region=PresenceRegion(probability_pct=90),
    )


def offset(east_yd: float, north_yd: float, depth_ft: float = 200.0) -> Position:
    return _offset(ORIGIN, east_yd * 0.9144, north_yd * 0.9144, depth_ft)


def drop(task_id: int, position: Position, planned_in_s: float, basis_estimate: TrackEstimate,
         source: str = "forward", revision: int = 0) -> DropTask:
    return DropTask(task_id=task_id, created_tick=basis_estimate.tick, source=source, position=position,
                    status="APPROVED", planned_tick=TICK + round(planned_in_s), sequence=task_id,
                    basis=basis_for(basis_estimate, position), revision=revision)


def flying_to(task_id: int) -> LayerState:
    return LayerState(tick=TICK, position=offset(-2000, 0, 0.0), heading_deg=90.0, speed_kt=200.0,
                      mode="TRANSIT", task_id=task_id)


def feed(tasks: list[DropTask], layer_state: LayerState | None = None, last: int | None = None) -> ReplanFeed:
    return ReplanFeed(tasks=tasks, layer_state=layer_state, layer=LayerConfig(enabled=True), datum=ORIGIN,
                      last_replan_tick=last)


def replan(est, tasks, layer_state=None, last=None, config=None, observers=None):
    return plan_replacement(TICK, est, observers if observers is not None else BEHIND, [],
                            feed(tasks, layer_state, last), config or ForwardDeploymentConfig(),
                            R_MAX_YD, 99, SensorModel())


BEHIND = [offset(-3000, 2000), offset(-3000, -2000)]
EAST = estimate(hdg=90.0)  # what the open drops were planned on


def east_plan() -> list[DropTask]:
    """Drops ahead of a target heading east (planned at TICK on EAST)."""
    return [drop(1, offset(4000, 1500), 300, EAST), drop(2, offset(9000, -1500), 900, EAST),
            drop(3, offset(14000, 1500), 1500, EAST)]


def test_kept_drops_flown_to_due_soon_revised_or_manual() -> None:
    config = ForwardDeploymentConfig()
    tasks = east_plan()
    assert [is_revisable(t, TICK, config, None) for t in tasks] == [True, True, True]
    assert not is_revisable(tasks[1], TICK, config, flying=2)  # the layer is flying to it
    assert not is_revisable(tasks[0], TICK + 100, config, None)  # due in 200 s < replan_freeze_s
    assert not is_revisable(drop(4, offset(9000, 0), 900, EAST, revision=2), TICK, config, None)
    assert not is_revisable(drop(5, offset(9000, 0), 900, EAST, source="manual"), TICK, config, None)
    # manual approval: what the operator approved stays; a proposal may still be replaced
    assert not is_revisable(tasks[1], TICK, config, None, auto_approval=False)
    proposed = tasks[1].model_copy(update={"status": "PROPOSED"})
    assert is_revisable(proposed, TICK, config, None, auto_approval=False)
    late = tasks[2].model_copy(update={"eta_s": 100.0, "planned_tick": None})
    assert not is_revisable(late, TICK, config, None)  # as soon as possible and nearly there


def test_trigger_only_when_the_estimate_moved_away_from_the_plan() -> None:
    config = ForwardDeploymentConfig()
    r_max = R_MAX_YD * 0.9144
    tasks = east_plan()
    # 300 s later the target has advanced on its planned track: no change
    advanced = _offset(ORIGIN, 8 * 0.5144444 * 300, 0.0, 400.0)
    assert estimate_change(estimate(at=advanced, tick=TICK + 300), tasks, config, r_max) is None
    # small noise (well below 2 sigma and 0.15 R_max) does not trigger either
    noisy = _offset(ORIGIN, 8 * 0.5144444 * 300 + 60, 40.0, 400.0)
    assert estimate_change(estimate(hdg=95.0, at=noisy, tick=TICK + 300), tasks, config, r_max) is None
    # a large position shift, a course change, a speed change or a detected maneuver do
    shifted = _offset(ORIGIN, 8 * 0.5144444 * 300, 1500.0, 400.0)
    assert "moved" in estimate_change(estimate(at=shifted, tick=TICK + 300), tasks, config, r_max)
    assert "heading" in estimate_change(estimate(hdg=150.0, tick=TICK), tasks, config, r_max)
    assert "speed" in estimate_change(estimate(stw=4.0, tick=TICK), tasks, config, r_max)
    turned = estimate(tick=TICK)
    turned.metadata["maneuver_detected_tick"] = TICK - 10
    assert estimate_change(turned, tasks, config, r_max) is None  # before the plan: already in it
    turned.metadata["maneuver_detected_tick"] = TICK + 5
    assert "maneuver" in estimate_change(turned, tasks, config, r_max)


def test_course_change_replaces_the_revisable_drops_but_keeps_the_one_flown_to() -> None:
    tasks = east_plan()
    north = estimate(hdg=0.0)  # the target turned north: the drops to the east are useless
    decision, reason = replan(north, tasks, layer_state=flying_to(1))
    assert decision is not None, reason
    assert decision.replaces == [2, 3]  # drop 1 is being flown to
    assert decision.revision == 1
    assert len(decision.bases) == len(decision.positions) >= 1
    assert all(b.hdg_deg == 0.0 and b.tick == TICK for b in decision.bases)
    for position in decision.positions:
        _, north_m, _ = local_offset_m(ORIGIN, position)
        assert north_m > 0  # ahead on the new heading
    assert decision.planned_ticks and all(t >= TICK for t in decision.planned_ticks)
    assert "replaced" in reason and "predicted error" in reason


def test_no_replan_while_the_plan_still_matches_or_within_the_interval() -> None:
    tasks = east_plan()
    decision, reason = replan(EAST, tasks)
    assert decision is None and reason == "open drops still match the estimate"
    north = estimate(hdg=0.0)
    decision, reason = replan(north, tasks, last=TICK - 60)
    assert decision is None and "replanned 60 s ago" in reason
    # a newly detected maneuver may replan at once, once
    north.metadata["maneuver_detected_tick"] = TICK - 10
    assert replan(north, tasks, last=TICK - 60)[0] is not None
    assert replan(north, tasks, last=TICK - 5)[0] is None  # that maneuver was handled at TICK - 5


def test_replace_only_when_clearly_better() -> None:
    tasks = east_plan()
    north = estimate(hdg=0.0)
    picky = ForwardDeploymentConfig(replan_min_improvement=1.0)  # nothing is good enough
    decision, reason = replan(north, tasks, config=picky)
    assert decision is None and "kept the open drops" in reason
    assert replan(north, tasks, config=ForwardDeploymentConfig(replan_enabled=False))[0] is None


def test_revision_limit_lets_every_drop_settle() -> None:
    """Even an estimate that keeps swinging cannot keep a drop from being laid: after
    replan_max_revisions replacements the drops are kept."""
    config = ForwardDeploymentConfig(replan_max_revisions=2, replan_min_interval_s=0)
    tasks = east_plan()
    revisions = []
    for k, hdg in enumerate([0.0, 90.0, 0.0, 90.0]):
        est = estimate(hdg=hdg)
        decision, _ = replan(est, tasks, config=config)
        if decision is None:
            break
        revisions.append(decision.revision)
        kept = [t for t in tasks if t.task_id not in decision.replaces]
        tasks = kept + [drop(100 * (k + 1) + i, p, t - TICK, est, revision=decision.revision)
                        for i, (p, t) in enumerate(zip(decision.positions, decision.planned_ticks, strict=True))]
    assert revisions and max(revisions) <= config.replan_max_revisions
    assert all(not is_revisable(t, TICK, config, None) for t in tasks if t.revision >= 2)


async def test_state_replaces_atomically_and_refuses_drops_no_longer_revisable() -> None:
    config = ScenarioConfig()
    config.layer.enabled = True
    config.layer.approval = "auto"
    state = SimulationState(config)
    state.tick = TICK
    bases = [basis_for(EAST, p) for p in (offset(4000, 1500), offset(9000, -1500))]
    record = await state.queue_deployment(DeploymentRequest(
        tick=TICK, positions=[offset(4000, 1500), offset(9000, -1500)], reason="plan",
        planned_ticks=[TICK + 600, TICK + 900], bases=bases))
    assert record.task_ids == [1, 2]  # the map shows a planned point only while its drop is open
    first = state.tasks[0]
    assert first.basis is not None and first.status == "APPROVED"
    feed_ = await state.deployment_feed()
    assert feed_.replan is not None and [t.task_id for t in feed_.replan.tasks] == [1, 2]

    # the layer now flies to drop 1: replacing 1 and 2 is refused as a whole
    state.layer_state = flying_to(first.task_id)
    with pytest.raises(ReplanRejected):
        await state.queue_deployment(DeploymentRequest(
            tick=TICK, positions=[offset(0, 6000)], reason="replan", replaces=[1, 2], revision=1))
    assert [t.status for t in state.tasks] == ["APPROVED", "APPROVED"]

    await state.queue_deployment(DeploymentRequest(
        tick=TICK, positions=[offset(0, 6000)], reason="replan", planned_ticks=[TICK + 800],
        bases=[basis_for(estimate(hdg=0.0), offset(0, 6000))], replaces=[2], revision=1))
    assert [t.status for t in state.tasks] == ["APPROVED", "REPLACED", "APPROVED"]
    assert state.tasks[2].revision == 1 and state.last_replan_tick == TICK
    assert math.isclose(state.tasks[2].basis.hdg_deg, 0.0)
