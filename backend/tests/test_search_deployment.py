"""Re-detection (再探知) search drops after a lost contact."""
import math

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import (
    DropTask,
    EstimateMode,
    ForwardDeploymentConfig,
    LayerConfig,
    Position,
    PresenceRegion,
    ReplanFeed,
    ScenarioConfig,
    TrackEstimate,
)
from aqua_drift.physics import local_offset_m
from aqua_drift.replanning import basis_for, plan_search_replacement
from aqua_drift.scenario import ScenarioRun
from aqua_drift.search_deployment import (
    cloud_from_particles,
    plan_search_drops,
    predicted_positions,
    search_points,
)

ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=400.0)
R_MAX_YD = 1000
R_MAX_M = R_MAX_YD * 0.9144
TICK = 2000


def ring_cloud(radius_m: float = 900.0, count: int = 300) -> dict[str, list[float]]:
    """The coasted cloud after non-detection: a ring around the last detection, not moving."""
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    x = np.zeros((count, 7))
    x[:, 0], x[:, 1] = radius_m * np.sin(angle), radius_m * np.cos(angle)
    return cloud_from_particles(x, np.full(count, 1.0 / count), np.zeros(2), count, np.random.default_rng(0))


def lost(cloud: dict[str, list[float]] | None, status: str = "LOST") -> TrackEstimate:
    return TrackEstimate(
        mode=EstimateMode.ONLINE, tick=TICK, observability_status=status, current_position=ORIGIN,
        depth_ft=400.0, presence_region=PresenceRegion(probability_pct=90),
        metadata={} if cloud is None else {"search_cloud": cloud},
    )


def test_search_points_cover_the_cloud_with_few_discs() -> None:
    cloud = ring_cloud()
    points, covered = search_points(cloud, 0.0, 0.8 * R_MAX_M, 0.7, 4, np.random.default_rng(1))
    assert 1 < len(points) <= 4
    assert covered >= 0.5
    # every drop sits on the ring: that is where the cloud is
    assert np.all(np.abs(np.hypot(points[:, 0], points[:, 1]) - 900.0) < 5.0)


def test_search_stops_once_enough_of_the_cloud_is_covered() -> None:
    cloud = ring_cloud()
    few, covered = search_points(cloud, 0.0, 0.8 * R_MAX_M, 0.3, 4, np.random.default_rng(1))
    many, _ = search_points(cloud, 0.0, 0.8 * R_MAX_M, 1.0, 4, np.random.default_rng(1))
    assert len(few) < len(many)
    assert covered >= 0.3


def test_cloud_drifts_with_its_through_water_velocity() -> None:
    """The drop point is where the target will be at the drop, not where the cloud is now."""
    x = np.zeros((50, 7))
    x[:, 3] = 4.0  # 4 m/s east through the water
    cloud = cloud_from_particles(x, np.full(50, 0.02), np.zeros(2), 50, np.random.default_rng(0))
    points, covered = search_points(cloud, 300.0, 0.8 * R_MAX_M, 0.9, 4, np.random.default_rng(1))
    assert covered >= 0.9
    assert np.allclose(points[0], [1200.0, 0.0], atol=1.0)


def test_plan_needs_a_lost_estimate_and_a_cloud() -> None:
    config = ForwardDeploymentConfig()
    positions, why = plan_search_drops(lost(ring_cloud(), "COASTING_TRACKING"), config, R_MAX_YD, 4, 0.0)
    assert not positions and "not lost" in why
    positions, why = plan_search_drops(lost(None), config, R_MAX_YD, 4, 0.0)
    assert not positions and "no particle cloud" in why
    config.search_enabled = False
    assert not plan_search_drops(lost(ring_cloud()), config, R_MAX_YD, 4, 0.0)[0]


def test_plan_places_drops_around_the_estimate_within_the_free_slots() -> None:
    config = ForwardDeploymentConfig()
    positions, reason = plan_search_drops(lost(ring_cloud()), config, R_MAX_YD, 2, 0.0)
    assert len(positions) == 2 and "search" in reason
    for p in positions:
        east, north, _ = local_offset_m(ORIGIN, p)
        assert abs(math.hypot(east, north) - 900.0) < 10.0
    assert plan_search_drops(lost(ring_cloud()), config, R_MAX_YD, 0, 0.0)[0] == []


def forward_drop(task_id: int, in_s: int) -> DropTask:
    estimate = lost(None, "TRACKING")
    return DropTask(task_id=task_id, created_tick=TICK - 300, source="forward",
                    position=_offset(ORIGIN, 3000.0, 0.0, 200.0), status="APPROVED",
                    planned_tick=TICK + in_s, sequence=task_id, basis=basis_for(estimate, ORIGIN))


def replan_feed(*tasks: DropTask) -> ReplanFeed:
    return ReplanFeed(tasks=list(tasks), layer=LayerConfig(enabled=True), datum=ORIGIN)


def test_search_replaces_revisable_forward_drops_and_keeps_the_others() -> None:
    config = ForwardDeploymentConfig()
    far, soon = forward_drop(1, 900), forward_drop(2, config.replan_freeze_s - 10)
    decision, _ = plan_search_replacement(TICK, lost(ring_cloud()), replan_feed(far, soon), config,
                                          R_MAX_YD, 0, 0.0, None)
    assert decision is not None and decision.source == "search"
    assert decision.replaces == [1]  # the drop due soon stays: the head of the queue is laid
    assert decision.positions and decision.planned_ticks is None


def test_no_search_before_the_interval_or_while_search_drops_are_open() -> None:
    config = ForwardDeploymentConfig()
    cloud = ring_cloud()
    decision, why = plan_search_replacement(TICK, lost(cloud), replan_feed(), config, R_MAX_YD, 4, 0.0,
                                            TICK - config.search_interval_s + 1)
    assert decision is None and why == "search interval"
    open_search = forward_drop(3, 900)
    open_search.source = "search"
    decision, why = plan_search_replacement(TICK, lost(cloud), replan_feed(open_search), config, R_MAX_YD, 4,
                                            0.0, None)
    assert decision is None and why == "search drops still open"
    decision, _ = plan_search_replacement(TICK, lost(cloud), replan_feed(), config, R_MAX_YD, 4, 0.0,
                                          TICK - config.search_interval_s)
    assert decision is not None


def test_closed_loop_lays_search_drops_after_a_lost_contact() -> None:
    """500 YD detection range: the target leaves the initial square, the estimate goes LOST and
    the search drops are laid on the cloud (not on a track)."""
    config = ScenarioConfig()
    config.max_slant_range_yd = 500.0
    config.deployment.surround_radius_yd = 400.0
    config.estimator.particle_count = 800
    config.estimator.lost_timeout_s = 120
    config.forward.max_sigma_fraction = 1e-6  # the track planner stays out: only the search places drops
    config.forward.replan_enabled = False
    run = ScenarioRun(config, 4, forward=True)
    for _ in range(500):
        run.step()
        if any(t.source == "search" for t in run.tasks):
            break
    search = [t for t in run.tasks if t.source == "search"]
    assert search, "no search drop after the contact was lost"
    assert all(t.status in ("APPROVED", "DONE") and t.basis is None for t in search)
    assert "search after lost contact" in search[0].reason
    assert not any(t.source == "forward" for t in run.tasks)


def test_maneuver_hypotheses_spread_the_cloud_off_the_old_course() -> None:
    """A cloud that keeps its course east misses a target that turned while the contact was gone:
    with maneuver hypotheses the predicted positions leave the old course line and one disc covers
    less of the cloud."""
    x = np.zeros((300, 7))
    x[:, 3] = 4.0  # 4 m/s east through the water
    cloud = cloud_from_particles(x, np.full(300, 1 / 300), np.zeros(2), 300, np.random.default_rng(0))
    straight = predicted_positions(cloud, 300.0, 0.0, 400.0, np.random.default_rng(1))
    assert np.allclose(straight[:, 1], 0.0) and np.allclose(straight[:, 0], 1200.0)
    mixed = predicted_positions(cloud, 300.0, 0.5, 400.0, np.random.default_rng(1))
    off_course = np.abs(mixed[:, 1]) > 100.0
    assert 0.3 < off_course.mean() < 0.7  # about the share that are hypotheses
    _, covered_straight = search_points(cloud, 300.0, 0.8 * R_MAX_M, 1.0, 1, np.random.default_rng(1))
    _, covered_mixed = search_points(cloud, 300.0, 0.8 * R_MAX_M, 1.0, 1, np.random.default_rng(1),
                                     0.5, 400.0)
    assert covered_straight > 0.99 and covered_mixed < 0.7


def test_no_maneuver_hypotheses_when_disabled_or_when_the_cloud_is_still() -> None:
    cloud = ring_cloud()  # zero velocity: a hypothesis changes nothing
    a = predicted_positions(cloud, 100.0, 0.0, 300.0, np.random.default_rng(1))
    b = predicted_positions(cloud, 100.0, 0.8, 300.0, np.random.default_rng(1))
    assert np.allclose(a, b)
