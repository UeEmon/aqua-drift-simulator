import math

from aqua_drift.forward_deployment import plan_forward_deployment
from aqua_drift.models import (
    EstimateMode,
    ForwardDeploymentConfig,
    Position,
    PresenceRegion,
    TrackEstimate,
    Uncertainty,
)
from aqua_drift.physics import local_offset_m

ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=400.0)


def estimate(status: str = "TRACKING", hdg: float = 90.0, stw: float = 8.0) -> TrackEstimate:
    return TrackEstimate(
        mode=EstimateMode.ONLINE,
        tick=100,
        observability_status=status,
        current_position=ORIGIN,
        depth_ft=400.0,
        hdg_deg=hdg,
        through_water_speed_kt=stw,
        uncertainty=Uncertainty(
            horizontal_major_yd=50, horizontal_minor_yd=20, horizontal_major_axis_deg=0,
            depth_sigma_ft=100, ground_speed_sigma_kt=0.2, through_water_speed_sigma_kt=0.2,
            cog_sigma_deg=1, hdg_sigma_deg=1, bias_sigma_hz=0.01,
        ),
        presence_region=PresenceRegion(probability_pct=90),
    )


def offset(east_yd: float, north_yd: float) -> Position:
    from aqua_drift.deployment import _offset

    return _offset(ORIGIN, east_yd * 0.9144, north_yd * 0.9144, 200.0)


def test_deploys_ahead_on_both_sides_when_predicted_position_uncovered() -> None:
    config = ForwardDeploymentConfig(strategy="fixed")
    behind = [offset(-3000, 2000), offset(-3000, -2000)]  # target heading east, observers behind
    positions, reason = plan_forward_deployment(100, estimate(), behind, [], config, 6000, None)
    assert len(positions) == config.observers_per_drop
    sides = []
    for position in positions:
        east, north, _ = local_offset_m(ORIGIN, position)
        assert east / 0.9144 > 3000  # ahead on the estimated heading (east)
        sides.append(math.copysign(1, north))
        assert 50 <= position.depth_ft <= 1500
    assert sorted(sides) == [-1, 1]  # one each side of the track
    assert "covered by" in reason


def test_no_deployment_when_covered_or_pending() -> None:
    config = ForwardDeploymentConfig(strategy="fixed")
    ahead = [offset(3000, 1500), offset(3000, -1500)]
    assert plan_forward_deployment(100, estimate(), ahead, [], config, 6000, None)[0] == []
    # already-pending placements count as coverage (no duplicate drops)
    assert plan_forward_deployment(100, estimate(), [], ahead, config, 6000, None)[0] == []


def test_guards_status_cooldown_speed_and_force() -> None:
    config = ForwardDeploymentConfig(strategy="fixed")
    assert plan_forward_deployment(100, estimate("AMBIGUOUS"), [], [], config, 6000, None)[0] == []
    assert plan_forward_deployment(100, estimate(), [], [], config, 6000, 50)[0] == []  # cooldown
    assert plan_forward_deployment(100, estimate(stw=0.1), [], [], config, 6000, None)[0] == []
    ahead = [offset(3000, 1500), offset(3000, -1500)]
    forced, reason = plan_forward_deployment(100, estimate(), ahead, [], config, 6000, 99, force=True)
    assert len(forced) == 2 and reason == "operator request"
    disabled = ForwardDeploymentConfig(enabled=False)
    assert plan_forward_deployment(100, estimate(), [], [], disabled, 6000, None)[0] == []


# ------------------------------------------------------------------ optimal strategy
def _local(position: Position) -> tuple[float, float]:
    east, north, _ = local_offset_m(ORIGIN, position)
    return east / 0.9144, north / 0.9144


def test_optimal_places_ahead_with_count_and_depths_from_the_information() -> None:
    config = ForwardDeploymentConfig()  # strategy "optimal" is the default
    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    positions, reason = plan_forward_deployment(100, estimate(), behind, [], config, 6000, None)
    assert 1 <= len(positions) <= config.max_per_drop
    assert reason.startswith("optimal (coverage)")
    lanes = set()
    for position in positions:
        east, north = _local(position)
        assert east > 0  # ahead on the estimated heading (east)
        assert abs(north) <= 6000  # within detection range of the (possibly turning) track
        assert position.depth_ft in config.depth_options_ft
        lanes.add(round(north / 1500))
    assert len(lanes) >= 2  # never one line of new observers (mirror ambiguity about it)
    # depth: at least one observer far from the estimated target depth (400 ft)
    assert max(abs(p.depth_ft - 400.0) for p in positions) >= 500


def test_optimal_count_follows_the_target_error_and_free_slots() -> None:
    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    loose = ForwardDeploymentConfig(target_error_yd=200.0)
    tight = ForwardDeploymentConfig(target_error_yd=5.0, min_relative_gain=0.0)
    few = plan_forward_deployment(100, estimate(), behind, [], loose, 6000, None)[0]
    many = plan_forward_deployment(100, estimate(), behind, [], tight, 6000, None)[0]
    assert 1 <= len(few) < len(many) <= tight.max_per_drop
    capped = plan_forward_deployment(100, estimate(), behind, [], tight, 6000, None, free_slots=1)[0]
    assert len(capped) == 1
    assert plan_forward_deployment(100, estimate(), behind, [], tight, 6000, None, free_slots=0)[0] == []


def test_optimal_does_not_deploy_when_the_field_already_tracks_well() -> None:
    config = ForwardDeploymentConfig(depth_weight=0.0, maneuver_weight=0.0)
    field = [offset(3000, 1500), offset(3000, -1500), offset(9000, 1500), offset(9000, -1500), offset(15000, 0)]
    positions, reason = plan_forward_deployment(100, estimate(), field, [], config, 6000, None)
    assert positions == []
    assert "already within" in reason
    forced, reason = plan_forward_deployment(100, estimate(), field, [], config, 6000, None, force=True)
    assert len(forced) >= 1 and "operator request" in reason


# ------------------------------------------------------------- maneuvers, bearings, gate
FIELD = [offset(3000, 1500), offset(3000, -1500), offset(9000, 1500), offset(9000, -1500), offset(15000, 0)]


def _plan(observers, config, sensor=None, **kwargs):
    from aqua_drift.optimal_deployment import plan_optimal_deployment

    return plan_optimal_deployment(estimate(), observers, [], config, 6000, 99, sensor=sensor, **kwargs)


def test_maneuver_hypotheses_cover_course_changes() -> None:
    """A field laid only along the current track is enough if the target holds its course,
    but not for the course-change hypotheses: the robust plan adds observers and closes the
    coverage gaps of the turning tracks."""
    straight = ForwardDeploymentConfig(depth_weight=0.0, maneuver_weight=0.0)
    robust = ForwardDeploymentConfig(depth_weight=0.0)
    assert _plan(FIELD, straight)[0] == []
    positions, reason, report = _plan(FIELD, robust)
    assert positions, reason
    assert report.hypotheses == 11
    assert report.coverage_loss_after < report.coverage_loss_before
    assert report.cost_after_yd < report.cost_before_yd


def test_turning_hypotheses_widen_the_candidates() -> None:
    import numpy as np

    from aqua_drift.optimal_deployment import hypotheses

    config = ForwardDeploymentConfig()
    hyps = hypotheses(np.array([0.0, 0.0, 150.0]), math.radians(90), 4.0, 0.0, math.radians(3), config, 1800, 457)
    assert abs(sum(h.weight for h in hyps) - 1.0) < 1e-9
    ends = {h.name: h.pos[-1] for h in hyps}
    assert ends["turn+90"][1] < -3000 and ends["turn-90"][1] > 3000  # south / north after the turn
    assert ends["slow"][0] < ends["track"][0] < ends["fast"][0]
    assert ends["down"][2] > 150 + 100 and ends["up"][2] < 150 - 80
    assert all(0 <= h.pos[:, 2].min() and h.pos[:, 2].max() <= 457 + 1e-6 for h in hyps)


def test_bearings_and_detection_gate_add_information() -> None:
    from aqua_drift.optimal_deployment import SensorModel

    config = ForwardDeploymentConfig(depth_weight=0.0, maneuver_weight=0.0)
    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    doppler = _plan(behind, config, SensorModel(use_bearing=False, use_gate=False), force=True)[2]
    bearing = _plan(behind, config, SensorModel(use_bearing=True, use_gate=False), force=True)[2]
    gate = _plan(behind, config, SensorModel(use_bearing=True, use_gate=True), force=True)[2]
    assert bearing.horizontal_before_yd < doppler.horizontal_before_yd
    assert gate.horizontal_before_yd <= bearing.horizontal_before_yd


def test_mirror_term_breaks_a_line_of_observers_without_bearings() -> None:
    """Observers in one line leave the mirror image of the track about it; without bearings
    that ambiguity costs, with bearings it is resolved."""
    import numpy as np

    from aqua_drift.optimal_deployment import SensorModel, _mirror_cost

    line = np.array([[x, 1500.0, 60.0] for x in (0.0, 2000.0, 4000.0, 6000.0)])
    off = np.vstack([line, [[3000.0, -2500.0, 60.0]]])
    sets = np.array([np.vstack([line, [[8000.0, 1500.0, 60.0]]]), off])
    start = np.zeros(sets.shape[0:2])
    ticks = np.arange(0.0, 1801.0, 60.0)
    pos = np.stack([ticks * 4.0, np.zeros_like(ticks), np.full_like(ticks, 150.0)], axis=1)
    vel = np.tile([4.0, 0.0, 0.0], (len(ticks), 1))
    evaluate = ticks >= 600
    no_bearing = _mirror_cost(sets, start, pos, vel, ticks, evaluate, SensorModel(use_bearing=False), 5486.0)
    with_bearing = _mirror_cost(sets, start, pos, vel, ticks, evaluate, SensorModel(use_bearing=True), 5486.0)
    assert no_bearing[0] > 1e5  # collinear: ~50 % chance of the 3000 m mirror
    assert no_bearing[1] < 1e-3 * no_bearing[0]  # one observer off the line resolves it
    assert with_bearing[0] < 1e-3 * no_bearing[0]  # bearings resolve it too


def test_detected_maneuver_replans_within_the_cooldown() -> None:
    config = ForwardDeploymentConfig()
    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    assert plan_forward_deployment(100, estimate(), behind, [], config, 6000, 50)[1] == "cooldown"
    turned = estimate()
    turned.metadata["maneuver_detected_tick"] = 90
    positions, reason = plan_forward_deployment(100, turned, behind, [], config, 6000, 50)
    assert positions and "replanned after a maneuver" in reason
    turned.metadata["maneuver_detected_tick"] = 40  # already handled by the deployment at 50
    assert plan_forward_deployment(100, turned, behind, [], config, 6000, 50)[1] == "cooldown"


# ------------------------------------------------------------- short detection range
def _layer_at(east_m: float, north_m: float, heading_deg: float):
    from aqua_drift.deployment import _offset
    from aqua_drift.optimal_deployment import LayerAvailability

    return LayerAvailability(ready_s=0.0, position=_offset(ORIGIN, east_m, north_m, 0.0), speed_kt=200.0,
                             max_bank_deg=15.0, heading_deg=heading_deg)


def test_short_range_plan_is_one_line_along_the_track_laid_in_one_pass() -> None:
    """Detection range (500 YD) far shorter than the layer's turn radius (~4 km): the observers of
    one plan lie on one line along the track, close ahead, and are due together, so the layer lays
    them in one straight pass instead of a loop per observer. The plan stays cheap (candidates only
    on a few lines along the track)."""
    from aqua_drift.optimal_deployment import plan_optimal_deployment

    behind = [offset(-1500, 300), offset(-1500, -300)]
    positions, reason, report = plan_optimal_deployment(
        estimate(), behind, [], ForwardDeploymentConfig(), 500, 99, coverage_short=True,
        layer=_layer_at(-8000.0, -3000.0, 60.0))
    assert len(positions) >= 2, reason
    assert report.candidates < 2000
    north = [local_offset_m(ORIGIN, p)[1] for p in positions]
    assert max(north) - min(north) < 1.0  # one line, parallel to the track (east)
    east = sorted(local_offset_m(ORIGIN, p)[0] for p in positions)
    assert east[-1] < 8.0 * 500 * 0.9144  # close ahead, not spread over the horizon
    span = max(report.drop_times_s) - min(report.drop_times_s)
    assert span <= (east[-1] - east[0]) / (200.0 * 0.5144444444444445) + 5.0  # straight along it


def test_long_range_plan_keeps_the_free_placement() -> None:
    from aqua_drift.optimal_deployment import plan_optimal_deployment

    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    config = ForwardDeploymentConfig(target_error_yd=5.0, min_relative_gain=0.0)
    lined = plan_optimal_deployment(estimate(), behind, [], config, 6000, 99, layer=_layer_at(0.0, -5000.0, 0.0))
    off = plan_optimal_deployment(estimate(), behind, [], config.model_copy(update={"line_laying": False}), 6000, 99,
                                  layer=_layer_at(0.0, -5000.0, 0.0))
    assert [(p.latitude, p.longitude) for p in lined[0]] == [(p.latitude, p.longitude) for p in off[0]]


def test_cost_only_is_the_cost_before_of_a_full_plan() -> None:
    from aqua_drift.optimal_deployment import plan_optimal_deployment

    config = ForwardDeploymentConfig()
    full = plan_optimal_deployment(estimate(), FIELD, [], config, 6000, 99, force=True)[2]
    only = plan_optimal_deployment(estimate(), FIELD, [], config, 6000, 99, cost_only=True)[2]
    assert only.candidates == 0
    assert abs(only.cost_before_yd - full.cost_before_yd) < 1e-6


# ------------------------------------------------------------- several observers in one drop
def _points(positions: list[Position]) -> list[tuple[int, int]]:
    return [tuple(round(v) for v in local_offset_m(ORIGIN, p)[0:2]) for p in positions]


def test_observers_stack_at_one_drop_point_only_up_to_the_layers_release() -> None:
    """With the layer releasing up to max_per_release observers in one drop, the planner may give a
    drop point other depths when they help most there: the observers at one point are due together
    and routed as one drop (one pass). It is a maximum: a drop point holds at most that many, and
    with one observer per drop the observers keep apart as before."""
    from aqua_drift.deployment import _offset
    from aqua_drift.optimal_deployment import (
        MIN_SEPARATION,
        LayerAvailability,
        plan_optimal_deployment,
    )

    behind = [offset(-3000, 2000), offset(-3000, -2000)]
    config = ForwardDeploymentConfig(depth_weight=10.0, target_error_yd=1.0, min_relative_gain=0.0, max_per_drop=8)
    plans = {}
    for release in (1, 2, 4):
        layer = LayerAvailability(ready_s=0.0, position=_offset(ORIGIN, 0.0, -5000.0, 0.0), speed_kt=200.0,
                                  max_bank_deg=15.0, heading_deg=0.0, max_per_release=release)
        positions, reason, report = plan_optimal_deployment(estimate(), behind, [], config, 6000, 99,
                                                            coverage_short=True, layer=layer)
        assert positions, reason
        points = _points(positions)
        for point in set(points):
            at = [k for k, p in enumerate(points) if p == point]
            assert len(at) <= release
            assert at == list(range(at[0], at[0] + len(at)))  # one drop after the other: adjacent
            assert len({report.drop_times_s[k] for k in at}) == 1  # released together
            assert len({positions[k].depth_ft for k in at}) == len(at)  # each at its own depth
        plans[release] = (points, reason)
    points = plans[1][0]
    assert len(set(points)) == len(points)
    assert all(math.dist(a, b) >= MIN_SEPARATION * 6000 * 0.9144 - 1 for a in points for b in points if a != b)
    stacked, reason = plans[4]
    assert len(set(stacked)) < len(stacked), (stacked, reason)  # needed here: several in one drop
    assert "drop points" in reason
