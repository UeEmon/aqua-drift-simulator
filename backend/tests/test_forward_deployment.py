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
        assert abs(north) <= 0.8 * 6000  # within detection range of the predicted track
        assert position.depth_ft in config.depth_options_ft
        lanes.add(round(north / 1500))
    assert len(lanes) == len(positions)  # one per lane: never a collinear line of new observers
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
    config = ForwardDeploymentConfig(depth_weight=0.0)
    field = [offset(3000, 1500), offset(3000, -1500), offset(9000, 1500), offset(9000, -1500), offset(15000, 0)]
    positions, reason = plan_forward_deployment(100, estimate(), field, [], config, 6000, None)
    assert positions == []
    assert "already within" in reason
    forced, reason = plan_forward_deployment(100, estimate(), field, [], config, 6000, None, force=True)
    assert len(forced) >= 1 and "operator request" in reason
