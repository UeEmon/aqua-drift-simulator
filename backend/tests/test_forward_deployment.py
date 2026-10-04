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
    config = ForwardDeploymentConfig()
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
    config = ForwardDeploymentConfig()
    ahead = [offset(3000, 1500), offset(3000, -1500)]
    assert plan_forward_deployment(100, estimate(), ahead, [], config, 6000, None)[0] == []
    # already-pending placements count as coverage (no duplicate drops)
    assert plan_forward_deployment(100, estimate(), [], ahead, config, 6000, None)[0] == []


def test_guards_status_cooldown_speed_and_force() -> None:
    config = ForwardDeploymentConfig()
    assert plan_forward_deployment(100, estimate("AMBIGUOUS"), [], [], config, 6000, None)[0] == []
    assert plan_forward_deployment(100, estimate(), [], [], config, 6000, 50)[0] == []  # cooldown
    assert plan_forward_deployment(100, estimate(stw=0.1), [], [], config, 6000, None)[0] == []
    ahead = [offset(3000, 1500), offset(3000, -1500)]
    forced, reason = plan_forward_deployment(100, estimate(), ahead, [], config, 6000, 99, force=True)
    assert len(forced) == 2 and reason == "operator request"
    disabled = ForwardDeploymentConfig(enabled=False)
    assert plan_forward_deployment(100, estimate(), [], [], disabled, 6000, None)[0] == []
