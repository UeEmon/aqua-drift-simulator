"""Planned observers start observing (drop + fall + start-up) while the target is still ahead of them,
before its closest point of approach (CPA) on the estimated track."""
import math
import random

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import (
    EstimateMode,
    ForwardDeploymentConfig,
    Position,
    PresenceRegion,
    TrackEstimate,
    Uncertainty,
)
from aqua_drift.optimal_deployment import (
    FINE_S,
    LayerAvailability,
    cpa_s,
    drop_times,
    plan_optimal_deployment,
)
from aqua_drift.physics import local_offset_m

ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)
KNOT = 0.5144444444444445


def _estimate(hdg: float, speed_kt: float, sigma_yd: float = 50.0) -> TrackEstimate:
    return TrackEstimate(
        mode=EstimateMode.ONLINE, tick=100, observability_status="TRACKING", current_position=ORIGIN,
        depth_ft=400.0, hdg_deg=hdg, through_water_speed_kt=speed_kt,
        uncertainty=Uncertainty(
            horizontal_major_yd=sigma_yd, horizontal_minor_yd=20, horizontal_major_axis_deg=0, depth_sigma_ft=100,
            ground_speed_sigma_kt=0.2, through_water_speed_sigma_kt=0.2, cog_sigma_deg=1, hdg_sigma_deg=1,
            bias_sigma_hz=0.01,
        ),
        presence_region=PresenceRegion(probability_pct=90),
    )


def _layer(east: float, north: float, **kw) -> LayerAvailability:
    args = {"ready_s": 0.0, "position": _offset(ORIGIN, east, north, 0.0), "speed_kt": 200.0, "max_bank_deg": 15.0,
            "heading_deg": 0.0, "fall_terminal_mps": 100.0 * 0.3048, "straight_before_s": 2.0, "straight_after_s": 5.0}
    args.update(kw)
    return LayerAvailability(**args)


def test_cpa_time_is_the_along_track_distance_over_the_speed() -> None:
    points = np.array([[0.0, 1000.0, 0.0], [500.0, 1000.0, 0.0], [0.0, -200.0, 0.0]])
    assert np.allclose(cpa_s(points, 0.0, 10.0), [100.0, 100.0, -20.0])  # beside the track: the same CPA
    assert np.allclose(cpa_s(points, math.pi / 2, 10.0), [0.0, 50.0, 0.0])
    assert np.all(np.isinf(cpa_s(points, 0.0, 0.0)))


def test_drop_time_leaves_the_start_up_and_never_starts_after_the_latest_start() -> None:
    point = np.array([[0.0, 0.0, 100.0]])
    t = np.arange(0.0, 4000.0 + 1e-9, FINE_S)
    track = np.stack([np.zeros_like(t), (t - 2000.0) - 1000.0, np.full_like(t, 100.0)], axis=1)  # enters at 2000 s
    earliest = np.array([300.0])
    args = (point, earliest, [track], 1000.0, 60.0, 0.0, 1.0)
    plain = drop_times(*args)[0]
    assert abs(plain - (2000.0 - 60.0)) < 1e-6
    assert abs(drop_times(*args, startup_s=30.0)[0] - (plain - 30.0)) < 1e-6
    # the latest start (ahead of the CPA) bounds it, but never before the layer can be there
    assert abs(drop_times(*args, startup_s=30.0, latest_start=np.array([1000.0]))[0] - 970.0) < 1e-6
    assert drop_times(*args, startup_s=30.0, latest_start=np.array([100.0]))[0] == 300.0


def _check_ahead(positions, report, hdg: float, speed_kt: float, config: ForwardDeploymentConfig,
                 margin_s: float) -> None:
    h, v = math.radians(hdg), speed_kt * KNOT
    for p, t in zip(positions, report.drop_times_s, strict=True):
        east, north, _ = local_offset_m(ORIGIN, p)
        cpa = (east * math.sin(h) + north * math.cos(h)) / v
        start = t + config.observer_startup_s  # + the fall (> 0)
        assert start <= cpa - margin_s + 1.0, (start, cpa)


def test_planned_observers_start_observing_ahead_of_the_target() -> None:
    rng = random.Random(3)
    config = ForwardDeploymentConfig()
    for _ in range(12):
        hdg, speed = rng.uniform(0.0, 360.0), rng.choice([8.0, 15.0, 25.0])
        r_max = rng.choice([1500.0, 6000.0])
        bearing, distance = rng.uniform(0.0, 2 * math.pi), rng.uniform(2000.0, 25000.0)
        layer = _layer(distance * math.sin(bearing), distance * math.cos(bearing), ready_s=rng.uniform(0.0, 300.0),
                       heading_deg=rng.uniform(0.0, 360.0))
        estimate = _estimate(hdg, speed)
        positions, _, report = plan_optimal_deployment(
            estimate, [], [], config, r_max, 99, coverage_short=True, layer=layer)
        if not positions:
            continue
        sigma_m = 50.0 * 0.9144
        _check_ahead(positions, report, hdg, speed, config, config.cpa_ahead_s + 2.0 * sigma_m / (speed * KNOT))


def test_a_layer_that_cannot_get_ahead_lays_nothing() -> None:
    # a fast target heading away from a layer far behind it that is busy for a while: nothing it could
    # lay would start observing before the target has passed
    estimate = _estimate(0.0, 25.0)
    config = ForwardDeploymentConfig(horizon_s=600, lead_time_s=60)
    layer = _layer(0.0, -60000.0, ready_s=900.0)
    positions, reason, _ = plan_optimal_deployment(estimate, [], [], config, 1500.0, 99, coverage_short=True,
                                                   layer=layer)
    assert positions == []
    assert "CPA" in reason or "before the target passes" in reason


def test_the_start_up_time_moves_the_drop_earlier() -> None:
    estimate = _estimate(90.0, 15.0)
    layer = _layer(0.0, -3000.0)
    times = {}
    for startup in (0, 120):
        config = ForwardDeploymentConfig(max_per_drop=2, observer_startup_s=startup)
        positions, _, report = plan_optimal_deployment(estimate, [], [], config, 6000, 99, coverage_short=True,
                                                       layer=layer)
        assert positions
        _check_ahead(positions, report, 90.0, 15.0, config, config.cpa_ahead_s)
        times[startup] = min(report.drop_times_s)
    assert times[120] <= times[0] + 1e-6
