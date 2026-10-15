"""Optimal drop time accounts for the observer's fall from the release altitude to the sea surface."""
import numpy as np

from aqua_drift import wind as windlib
from aqua_drift.deployment import _offset
from aqua_drift.models import ForwardDeploymentConfig, Position
from aqua_drift.optimal_deployment import FINE_S, LayerAvailability, drop_times

ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)
TERMINAL = 100.0 * 0.3048
KNOT = 0.5144444444444445


def _layer(**kw) -> LayerAvailability:
    args = {"ready_s": 0.0, "position": _offset(ORIGIN, 0.0, -20000.0, 0.0), "speed_kt": 200.0,
            "max_bank_deg": 15.0, "heading_deg": 0.0, "fall_terminal_mps": TERMINAL, "altitude_ft": 1000.0,
            "cruise_altitude_ft": 3000.0, "drop_altitude_ft": 1000.0, "climb_rate_fpm": 2000.0}
    args.update(kw)
    return LayerAvailability(**args)


def _fall(altitude_ft: float) -> float:
    return windlib.fall(altitude_ft, 200.0 * KNOT, 0.0, None, TERMINAL).time_s


def test_release_altitude_follows_the_layers_climb_and_descent() -> None:
    layer = _layer(altitude_ft=3000.0)
    earliest = np.array([30.0, 600.0])
    # dropped as early as it can: descends from 3000 Ft at 2000 ft/min over the flight
    assert np.allclose(layer.release_altitude_ft(earliest, earliest), [2000.0, 1000.0])
    # waiting first: back at the cruise altitude when it leaves, then the same descent
    low = _layer(altitude_ft=1000.0)
    assert np.allclose(low.release_altitude_ft(earliest, earliest + 600.0), [2000.0, 1000.0])
    assert np.allclose(low.release_altitude_ft(earliest, earliest), [1000.0, 1000.0])


def test_drop_time_is_the_fall_time_before_the_observer_must_be_in_the_water() -> None:
    # the target comes within 1000 m of the point at 2000 s
    point = np.array([[0.0, 0.0, 100.0]])
    t = np.arange(0.0, 4000.0 + 1e-9, FINE_S)
    track = np.stack([np.zeros_like(t), (t - 2000.0) - 1000.0, np.full_like(t, 100.0)], axis=1)
    earliest = np.array([300.0])
    args = (point, earliest, [track], 1000.0, 60.0, 0.0, 1.0)
    without = drop_times(*args)
    layer = _layer()
    with_fall = drop_times(*args, fall_s=layer.fall_s)
    fall = _fall(float(layer.release_altitude_ft(earliest, with_fall)[0]))
    assert fall > 5.0
    assert abs((without[0] - with_fall[0]) - fall) < 0.5
    # no fall without one (layer without wind / scenario): unchanged
    assert drop_times(*args, fall_s=_layer(fall_terminal_mps=None).fall_s)[0] == without[0]
    # never before the layer can be there
    late = np.array([without[0] - 1.0])
    assert drop_times(point, late, [track], 1000.0, 60.0, 0.0, 1.0, fall_s=layer.fall_s)[0] == late[0]


def test_planned_drops_are_earlier_by_the_fall() -> None:
    from aqua_drift.models import EstimateMode, PresenceRegion, TrackEstimate, Uncertainty
    from aqua_drift.optimal_deployment import plan_optimal_deployment

    estimate = TrackEstimate(
        mode=EstimateMode.ONLINE, tick=100, observability_status="TRACKING", current_position=ORIGIN,
        depth_ft=400.0, hdg_deg=90.0, through_water_speed_kt=15.0,
        uncertainty=Uncertainty(
            horizontal_major_yd=50, horizontal_minor_yd=20, horizontal_major_axis_deg=0, depth_sigma_ft=100,
            ground_speed_sigma_kt=0.2, through_water_speed_sigma_kt=0.2, cog_sigma_deg=1, hdg_sigma_deg=1,
            bias_sigma_hz=0.01,
        ),
        presence_region=PresenceRegion(probability_pct=90),
    )
    config = ForwardDeploymentConfig(max_per_drop=2)
    near = _offset(ORIGIN, 0.0, -3000.0, 0.0)
    a, _, plain = plan_optimal_deployment(estimate, [], [], config, 6000, 99, coverage_short=True,
                                          layer=_layer(fall_terminal_mps=None, position=near))
    b, _, falling = plan_optimal_deployment(estimate, [], [], config, 6000, 99, coverage_short=True,
                                            layer=_layer(position=near))
    assert a == b and a
    diffs = [p - f for p, f in zip(plain.drop_times_s, falling.drop_times_s, strict=True)]
    assert all(d >= -1e-6 for d in diffs)
    # the drop the route does not hold back (scheduled before detection) is the fall time earlier
    assert abs(max(diffs) - _fall(1000.0)) < 1.0
