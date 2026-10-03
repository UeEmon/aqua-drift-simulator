import math

import pytest

from aqua_drift.models import ObserverState, Position, ScenarioConfig, Velocity
from aqua_drift.physics import (
    advance_observer,
    advance_target,
    current_at,
    doppler_observation,
    initial_target,
)


def test_current_field_affine_gradient() -> None:
    config = ScenarioConfig()
    config.current_field.gradient_per_nm = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    reference = config.current_field.reference_position
    one_nm_north = Position(
        latitude=reference.latitude + 1.0 / 60.0,
        longitude=reference.longitude,
        depth_ft=reference.depth_ft,
    )
    result = current_at(config.current_field, one_nm_north)
    assert result.north_kt == pytest.approx(config.current_field.base_velocity.north_kt + 1.0, rel=0.01)


def test_target_uses_rate_limited_changes() -> None:
    config = ScenarioConfig()
    state = initial_target(config)
    config.target.desired_hdg_deg = 180.0
    config.target.desired_through_water_speed_kt = 12.0
    config.target.desired_depth_ft = 600.0
    config.target.hdg_rate_deg_per_sec = 2.0
    config.target.speed_rate_kt_per_sec = 0.5
    config.target.depth_rate_ft_per_sec = 4.0

    updated = advance_target(config, state, 1.0, Velocity())

    assert updated.hdg_deg == pytest.approx(92.0)
    assert updated.through_water_speed_kt == pytest.approx(8.5)
    assert updated.position.depth_ft == pytest.approx(504.0)


def test_observer_drifts_only_with_current() -> None:
    config = ScenarioConfig()
    observer = ObserverState(
        observer_id="observer-a",
        tick=0,
        position=Position(latitude=35.0, longitude=140.0, depth_ft=100.0),
        ground_velocity=Velocity(),
    )
    updated = advance_observer(config, observer, 10.0, Velocity(east_kt=2.0))
    assert updated.position.longitude > observer.position.longitude
    assert updated.position.latitude == pytest.approx(observer.position.latitude)


def test_doppler_has_no_bearing_field() -> None:
    config = ScenarioConfig()
    target = initial_target(config)
    observer = ObserverState(
        observer_id="observer-a",
        tick=0,
        position=Position(latitude=35.0, longitude=140.01, depth_ft=100.0),
        ground_velocity=Velocity(),
    )
    observation, truth = doppler_observation(config, target, observer)
    assert truth.slant_range_yd > 0
    assert observation.detected
    assert math.isfinite(observation.observed_frequency_hz)
    dumped = observation.model_dump()
    assert "bearing" not in dumped
    assert "slant_range_yd" not in dumped  # range is truth, never an observation


def test_out_of_range_is_explicit_non_detection() -> None:
    config = ScenarioConfig(max_slant_range_yd=100.0)
    target = initial_target(config)
    observer = ObserverState(
        observer_id="observer-a",
        tick=0,
        position=Position(latitude=35.0, longitude=140.2, depth_ft=100.0),
        ground_velocity=Velocity(),
    )
    observation, _ = doppler_observation(config, target, observer)
    assert observation.detected is False
    assert observation.observed_frequency_hz is None


def test_doppler_frequency_sign() -> None:
    config = ScenarioConfig()
    target = initial_target(config)  # heading east at 8 kt
    ahead = ObserverState(
        observer_id="ahead",
        tick=0,
        position=Position(latitude=35.0, longitude=140.02, depth_ft=500.0),
        ground_velocity=config.current_field.base_velocity,
    )
    observation, truth = doppler_observation(config, target, ahead)
    assert observation.observed_frequency_hz > config.source.source_frequency_hz  # closing
    assert truth.relative_radial_speed_kt == pytest.approx(8.0, abs=0.05)
