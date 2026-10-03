from aqua_drift.models import EstimateMode, ObserverState, Position, ScenarioConfig, Velocity
from aqua_drift.services.estimator import build_estimate


def test_estimator_does_not_assert_position_without_direction() -> None:
    config = ScenarioConfig()
    data = {
        "tick": 10,
        "config": config.model_dump(mode="json"),
        "target": None,
        "observers": [
            {
                "state": ObserverState(
                    observer_id="observer-a",
                    tick=10,
                    position=Position(latitude=35.0, longitude=140.0, depth_ft=100.0),
                    ground_velocity=Velocity(),
                ).model_dump(mode="json"),
                "registered_tick": 0,
                "last_tick": 10,
            }
        ],
        "doppler": [
            {
                "observer_id": "observer-a",
                "tick": 10,
                "observed_frequency_hz": 400.1,
                "recognized_frequency_hz": 400.0,
                "relative_radial_speed_kt": 1.0,
                "relative_speed_kt": 3.0,
                "slant_range_yd": 2000.0,
                "is_new_closest": True,
            }
        ],
        "estimates": [],
        "archived_observer_ids": [],
    }

    estimate = build_estimate(data, EstimateMode.ONLINE)

    assert estimate.current_position is None
    assert estimate.observability_status == "UNOBSERVABLE_WITHOUT_DIRECTION"
    assert estimate.metadata["direction_input_available"] is False
    assert estimate.presence_region.components[0].radius_yd == 2000.0
