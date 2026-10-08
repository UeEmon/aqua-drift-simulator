import math
import random

import pytest
from pydantic import ValidationError

from aqua_drift import wind as windlib
from aqua_drift.deployment import _offset
from aqua_drift.layer import advance, initial_state
from aqua_drift.models import (
    DeploymentRequest,
    DropTask,
    LayerConfig,
    LayerFeed,
    Position,
    ScenarioConfig,
    WindConfig,
    WindLevel,
)
from aqua_drift.physics import local_offset_m
from aqua_drift.state import SimulationState

DATUM = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)
KT = windlib.KNOT_TO_MPS


def test_wind_profile_every_1000_ft_from_the_sea_surface_to_30000_ft() -> None:
    config = WindConfig()
    assert [level.altitude_ft for level in config.levels] == list(range(0, 30001, 1000))
    profile = windlib.WindProfile(config)
    # 10 kt from 250 deg at the surface, 70 kt from 290 deg at 30,000 ft
    direction, speed = windlib.wind_from(*[v / KT for v in profile.at(0.0)])
    assert direction == pytest.approx(250.0, abs=0.1) and speed == pytest.approx(10.0, abs=0.01)
    direction, speed = windlib.wind_from(*[v / KT for v in profile.at(30000.0)])
    assert direction == pytest.approx(290.0, abs=0.1) and speed == pytest.approx(70.0, abs=0.01)
    # linear between the levels (east / north components), the top level above 30,000 ft
    custom = WindConfig(levels=[WindLevel(altitude_ft=0, direction_deg=0, speed_kt=0),
                                WindLevel(altitude_ft=2000, direction_deg=270, speed_kt=20)])
    east, north = windlib.WindProfile(custom).at(1000.0)
    assert east / KT == pytest.approx(10.0) and north == pytest.approx(0.0, abs=1e-9)
    assert windlib.WindProfile(custom).at(5000.0)[0] / KT == pytest.approx(20.0)
    # off: no wind at any altitude
    assert windlib.WindProfile(WindConfig(enabled=False)).at(10000.0) == (0.0, 0.0)
    # levels only on the 1,000 ft grid, up to 30,000 ft, each altitude once
    with pytest.raises(ValidationError):
        WindLevel(altitude_ft=1500, direction_deg=0, speed_kt=10)
    with pytest.raises(ValidationError):
        WindLevel(altitude_ft=31000, direction_deg=0, speed_kt=10)
    with pytest.raises(ValidationError):
        WindConfig(levels=[WindLevel(altitude_ft=0, direction_deg=0, speed_kt=1)] * 2)


def test_free_fall_reaches_terminal_velocity_and_drifts_with_the_wind() -> None:
    terminal = 30.0
    still = windlib.fall(1000.0, 0.0, 0.0, None, terminal)
    # 304.8 m: about 3 s to reach the terminal velocity, then 30 m/s
    assert 10.5 < still.time_s < 12.5 and abs(still.east_m) < 1e-9
    # thrown forward by the layer's speed (slowed down by the drag)
    thrown = windlib.fall(1000.0, 100.0, 0.0, None, terminal)
    assert 150.0 < thrown.east_m < 330.0 and abs(thrown.north_m) < 1e-9
    # a uniform wind carries it downwind by about the fall time less the drag lag
    drifted = windlib.fall(1000.0, 0.0, 0.0, windlib.uniform(0.0, 10.0), terminal)
    assert 0.6 * 10.0 * still.time_s < drifted.north_m < 10.0 * still.time_s


@pytest.mark.parametrize("altitude", [1000.0, 10000.0, 30000.0])
def test_mean_wind_from_the_entry_point_and_the_no_wind_prediction(altitude: float) -> None:
    profile = windlib.WindProfile(WindConfig())
    terminal = profile.terminal_mps
    # a uniform wind is recovered exactly
    actual = windlib.fall(altitude, 90.0, 40.0, windlib.uniform(-6.0, 3.0), terminal)
    mean = windlib.estimate_mean_wind(altitude, 90.0, 40.0, actual.east_m, actual.north_m, terminal)
    assert mean.east_mps == pytest.approx(-6.0, abs=0.02) and mean.north_mps == pytest.approx(3.0, abs=0.02)
    # the default profile: close to its vector mean from the drop altitude to the sea surface
    actual = windlib.fall(altitude, 90.0, 40.0, profile.at, terminal)
    mean = windlib.estimate_mean_wind(altitude, 90.0, 40.0, actual.east_m, actual.north_m, terminal)
    direction, speed = windlib.wind_from(mean.east_mps / KT, mean.north_mps / KT)
    true_direction, true_speed = windlib.wind_from(*[v / KT for v in profile.mean(altitude)])
    assert abs(direction - true_direction) < 3.0
    assert speed == pytest.approx(true_speed, rel=0.1)


def _single_drop(config: LayerConfig, estimate, seed: int = 3):
    """One drop 8.5 km away; returns the release and the miss (m) of the entry point."""
    rng = random.Random(seed)
    state = initial_state(config, DATUM, 0, rng)
    point = _offset(DATUM, 8000.0, -3000.0, 300.0)
    task = DropTask(task_id=1, created_tick=0, source="forward", position=point, status="APPROVED")
    for tick in range(1, 900):
        feed = LayerFeed(tick=tick, config=config, tasks=[task], datum=DATUM, current_east_kt=1.0,
                         current_north_kt=0.3, wind=WindConfig(), wind_estimate=estimate)
        state, update = advance(feed, state, rng)
        if 1 in update.completed:
            drop = update.releases[1]
            assert update.completed[1] == drop.splash_position
            east, north, _ = local_offset_m(drop.planned_position, drop.splash_position)
            return drop, math.hypot(east, north)
        task = task.model_copy(update={"position": update.task_positions[1]})
    raise AssertionError("not dropped")


def test_layer_flies_in_three_dimensions_in_the_wind_at_its_altitude() -> None:
    config = LayerConfig(cruise_altitude_ft=20000.0)
    rng = random.Random(5)
    state = initial_state(config, DATUM, 0, rng).model_copy(update={"altitude_ft": 3000.0})
    start = state.position
    for tick in range(1, 121):
        state, _ = advance(LayerFeed(tick=tick, config=config, tasks=[], datum=DATUM, wind=WindConfig()), state, rng)
    # climbs to the cruise altitude at the climb rate (2,000 ft/min)
    assert state.altitude_ft == pytest.approx(3000.0 + 2 * 2000.0)
    # ground velocity = air velocity + the wind at its altitude (west wind: carried east)
    wind_e, wind_n = windlib.WindProfile(WindConfig()).at(state.altitude_ft)
    v = state.speed_kt * KT
    heading = math.radians(state.heading_deg)
    ground = math.hypot(v * math.sin(heading) + wind_e, v * math.cos(heading) + wind_n) / KT
    assert state.ground_speed_kt == pytest.approx(ground, abs=0.01)
    direction, speed = windlib.wind_from(wind_e / KT, wind_n / KT)
    assert state.wind_direction_deg == pytest.approx(direction) and state.wind_speed_kt == pytest.approx(speed)
    # the same second with and without wind: the wind carries the layer by its own velocity
    feed = LayerFeed(tick=121, config=config, tasks=[], datum=DATUM, wind=WindConfig())
    windy, _ = advance(feed, state, random.Random(1))
    still, _ = advance(feed.model_copy(update={"wind": WindConfig(enabled=False)}), state, random.Random(1))
    east, north, _ = local_offset_m(still.position, windy.position)
    wind_e, wind_n = windlib.WindProfile(WindConfig()).at(state.altitude_ft)
    assert east == pytest.approx(wind_e, abs=0.05) and north == pytest.approx(wind_n, abs=0.05)
    assert local_offset_m(start, state.position) != (0.0, 0.0, 0.0)


def test_release_point_is_corrected_with_the_estimated_mean_wind() -> None:
    config = LayerConfig(cruise_altitude_ft=10000.0, drop_altitude_ft=10000.0)
    # first drop: no estimate yet, released for the no-wind free fall -> carried off by the wind
    drop, miss = _single_drop(config, None)
    assert drop.altitude_ft == pytest.approx(10000.0) and 80.0 < drop.fall_time_s < 130.0
    profile = windlib.WindProfile(WindConfig())
    estimate = windlib.wind_estimate(drop, profile)
    assert miss > 700.0 and estimate.offset_yd * 0.9144 == pytest.approx(miss, rel=0.15)
    assert abs(estimate.direction_deg - estimate.true_direction_deg) < 3.0
    assert estimate.speed_kt == pytest.approx(estimate.true_speed_kt, rel=0.1)
    # next drop: released upwind by the drift of the estimated mean wind
    _, corrected = _single_drop(config, estimate)
    assert corrected < 0.3 * miss
    # correction off: as without an estimate
    _, uncorrected = _single_drop(config.model_copy(update={"wind_correction": False}), estimate)
    assert uncorrected > 0.8 * miss


@pytest.mark.asyncio
async def test_observer_enters_the_water_after_the_fall_and_the_next_drop_is_corrected() -> None:
    config = ScenarioConfig()
    config.layer.cruise_altitude_ft = config.layer.drop_altitude_ft = 10000.0
    sim = SimulationState(config)
    # far apart: the second drop is released after the first observer is in the water
    points = [_offset(DATUM, 7000.0, 2000.0, 300.0), _offset(DATUM, -12000.0, 9000.0, 300.0)]
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=points, reason="plan"))
    rng = random.Random(2)
    state = None
    falling_seen = False
    for _ in range(1500):
        sim.tick += 1
        feed = await sim.layer_feed()
        if state is None:
            state = initial_state(feed.config, feed.datum, feed.tick - 1, rng)
        state, update = advance(feed, state, rng)
        await sim.set_layer_update(update)
        status = (await sim.snapshot()).deployment
        falling_seen = falling_seen or status.falling > 0
        if len(status.wind_estimates) == 2:
            break
    status = (await sim.snapshot()).deployment
    assert falling_seen and status.falling == 0 and status.pending_placements == 2
    first, second = status.wind_estimates
    tasks = {t.task_id: t for t in status.tasks}
    assert tasks[first.task_id].miss_yd == first.miss_yd and tasks[first.task_id].splash_tick == first.tick
    # the first drop is carried off by the wind; the second is released for the estimated mean wind
    assert first.miss_yd > 700.0
    assert second.miss_yd < 0.4 * first.miss_yd
    assert tasks[second.task_id].done_tick >= first.tick  # released after the first was in the water
    # the layer is handed the latest estimate for the next release
    assert (await sim.layer_feed()).wind_estimate == second
