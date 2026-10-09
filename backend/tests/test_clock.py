import pytest
from fastapi.testclient import TestClient

from aqua_drift.api import app
from aqua_drift.models import DopplerBatch, EstimatorOutput, ObserverState, Position, ScenarioConfig
from aqua_drift.physics import initial_target
from aqua_drift.state import ESTIMATOR_SLACK_TICKS, SimulationState


def observer(tick: int) -> ObserverState:
    position = Position(latitude=35.0, longitude=140.0, depth_ft=100.0)
    return ObserverState(observer_id="one", tick=tick, position=position)


def target(config: ScenarioConfig, tick: int):
    state = initial_target(config)
    state.tick = tick
    return state


@pytest.mark.asyncio
async def test_clock_waits_for_every_worker_of_the_current_tick() -> None:
    config = ScenarioConfig(doppler_interval_seconds=1)
    state = SimulationState(config, autostart_estimation=False)
    await state.set_tick(5)
    assert not (await state.clock_status()).synced  # no target yet
    await state.set_target(target(config, 5))
    await state.set_observer(observer(4))
    assert not (await state.clock_status()).synced  # observer still on the previous tick
    await state.set_observer(observer(5))
    assert not (await state.clock_status()).synced  # Doppler epoch 5 not published
    await state.add_batch(DopplerBatch(tick=5, observations=[]))
    assert (await state.clock_status()).synced


@pytest.mark.asyncio
async def test_clock_waits_for_a_lagging_estimator() -> None:
    config = ScenarioConfig(doppler_interval_seconds=1)
    state = SimulationState(config, autostart_estimation=True)
    tick = ESTIMATOR_SLACK_TICKS + 5
    await state.set_tick(tick)
    await state.set_target(target(config, tick))
    await state.add_batch(DopplerBatch(tick=tick, observations=[]))
    assert not (await state.clock_status()).synced
    await state.set_estimator_output(EstimatorOutput(tick=tick - 1, estimates=[], cpa=[]))
    assert (await state.clock_status()).synced


def test_time_scale_round_trip_and_limits() -> None:
    with TestClient(app) as client:
        assert client.get("/api/clock").json() == {"time_scale": 1.0}
        assert client.put("/api/clock", json={"time_scale": 4.5}).json() == {"time_scale": 4.5}
        assert client.get("/internal/clock").json()["time_scale"] == 4.5
        assert client.get("/internal/sim-state").json()["time_scale"] == 4.5
        assert client.put("/api/clock", json={"time_scale": 0}).status_code == 200  # pause
        assert client.put("/api/clock", json={"time_scale": -1}).status_code == 422
        assert client.put("/api/clock", json={"time_scale": 101}).status_code == 422
        client.put("/api/clock", json={"time_scale": 1})


@pytest.mark.asyncio
async def test_first_tick_fixes_the_time_of_day_of_tick_zero() -> None:
    state = SimulationState(ScenarioConfig(), autostart_estimation=False)
    assert (await state.snapshot()).epoch_s is None
    await state.set_tick(3, wall_s=1_000_003.0)
    await state.set_tick(4, wall_s=1_000_009.0)  # later ticks never move it
    assert (await state.clock_status()).epoch_s == 1_000_000.0
    assert (await state.snapshot()).epoch_s == 1_000_000.0


def test_schedule_follows_the_system_time() -> None:
    from aqua_drift.services.clock import Schedule

    schedule = Schedule()
    schedule.anchor(0, 1.0, 500.0)
    assert schedule.due(10) == 510.0  # tick n due at anchor + n s, however late earlier ticks were
    schedule.anchor(10, 4.0, 520.0)  # speed change: re-anchored at the current tick and time
    assert schedule.due(14) == 521.0
    restarted = Schedule()
    restarted.anchor(25, 1.0, 525.4, epoch_s=500.0)  # clock restart while still on system time
    assert restarted.due(26) == 526.0


def test_late_ticks_are_caught_up_only_at_real_time_speed() -> None:
    from aqua_drift.services.clock import MAX_LAG_S, Schedule, next_due

    schedule = Schedule()
    assert next_due(schedule, 0, 1.0, 100.0, None) == 101.0
    # 1x, 5 s late (slow containers): tick 11 stays due at 111 -> caught up to the system time
    assert next_due(schedule, 10, 1.0, 116.0, None) == 111.0
    # 1x, more than MAX_LAG_S late (host suspended): continue from now instead of a burst
    late = 111.0 + MAX_LAG_S + 5
    assert next_due(schedule, 10, 1.0, late, None) == late
    # 10x: the simulation time is not the system time, so a late tick is never caught up and
    # the clock never runs faster than the set speed
    fast = Schedule()
    assert next_due(fast, 0, 10.0, 200.0, None) == 200.1
    assert next_due(fast, 1, 10.0, 200.2, None) == 200.2  # on time within one interval
    assert next_due(fast, 2, 10.0, 205.0, None) == 205.0  # 5 s stall: no backlog of 48 ticks
    assert abs(next_due(fast, 3, 10.0, 205.0, None) - 205.1) < 1e-9
    # 0.5x the same
    slow = Schedule()
    assert next_due(slow, 0, 0.5, 300.0, None) == 302.0
    assert next_due(slow, 1, 0.5, 310.0, None) == 310.0
