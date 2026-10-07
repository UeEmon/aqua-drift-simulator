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
