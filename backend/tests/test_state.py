import pytest

from aqua_drift.models import ObserverState, Position, ScenarioConfig, Velocity
from aqua_drift.state import ObserverRejected, SimulationState


def observer(observer_id: str, tick: int = 0) -> ObserverState:
    return ObserverState(
        observer_id=observer_id,
        tick=tick,
        position=Position(latitude=35.0, longitude=140.0, depth_ft=100.0),
        ground_velocity=Velocity(),
    )


@pytest.mark.asyncio
async def test_fifo_evicts_oldest_and_retains_archive() -> None:
    state = SimulationState(ScenarioConfig(observer_limit=2))
    await state.set_observer(observer("one"))
    await state.set_observer(observer("two"))
    evicted = await state.set_observer(observer("three"))

    snapshot = await state.snapshot()
    assert evicted == "one"
    assert [item.state.observer_id for item in snapshot.observers] == ["two", "three"]
    assert snapshot.archived_observer_ids == ["one"]


@pytest.mark.asyncio
async def test_observer_expires_after_three_hours() -> None:
    state = SimulationState(ScenarioConfig(max_observation_seconds=10800))
    await state.set_observer(observer("one", 0))
    with pytest.raises(ObserverRejected):
        await state.set_observer(observer("one", 10800))
    snapshot = await state.snapshot()
    assert "one" in snapshot.archived_observer_ids


@pytest.mark.asyncio
async def test_runtime_reset_lets_observers_reregister() -> None:
    state = SimulationState(ScenarioConfig(observer_limit=2))
    await state.set_observer(observer("one", 100))
    await state.reset_runtime()
    snapshot = await state.snapshot()
    assert snapshot.observers == []  # records cleared: 3 h limit restarts on re-registration
    assert snapshot.archived_observer_ids == []  # a reset never archives anyone
    await state.set_observer(observer("one", 101))
    assert [r.registered_tick for r in (await state.snapshot()).observers] == [101]


@pytest.mark.asyncio
async def test_estimation_start_stop_runs() -> None:
    state = SimulationState(ScenarioConfig(), autostart_estimation=False)
    assert (await state.snapshot()).estimation.running is False
    await state.set_tick(40)
    started = await state.start_estimation()
    assert started.running and started.run_id == 1 and started.started_tick == 40
    feed = await state.estimator_feed(-1)
    assert feed.estimation.run_id == 1
    await state.set_tick(90)
    stopped = await state.stop_estimation()
    assert not stopped.running and stopped.stopped_tick == 90
    restarted = await state.start_estimation()
    assert restarted.run_id == 2 and restarted.started_tick == 90


@pytest.mark.asyncio
async def test_reset_replaces_default_observers_around_new_target() -> None:
    from aqua_drift.physics import local_offset_m

    state = SimulationState(ScenarioConfig())
    first = await state.assign_position("a")
    config = ScenarioConfig()
    config.target.initial_position = Position(latitude=34.0, longitude=139.0, depth_ft=300.0)
    await state.set_config(config)
    await state.reset_runtime(replace_observers=True)
    second = await state.assign_position("a")
    assert second != first
    east, north, _ = local_offset_m(config.target.initial_position, second)
    assert (east**2 + north**2) ** 0.5 < 4000
    await state.reset_runtime(replace_observers=False)
    assert await state.assign_position("a") == second


@pytest.mark.asyncio
async def test_standby_observers_take_forward_deployments() -> None:
    from aqua_drift.models import DeploymentRequest

    config = ScenarioConfig()
    config.deployment.initial_count = 2
    state = SimulationState(config)
    assert await state.assign_position("a") is not None
    assert await state.assign_position("b") is not None
    assert await state.assign_position("c") is None  # standby
    assert (await state.snapshot()).deployment.standby_count == 1
    drop = Position(latitude=35.05, longitude=140.05, depth_ft=300.0)
    await state.queue_deployment(DeploymentRequest(tick=10, positions=[drop], reason="test"))
    assert await state.assign_position("c") == drop
    status = (await state.snapshot()).deployment
    assert status.last_deploy_tick == 10 and len(status.history) == 1 and status.standby_count == 0
    await state.queue_deployment(DeploymentRequest(tick=20, positions=[drop], reason="test"))
    await state.reset_runtime()
    status = (await state.snapshot()).deployment
    assert status.pending_placements == 0 and status.history == []  # automatic drops dropped
