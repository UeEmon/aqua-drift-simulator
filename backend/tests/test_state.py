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
    assert snapshot.archived_observer_ids == ["one#0"]  # slot one, session 0


@pytest.mark.asyncio
async def test_observer_expires_after_three_hours() -> None:
    state = SimulationState(ScenarioConfig(max_observation_seconds=10800))
    await state.set_observer(observer("one", 0))
    with pytest.raises(ObserverRejected):
        await state.set_observer(observer("one", 10800))
    snapshot = await state.snapshot()
    assert "one#0" in snapshot.archived_observer_ids


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
    config.layer.enabled = False  # immediate placements (the layer flow is tested in test_layer.py)
    state = SimulationState(config)
    assert await state.assign_position("a") is not None
    assert await state.assign_position("b") is not None
    assert await state.assign_position("c") is None  # standby
    assert (await state.snapshot()).deployment.standby_count == 1
    drop = Position(latitude=35.05, longitude=140.05, depth_ft=300.0)
    await state.queue_deployment(DeploymentRequest(tick=10, positions=[drop], reason="test"))
    assigned = await state.assign_position("c")
    assert (assigned.latitude, assigned.longitude, assigned.depth_ft) == (35.05, 140.05, 300.0)
    status = (await state.snapshot()).deployment
    assert status.last_deploy_tick == 10 and len(status.history) == 1 and status.standby_count == 0
    await state.queue_deployment(DeploymentRequest(tick=20, positions=[drop], reason="test"))
    await state.reset_runtime()
    status = (await state.snapshot()).deployment
    assert status.pending_placements == 0 and status.history == []  # automatic drops dropped


@pytest.mark.asyncio
async def test_observer_slot_is_reused_with_new_session() -> None:
    state = SimulationState(ScenarioConfig(observer_limit=1))
    first = await state.assign_position("obs-01")
    assert first is not None and first.session == 0
    await state.set_observer(observer("obs-01", 0))
    await state.assign_position("obs-02")
    await state.set_observer(observer("obs-02", 5))  # limit 1 -> obs-01 evicted (oldest)
    snapshot = await state.snapshot()
    assert snapshot.archived_observer_ids == ["obs-01#0"]
    # the old obs-01 container may still post once: rejected, so it exits
    with pytest.raises(ObserverRejected):
        await state.set_observer(observer("obs-01", 6))
    # a new container for slot obs-01 gets a new session and may observe again
    reused = await state.assign_position("obs-01")
    assert reused is not None and reused.session == 1
    state_1 = observer("obs-01", 7)
    state_1.session = 1
    await state.set_observer(state_1)  # evicts obs-02 (limit 1)
    snapshot = await state.snapshot()
    assert [r.state.observer_id for r in snapshot.observers] == ["obs-01"]
    assert snapshot.archived_observer_ids == ["obs-01#0", "obs-02#0"]  # history kept


@pytest.mark.asyncio
async def test_evict_oldest_and_orchestrator_feed() -> None:
    config = ScenarioConfig()
    config.deployment.initial_count = 2
    state = SimulationState(config)
    for name in ("obs-01", "obs-02"):
        await state.assign_position(name)
        await state.set_observer(observer(name, 0))
    feed = await state.orchestrator_feed()
    assert feed.active_ids == ["obs-01", "obs-02"] and feed.initial_remaining == 0
    assert feed.limit == 99
    assert await state.evict_oldest(1) == ["obs-01"]
    assert (await state.orchestrator_feed()).active_ids == ["obs-02"]
