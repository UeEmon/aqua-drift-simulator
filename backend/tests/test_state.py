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
async def test_runtime_reset_retains_active_observers() -> None:
    state = SimulationState(ScenarioConfig(observer_limit=2))
    await state.set_observer(observer("one"))
    await state.reset_runtime()
    snapshot = await state.snapshot()
    assert [item.state.observer_id for item in snapshot.observers] == ["one"]
