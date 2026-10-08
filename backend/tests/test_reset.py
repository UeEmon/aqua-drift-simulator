"""Runtime reset (初期状態を適用して再スタート): nothing of the previous run may reach the new
one, neither through what the API keeps nor through posts the containers computed before the
reset and deliver after it."""
import pytest
from fastapi.testclient import TestClient

from aqua_drift.api import app
from aqua_drift.models import (
    DeploymentRequest,
    DopplerBatch,
    EstimateMode,
    EstimatorOutput,
    ObserverPlacement,
    ObserverState,
    Position,
    PresenceRegion,
    ScenarioConfig,
    TrackEstimate,
    Velocity,
)
from aqua_drift.physics import initial_target
from aqua_drift.state import SimulationState, StaleGeneration

OLD = Position(latitude=35.2, longitude=140.3, depth_ft=500.0)


def estimate(tick: int, position: Position = OLD) -> TrackEstimate:
    return TrackEstimate(
        mode=EstimateMode.ONLINE, tick=tick, observability_status="TRACKING", current_position=position,
        depth_ft=position.depth_ft, presence_region=PresenceRegion(probability_pct=90),
    )


def target_at(config: ScenarioConfig, tick: int, position: Position):
    state = initial_target(config)
    state.tick = tick
    state.position = position
    return state


@pytest.mark.asyncio
async def test_reset_clears_everything_of_the_old_run() -> None:
    config = ScenarioConfig()
    config.layer.enabled = True
    sim = SimulationState(config)
    await sim.set_tick(50)
    await sim.set_target(target_at(config, 50, OLD))
    await sim.set_estimator_output(EstimatorOutput(tick=50, estimates=[estimate(50)], cpa=[]))
    await sim.queue_deployment(DeploymentRequest(tick=50, positions=[OLD], reason="forward"))
    await sim.queue_placement(ObserverPlacement(position=OLD))  # operator drop: a layer task
    assert len(sim.tasks) == 2

    await sim.reset_runtime()

    snapshot = await sim.snapshot()
    assert snapshot.target is None and snapshot.estimates == [] and snapshot.current_estimate is None
    assert snapshot.deployment.tasks == [] and snapshot.deployment.pending_placements == 0
    assert snapshot.deployment.layer is None and snapshot.deployment.history == []
    assert sim.estimator_tick == -1
    layer = await sim.layer_feed()
    assert layer.datum == config.target.initial_position  # orbit centre: not the old estimate
    feed = await sim.deployment_feed()
    assert feed.estimates == [] and feed.pending_positions == [] and feed.generation == 1


@pytest.mark.asyncio
async def test_posts_computed_before_the_reset_are_rejected() -> None:
    config = ScenarioConfig()
    config.layer.enabled = False
    sim = SimulationState(config)
    await sim.set_tick(50)
    old_generation, old_run = sim.generation, sim.estimation.run_id
    await sim.reset_runtime()

    with pytest.raises(StaleGeneration):
        await sim.set_estimator_output(
            EstimatorOutput(tick=51, estimates=[estimate(51)], cpa=[]), old_generation, old_run
        )
    with pytest.raises(StaleGeneration):
        await sim.set_target(target_at(config, 51, OLD), old_generation)
    with pytest.raises(StaleGeneration):
        await sim.set_observer(
            ObserverState(observer_id="a", tick=51, position=OLD, ground_velocity=Velocity()), old_generation
        )
    with pytest.raises(StaleGeneration):
        await sim.add_batch(DopplerBatch(tick=51, observations=[]), old_generation)
    with pytest.raises(StaleGeneration):
        await sim.queue_deployment(DeploymentRequest(tick=51, positions=[OLD], reason="old"), generation=old_generation)

    snapshot = await sim.snapshot()
    assert snapshot.estimates == [] and snapshot.target is None and snapshot.observers == []
    assert snapshot.deployment.pending_placements == 0 and not sim.batches

    # the same posts for the current run are accepted
    new = Position(latitude=34.0, longitude=139.0, depth_ft=200.0)
    await sim.set_target(target_at(config, 52, new), sim.generation)
    await sim.set_estimator_output(
        EstimatorOutput(tick=52, estimates=[estimate(52, new)], cpa=[]), sim.generation, sim.estimation.run_id
    )
    snapshot = await sim.snapshot()
    assert snapshot.target.position == new and snapshot.estimates[0].current_position == new


@pytest.mark.asyncio
async def test_output_of_a_replaced_estimation_run_is_rejected() -> None:
    sim = SimulationState(ScenarioConfig())
    old_run = sim.estimation.run_id
    await sim.start_estimation()
    with pytest.raises(StaleGeneration):
        await sim.set_estimator_output(EstimatorOutput(tick=1, estimates=[estimate(1)], cpa=[]), sim.generation, old_run)
    assert (await sim.snapshot()).estimates == []


def test_api_drops_stale_posts_without_failing_the_container() -> None:
    with TestClient(app) as client:
        old = client.get("/api/snapshot").json()["generation"]
        assert client.post("/api/reset").status_code == 200
        response = client.post(
            "/internal/estimate",
            params={"generation": old, "run_id": 0},
            json=EstimatorOutput(tick=1, estimates=[estimate(1)], cpa=[]).model_dump(mode="json"),
        )
        assert response.status_code == 200 and "ignored" in response.json()
        assert client.get("/api/snapshot").json()["estimates"] == []
