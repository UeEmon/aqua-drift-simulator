import math
import random

import pytest

from aqua_drift.deployment import _offset
from aqua_drift.layer import advance, initial_state, orbit_radius_m, step, turn_radius_m
from aqua_drift.models import (
    DeploymentRequest,
    LayerConfig,
    ObserverPlacement,
    Position,
    ScenarioConfig,
)
from aqua_drift.physics import local_offset_m
from aqua_drift.state import SimulationState

DATUM = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)


def test_turn_radius_and_orbit_respect_the_bank_limit() -> None:
    config = LayerConfig()
    # 200 kt, 15 deg bank: r = V^2 / (g tan 15) ~ 4.0 km
    assert 3900 < turn_radius_m(200.0, 15.0) < 4100
    rng = random.Random(3)
    state = initial_state(config, DATUM, 0, rng)
    worst = 0.0
    for _ in range(1200):
        state, arrived = step(state, config, None, None, DATUM, rng)
        worst = max(worst, abs(state.bank_deg))
        assert not arrived
    east, north, _ = local_offset_m(DATUM, state.position)
    radius = orbit_radius_m(config, state.speed_kt)
    assert worst <= 15.0 + 1e-6
    assert abs(math.hypot(east, north) - radius) < 0.05 * radius  # settled on the circle
    assert 150.0 <= state.speed_kt <= 250.0


def test_layer_reaches_points_ahead_and_behind_with_bank_limit() -> None:
    config = LayerConfig()
    rng = random.Random(5)
    start = initial_state(config, DATUM, 0, rng)
    speeds = set()
    for east, north in ((12000.0, -4000.0), (-3000.0, 0.0), (500.0, -800.0)):
        point = _offset(start.position, east, north, 300.0)
        state, worst = start, 0.0
        for task_id in range(1, 1500):
            state, arrived = step(state, config, point, 7, DATUM, rng)
            worst = max(worst, abs(state.bank_deg))
            if arrived:
                break
        assert arrived, (east, north)
        assert worst <= 15.0 + 1e-6
        speeds.add(round(state.speed_kt))
        assert 150.0 <= state.speed_kt <= 250.0
        # flight time is at least the straight distance at the leg speed
        assert task_id >= math.hypot(east, north) / (250 * 0.5144) - 2


async def _layer_steps(sim: SimulationState, seconds: int, rng: random.Random, state=None):
    """What the layer container does every tick, in process."""
    for _ in range(seconds):
        sim.tick += 1
        feed = await sim.layer_feed()
        if state is None:
            state = initial_state(feed.config, feed.datum, feed.tick - 1, rng)
        state, update = advance(feed, state, rng)
        await sim.set_layer_update(update)
    return state


@pytest.mark.asyncio
async def test_auto_approval_layer_flies_and_lays_the_observer() -> None:
    sim = SimulationState(ScenarioConfig())
    drop = _offset(DATUM, 6000.0, 2000.0, 1000.0)
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[drop], reason="plan"))
    status = (await sim.snapshot()).deployment
    assert status.tasks[-1].status == "APPROVED"  # automatic approval (default)
    assert status.pending_placements == 0  # not in the water yet
    feed = await sim.deployment_feed()
    assert len(feed.pending_positions) == 1  # the planner counts the open task
    rng = random.Random(1)
    state = await _layer_steps(sim, 20, rng)
    status = (await sim.snapshot()).deployment
    assert status.layer.mode == "TRANSIT" and status.tasks[-1].eta_s > 0
    for _ in range(40):
        state = await _layer_steps(sim, 10, rng, state)
        if (await sim.snapshot()).deployment.tasks[-1].status == "DONE":
            break
    status = (await sim.snapshot()).deployment
    task = status.tasks[-1]
    assert task.status == "DONE"
    assert status.pending_placements == 1  # now the observer container is started
    assert task.position.depth_ft == 1000.0
    assigned = await sim.assign_position("obs-09")
    assert (assigned.latitude, assigned.longitude) == (task.position.latitude, task.position.longitude)
    assert status.layer.mode == "ORBIT" or status.layer.task_id != task.task_id


@pytest.mark.asyncio
async def test_manual_approval_proposes_then_operator_decides() -> None:
    config = ScenarioConfig()
    config.layer.approval = "manual"
    sim = SimulationState(config)
    a = _offset(DATUM, 5000.0, 0.0, 500.0)
    b = _offset(DATUM, 5000.0, 3000.0, 60.0)
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[a, b], reason="plan"))
    tasks = (await sim.snapshot()).deployment.tasks
    assert [t.status for t in tasks] == ["PROPOSED", "PROPOSED"]
    assert (await sim.layer_feed()).tasks == []  # the layer waits for the operator
    approved = await sim.decide_tasks([tasks[0].task_id], approve=True)
    rejected = await sim.decide_tasks([tasks[1].task_id], approve=False)
    assert [t.status for t in approved] == ["APPROVED"] and [t.status for t in rejected] == ["REJECTED"]
    assert [t.task_id for t in (await sim.layer_feed()).tasks] == [tasks[0].task_id]
    # an operator placement needs no further approval
    await sim.queue_placement(ObserverPlacement(position=a))
    assert (await sim.snapshot()).deployment.tasks[-1].status == "APPROVED"
    # unanswered proposals expire; switching to automatic approval approves open proposals
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[b], reason="plan"))
    sim.tick = config.layer.proposal_timeout_s + 5
    assert (await sim.snapshot()).deployment.tasks[-1].status == "EXPIRED"
    await sim.queue_deployment(DeploymentRequest(tick=sim.tick, positions=[b], reason="plan"))
    auto = config.model_copy(deep=True)
    auto.layer.approval = "auto"
    await sim.set_config(auto)
    assert (await sim.snapshot()).deployment.tasks[-1].status == "APPROVED"


@pytest.mark.asyncio
async def test_drop_point_drifts_with_the_estimated_current() -> None:
    from aqua_drift.models import CurrentEstimate, Velocity

    sim = SimulationState(ScenarioConfig())
    sim.current_estimate = CurrentEstimate(
        base_velocity=Velocity(east_kt=2.0, north_kt=0.0), gradient_per_nm=[[0.0] * 3] * 3,
        reference_position=DATUM, observer_count=4, sample_count=10, window_seconds=60, residual_kt=0.0,
    )
    drop = _offset(DATUM, 30000.0, 0.0, 200.0)  # far: still in transit after 100 s
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[drop], reason="plan"))
    await _layer_steps(sim, 100, random.Random(2))
    task = (await sim.snapshot()).deployment.tasks[-1]
    east, north, _ = local_offset_m(drop, task.position)
    assert task.status == "APPROVED"
    assert abs(east - 2.0 * 0.5144 * 100) < 5.0 and abs(north) < 1.0
