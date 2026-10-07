import math
import random

import pytest

from aqua_drift.deployment import _offset
from aqua_drift.layer import (
    advance,
    initial_state,
    orbit_radius_m,
    planned_path,
    step,
    turn_radius_m,
)
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



def test_planned_path_follows_the_turn_through_the_drop_points() -> None:
    config = LayerConfig()
    rng = random.Random(5)
    start = initial_state(config, DATUM, 0, rng)
    behind = _offset(start.position, -2000.0, -9000.0, 300.0)  # behind: the path turns first
    further = _offset(behind, 8000.0, -3000.0, 300.0)
    path = planned_path(start, config, [behind, further])
    assert planned_path(start, config, []) == []
    assert path[0] == (round(start.position.latitude, 5), round(start.position.longitude, 5))
    assert path[-1] == (round(further.latitude, 5), round(further.longitude, 5))
    assert (round(behind.latitude, 5), round(behind.longitude, 5)) in path
    # a curved path, not straight legs: it starts along the current heading and turns
    first = local_offset_m(start.position, Position(latitude=path[1][0], longitude=path[1][1], depth_ft=0.0))
    heading = math.radians(start.heading_deg)
    assert first[0] * math.sin(heading) + first[1] * math.cos(heading) > 0
    assert len(path) > 10

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


async def _drop_time(planned_in: int, east: float, north: float, seed: int = 4) -> tuple[int, int, list[str]]:
    sim = SimulationState(ScenarioConfig())
    rng = random.Random(seed)
    state = await _layer_steps(sim, 60, rng)  # settle on the orbit
    point = _offset(state.position, east, north, 500.0)
    request = DeploymentRequest(tick=sim.tick, positions=[point], reason="plan", planned_ticks=[sim.tick + planned_in])
    await sim.queue_deployment(request)
    planned = sim.tick + planned_in
    modes = []
    for _ in range(planned_in + 600):
        state = await _layer_steps(sim, 1, rng, state)
        modes.append(state.mode)
        task = (await sim.snapshot()).deployment.tasks[-1]
        if task.status == "DONE":
            return task.done_tick, planned, modes
    return -1, planned, modes


@pytest.mark.asyncio
async def test_layer_lays_the_observer_at_the_planned_time() -> None:
    # far point, plenty of time: keeps circling the target first, leaves late, arrives on time
    done, planned, modes = await _drop_time(500, 15000.0, -5000.0)
    assert abs(done - planned) <= 5
    assert modes[:100].count("ORBIT") > 90  # did not leave at once
    # close point: gets there early and holds over the point until the planned time
    done, planned, modes = await _drop_time(400, 1500.0, 1500.0)
    assert abs(done - planned) <= 5
    assert "HOLD" in modes or modes[:150].count("ORBIT") > 100


def _timed_drop_error(seed: int) -> tuple[int | None, bool]:
    """One random timed drop as the planner sets it (planned 30..290 s ahead, never before the
    layer's earliest time) with a random current. Returns (drop - planned time in s, feasible:
    some speed of the band can get there by the planned time)."""
    import numpy as np

    from aqua_drift.layer import flight_time_s, speed_band
    from aqua_drift.models import DropTask, LayerFeed
    from aqua_drift.optimal_deployment import LayerAvailability

    r = random.Random(seed)
    config = LayerConfig()
    rng = random.Random(seed + 1000)
    state = initial_state(config, DATUM, 0, rng)
    state, _ = advance(LayerFeed(tick=60, config=config, tasks=[], datum=DATUM), state, rng)
    distance, bearing = r.uniform(1000.0, 25000.0), r.uniform(0.0, 2.0 * math.pi)
    point = _offset(DATUM, distance * math.sin(bearing), distance * math.cos(bearing), 500.0)
    availability = LayerAvailability(ready_s=0.0, position=state.position, speed_kt=config.speed_kt,
                                     max_bank_deg=config.max_bank_deg, heading_deg=state.heading_deg)
    east, north, _ = local_offset_m(DATUM, point)
    earliest = float(availability.earliest_s(DATUM, np.array([[east, north, 0.0]]))[0])
    planned = 60 + max(math.ceil(earliest), r.randint(30, 290))
    current = (r.uniform(-2.0, 2.0), r.uniform(-2.0, 2.0))
    quickest = min(flight_time_s(state, config, point, sp, dt=1.0) for sp in speed_band(config))
    task = DropTask(task_id=1, created_tick=60, source="forward", position=point, status="APPROVED",
                    planned_tick=planned)
    for tick in range(61, planned + 600):
        feed = LayerFeed(tick=tick, config=config, tasks=[task], datum=DATUM,
                         current_east_kt=current[0], current_north_kt=current[1])
        state, update = advance(feed, state, rng)
        if 1 in update.completed:
            return tick - planned, planned - 60 >= quickest + 5
    return None, planned - 60 >= quickest + 5


def test_layer_drops_within_seconds_of_the_planned_time() -> None:
    """Random drops as the planner schedules them: the layer is on time, adjusting the time
    with the flight path (docs/handoff.md reported +-24 s and up to +52 s; the coarse
    flight-time simulation, no correction on the way and the departure rule were the causes)."""
    errors = [e for e, feasible in map(_timed_drop_error, range(60)) if feasible]
    assert len(errors) >= 50 and None not in errors
    assert sum(abs(e) <= 5 for e in errors) >= 0.95 * len(errors)
    assert max(abs(e) for e in errors) <= 30


def test_timed_leg_takes_up_time_with_a_left_detour_at_constant_speed() -> None:
    """Time is adjusted with the flight path: an early layer makes a detour turning left (4.15)
    and keeps its speed."""
    from aqua_drift.layer import _arrival_s
    from aqua_drift.models import DropTask, LayerFeed

    config = LayerConfig()
    rng = random.Random(2)
    state = initial_state(config, DATUM, 0, rng).model_copy(
        update={"speed_kt": 190.0, "heading_deg": 0.0, "mode": "TRANSIT", "task_id": 1})
    point = _offset(state.position, 0.0, 12000.0, 500.0)
    planned = round(_arrival_s(state, config, point)) + 90  # 90 s early on the direct path
    task = DropTask(task_id=1, created_tick=0, source="forward", position=point, status="APPROVED",
                    planned_tick=planned)
    banks, speeds = [], []
    for tick in range(1, planned + 300):
        state, update = advance(LayerFeed(tick=tick, config=config, tasks=[task], datum=DATUM), state, rng)
        banks.append(state.bank_deg)
        speeds.append(state.speed_kt)
        if 1 in update.completed:
            break
    assert abs(tick - planned) <= 5
    assert set(speeds) == {190.0}
    assert sum(b < -10.0 for b in banks) > 20  # the detour is a left turn


def test_timed_leg_adjusts_the_speed_on_the_way() -> None:
    from aqua_drift.layer import SPEED_RATE_KT_S

    config = LayerConfig()
    state = initial_state(config, DATUM, 0, random.Random(1)).model_copy(update={"speed_kt": 200.0})
    target = _offset(DATUM, 20000.0, 0.0, 0.0)
    state, _ = step(state, config, target, 1, DATUM, random.Random(1), wanted_speed_kt=200.0)
    state, _ = step(state, config, target, 1, DATUM, random.Random(1), wanted_speed_kt=150.0)
    assert state.speed_kt == pytest.approx(200.0 - SPEED_RATE_KT_S)  # rate-limited
    for _ in range(20):
        state, _ = step(state, config, target, 1, DATUM, random.Random(1), wanted_speed_kt=150.0)
    assert state.speed_kt == pytest.approx(150.0)


def test_planner_schedules_drops_before_detection_and_after_the_layer_can_be_there() -> None:
    from aqua_drift.forward_deployment import plan_forward_deployment_scheduled
    from aqua_drift.models import (
        EstimateMode,
        ForwardDeploymentConfig,
        PresenceRegion,
        TrackEstimate,
        Uncertainty,
    )
    from aqua_drift.optimal_deployment import LayerAvailability

    origin = Position(latitude=35.0, longitude=140.0, depth_ft=400.0)
    estimate = TrackEstimate(
        mode=EstimateMode.ONLINE, tick=100, observability_status="TRACKING", current_position=origin,
        depth_ft=400.0, hdg_deg=90.0, through_water_speed_kt=8.0,
        uncertainty=Uncertainty(
            horizontal_major_yd=50, horizontal_minor_yd=20, horizontal_major_axis_deg=0, depth_sigma_ft=100,
            ground_speed_sigma_kt=0.2, through_water_speed_sigma_kt=0.2, cog_sigma_deg=1, hdg_sigma_deg=1,
            bias_sigma_hz=0.01,
        ),
        presence_region=PresenceRegion(probability_pct=90),
    )
    behind = [_offset(origin, -2700.0, 1800.0, 200.0), _offset(origin, -2700.0, -1800.0, 200.0)]
    layer = LayerAvailability(ready_s=0.0, position=_offset(origin, 0.0, -20000.0, 0.0), speed_kt=200.0,
                              max_bank_deg=15.0, heading_deg=0.0)
    config = ForwardDeploymentConfig()
    positions, reason, planned = plan_forward_deployment_scheduled(
        100, estimate, behind, [], config, 6000, None, layer=layer)
    assert positions and planned and len(planned) == len(positions)
    assert "drop in" in reason
    earliest = layer.earliest_s(origin, __import__("numpy").array([
        [*local_offset_m(origin, p)[0:2], 0.0] for p in positions]))
    for tick, first in zip(planned, earliest, strict=True):
        assert tick >= 100 + first - 1  # not before the layer can be there
        assert tick <= 100 + config.horizon_s
    # without scheduling the drops are as early as the layer allows
    asap = ForwardDeploymentConfig(schedule_drops=False)
    _, _, early = plan_forward_deployment_scheduled(100, estimate, behind, [], asap, 6000, None, layer=layer)
    assert min(early) <= min(planned)


def test_left_turn_is_the_standard_and_right_only_when_clearly_shorter() -> None:
    from aqua_drift.layer import dubins_turn

    r = 4000.0
    # point dead astern: both sides equal -> left (standard)
    assert dubins_turn(0.0, -20000.0, 0.0, r)[0] == -1
    # point slightly to the right: right is shorter, but by less than the margin -> still left
    side_small, _, _ = dubins_turn(300.0, -20000.0, 0.0, r, margin_m=10 * 200 * 0.5144)
    assert side_small == -1
    # point well to the right and ahead: right turn is far shorter -> right
    assert dubins_turn(9000.0, 3000.0, 0.0, r, margin_m=10 * 200 * 0.5144)[0] == 1
    # a right-turn standard mirrors it
    assert dubins_turn(0.0, -20000.0, 0.0, r, prefer=1)[0] == 1
    assert dubins_turn(-9000.0, 3000.0, 0.0, r, prefer=1, margin_m=1000.0)[0] == -1


def test_orbit_is_flown_counter_clockwise_with_left_bank_by_default() -> None:
    rng = random.Random(8)
    for turn, sign in (("left", -1), ("right", 1)):
        config = LayerConfig(preferred_turn=turn)
        state = initial_state(config, DATUM, 0, rng)
        banks = []
        for _ in range(900):
            state, _ = step(state, config, None, None, DATUM, rng)
            banks.append(state.bank_deg)
        settled = banks[-300:]
        assert all(sign * b > 0 for b in settled), turn  # steady turn in the preferred sense
        assert max(abs(b) for b in banks) <= 15.0 + 1e-6


@pytest.mark.asyncio
async def test_paused_layer_keeps_orbiting_and_operator_can_cancel_or_reschedule() -> None:
    config = ScenarioConfig()
    config.layer.paused = True
    sim = SimulationState(config)
    a = _offset(DATUM, 5000.0, 0.0, 500.0)
    b = _offset(DATUM, 6000.0, 2000.0, 60.0)
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[a, b], reason="plan", planned_ticks=[300, 400]))
    rng = random.Random(3)
    state = await _layer_steps(sim, 120, rng)
    status = (await sim.snapshot()).deployment
    assert state.mode == "ORBIT" and status.layer.task_id is None  # paused: no departure
    first, second = status.tasks[-2], status.tasks[-1]
    moved = await sim.reschedule_task(first.task_id, None)
    assert moved is not None and moved.planned_tick is None
    assert await sim.reschedule_task(999, 10) is None
    cancelled = await sim.cancel_tasks([second.task_id])
    assert [t.status for t in cancelled] == ["CANCELLED"]
    assert [t.task_id for t in (await sim.layer_feed()).tasks] == [first.task_id]
    # resume: the layer leaves for the remaining drop at once
    resumed = config.model_copy(deep=True)
    resumed.layer.paused = False
    await sim.set_config(resumed)
    state = await _layer_steps(sim, 10, rng, state)
    assert state.mode == "TRANSIT" and state.task_id == first.task_id
    assert [t.status for t in await sim.cancel_tasks(None)] == ["CANCELLED"]
