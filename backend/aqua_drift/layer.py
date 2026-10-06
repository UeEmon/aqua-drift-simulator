"""設標者 (layer) kinematics: the craft that lays additional observers.

* moves over the sea surface at a speed drawn for every leg from speed_kt +- speed_spread_kt
* turns with bank <= max_bank_deg: turn rate omega = g tan(bank) / V, turn radius V^2 / (g tan(bank))
* TRANSIT: flies to the drop point (pure pursuit with the turn-rate limit). When the point is
  inside the turning circle on the side it would turn to, it flies straight on first (otherwise
  it would circle the point forever) and comes back. The observer is laid when the layer is
  within capture_radius_yd of the point.
* ORBIT: without a task it circles the estimated target position clockwise (vector-field
  guidance onto a circle whose radius is at least 1.2 x the turn radius, so the bank limit
  holds on the circle).
"""
from __future__ import annotations

import math
import random

from aqua_drift.deployment import _offset
from aqua_drift.models import LayerConfig, LayerFeed, LayerState, LayerUpdate, Position
from aqua_drift.physics import local_offset_m

G = 9.80665
KNOT_TO_MPS = 0.5144444444444445
YD_TO_M = 0.9144


def turn_radius_m(speed_kt: float, bank_deg: float) -> float:
    v = speed_kt * KNOT_TO_MPS
    return v * v / (G * math.tan(math.radians(bank_deg)))


def orbit_radius_m(config: LayerConfig, speed_kt: float) -> float:
    return max(config.orbit_radius_yd * YD_TO_M, 1.2 * turn_radius_m(speed_kt, config.max_bank_deg))


def leg_speed_kt(config: LayerConfig, rng: random.Random) -> float:
    low = max(config.speed_kt - config.speed_spread_kt, 1.0)
    high = max(config.speed_kt + config.speed_spread_kt, low)
    return rng.uniform(low, high)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def initial_state(config: LayerConfig, datum: Position, tick: int, rng: random.Random) -> LayerState:
    speed = leg_speed_kt(config, rng)
    radius = orbit_radius_m(config, speed)
    start = _offset(datum, 0.0, radius, 0.0)  # north of the datum, heading east = clockwise orbit
    return LayerState(
        tick=tick, position=start, heading_deg=90.0, speed_kt=speed, mode="ORBIT",
        orbit_center=datum, orbit_radius_yd=radius / YD_TO_M,
    )


def step(
    state: LayerState,
    config: LayerConfig,
    target: Position | None,
    task_id: int | None,
    datum: Position,
    rng: random.Random,
    dt: float = 1.0,
) -> tuple[LayerState, bool]:
    """Advance the layer by dt seconds. Returns (new state, arrived at the drop point)."""
    mode = "TRANSIT" if target is not None else "ORBIT"
    speed = state.speed_kt
    if mode != state.mode or task_id != state.task_id:
        speed = leg_speed_kt(config, rng)  # a new leg: new speed within +- spread
    v = speed * KNOT_TO_MPS
    omega_max = G * math.tan(math.radians(config.max_bank_deg)) / v
    radius_turn = v / omega_max
    heading = math.radians(state.heading_deg)
    position = state.position
    arrived = False
    eta = None
    orbit_radius = orbit_radius_m(config, speed)

    if mode == "TRANSIT":
        east, north, _ = local_offset_m(position, target)
        distance = math.hypot(east, north)
        if distance <= config.capture_radius_yd * YD_TO_M:
            arrived = True
            desired = heading
        else:
            desired = math.atan2(east, north)
        error = _wrap(desired - heading)
        turn = max(-omega_max * dt, min(omega_max * dt, error))
        if not arrived and abs(error) > 1e-6:
            side = 1.0 if error > 0 else -1.0  # +1: turn right (clockwise)
            # centre of the turning circle on that side (right = (cos h, -sin h))
            cx = side * radius_turn * math.cos(heading)
            cy = -side * radius_turn * math.sin(heading)
            if math.hypot(east - cx, north - cy) < radius_turn * 0.98:
                turn = 0.0  # point inside the turning circle: fly on, come back later
        eta = distance / v + abs(error) / omega_max
    else:
        # vector field onto a clockwise circle around the datum
        east, north, _ = local_offset_m(datum, position)
        distance = math.hypot(east, north)
        bearing = math.atan2(east, north)
        desired = bearing + math.pi / 2.0 + math.atan(2.0 * (distance - orbit_radius) / orbit_radius)
        error = _wrap(desired - heading)
        turn = max(-omega_max * dt, min(omega_max * dt, error))

    new_heading = heading + turn
    mid = heading + 0.5 * turn
    moved = _offset(position, v * dt * math.sin(mid), v * dt * math.cos(mid), 0.0)
    rate = turn / dt
    bank = math.degrees(math.atan(v * rate / G))
    return LayerState(
        tick=state.tick + int(dt),
        position=moved,
        heading_deg=math.degrees(new_heading) % 360.0,
        speed_kt=speed,
        bank_deg=bank,
        mode=mode,
        task_id=task_id if mode == "TRANSIT" else None,
        orbit_center=datum if mode == "ORBIT" else None,
        orbit_radius_yd=orbit_radius / YD_TO_M if mode == "ORBIT" else 0.0,
        eta_s=eta,
    ), arrived


def drift(position: Position, east_kt: float, north_kt: float, dt: float = 1.0) -> Position:
    """A drop point planned in the water frame moves with the (estimated) current."""
    return _offset(position, east_kt * KNOT_TO_MPS * dt, north_kt * KNOT_TO_MPS * dt, position.depth_ft)


def advance(feed: LayerFeed, state: LayerState, rng: random.Random) -> tuple[LayerState, LayerUpdate]:
    """Advance the layer from state.tick to feed.tick (1 s steps): drop points drift with the
    estimated current, the first approved task is flown to, reached points are completed."""
    positions = {t.task_id: t.position for t in feed.tasks}
    completed: dict[int, Position] = {}
    for _ in range(max(min(feed.tick - state.tick, 30), 0)):
        positions = {k: drift(p, feed.current_east_kt, feed.current_north_kt) for k, p in positions.items()}
        open_tasks = [t for t in feed.tasks if t.task_id not in completed]
        task = open_tasks[0] if open_tasks else None
        state, arrived = step(
            state, feed.config, positions[task.task_id] if task else None,
            task.task_id if task else None, feed.datum, rng,
        )
        if arrived and task is not None:
            completed[task.task_id] = positions[task.task_id]
    state.tick = max(state.tick, feed.tick)
    eta: dict[int, float] = {}
    previous, elapsed = state.position, 0.0
    for task in (t for t in feed.tasks if t.task_id not in completed):
        point = positions[task.task_id]
        if state.task_id == task.task_id and state.eta_s is not None:
            elapsed = state.eta_s  # current leg (includes the turn)
        else:  # following legs: straight at the nominal speed
            east, north, _ = local_offset_m(previous, point)
            elapsed += math.hypot(east, north) / (feed.config.speed_kt * KNOT_TO_MPS)
        eta[task.task_id] = round(elapsed, 1)
        previous = point
    return state, LayerUpdate(
        state=state,
        task_positions={k: v for k, v in positions.items() if k not in completed},
        task_eta_s=eta,
        completed=completed,
    )
