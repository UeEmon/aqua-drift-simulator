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
* Scheduling: every drop has a planned time (the optimal drop time from the planner). The
  layer leaves the orbit when it must to arrive on time, circles the drop point (HOLD) if it is
  early and lays the observer at the planned time.
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


def eta_to(state: LayerState, config: LayerConfig, point: Position) -> float:
    """Time to reach a point from the current state: straight distance at the current speed
    plus the turn towards it at the bank limit (and a half circle when the point is close
    behind, inside the turning circle)."""
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    omega = G * math.tan(math.radians(config.max_bank_deg)) / v
    east, north, _ = local_offset_m(state.position, point)
    distance = math.hypot(east, north)
    error = abs(_wrap(math.atan2(east, north) - math.radians(state.heading_deg)))
    extra = 0.0
    if distance < 2.0 * v / omega and error > math.pi / 2:
        extra = math.pi * (v / omega) / v  # loop around first
    return distance / v + error / omega + extra


def speed_band(config: LayerConfig, count: int = 5) -> list[float]:
    low = max(config.speed_kt - config.speed_spread_kt, 1.0)
    high = config.speed_kt + config.speed_spread_kt
    return [low + (high - low) * k / (count - 1) for k in range(count)] if high > low else [low]


def dubins_turn(east: float, north: float, heading: float, radius: float) -> tuple[int, float, float]:
    """Quickest turn-then-straight path (Dubins CS) to a point at (east, north) metres from the
    craft heading `heading` (rad, clockwise from north) with turn radius `radius`.
    Returns (side: +1 right / -1 left / 0 none possible, turn angle rad, straight length m)."""
    best = (0, 0.0, 0.0)
    best_length = math.inf
    for side in (1, -1):
        # turning-circle centre: right = (cos h, -sin h), left = opposite
        cx, cy = side * radius * math.cos(heading), -side * radius * math.sin(heading)
        dx, dy = east - cx, north - cy
        d = math.hypot(dx, dy)
        if d < radius:
            continue  # the point is inside this turning circle
        straight = math.sqrt(max(d * d - radius * radius, 0.0))
        phi_p = math.atan2(dx, dy)  # bearing of the point from the centre
        theta0 = math.atan2(-cx, -cy)  # bearing of the craft from the centre
        tangent = math.atan2(straight, radius)
        if side == 1:  # clockwise: leave the circle at theta = phi_p - tangent
            arc = (phi_p - tangent - theta0) % (2 * math.pi)
        else:  # counter-clockwise
            arc = (theta0 - (phi_p + tangent)) % (2 * math.pi)
        length = arc * radius + straight
        if length < best_length:
            best_length, best = length, (side, arc, straight)
    return best


def flight_time_s(state: LayerState, config: LayerConfig, point: Position, speed_kt: float,
                  limit_s: float = 1200.0, dt: float = 5.0) -> float:
    """Flight time to a point by simulating the same guidance (coarse 5 s steps) at a fixed
    speed: accounts for the turn limit, loops and the fly-on-then-return manoeuvre."""
    probe = state.model_copy(update={"speed_kt": speed_kt, "mode": "TRANSIT", "task_id": -1})
    elapsed = 0.0
    rng = random.Random(0)
    while elapsed < limit_s:
        probe, arrived = step(probe, config, point, -1, point, rng, dt=dt)
        elapsed += dt
        if arrived:
            return elapsed
    return limit_s


def step(
    state: LayerState,
    config: LayerConfig,
    target: Position | None,
    task_id: int | None,
    datum: Position,
    rng: random.Random,
    dt: float = 1.0,
    hold_center: Position | None = None,
    wanted_speed_kt: float | None = None,
) -> tuple[LayerState, bool]:
    """Advance the layer by dt seconds. Returns (new state, arrived at the drop point).

    target given -> TRANSIT to it; hold_center given -> HOLD (circle the drop point, early for
    its planned time); otherwise ORBIT around the datum (estimated target position)."""
    mode = "TRANSIT" if target is not None else ("HOLD" if hold_center is not None else "ORBIT")
    speed = state.speed_kt
    if mode != state.mode or task_id != state.task_id:
        if wanted_speed_kt is not None:  # timed leg: the speed (within the band) that arrives on time
            low = max(config.speed_kt - config.speed_spread_kt, 1.0)
            speed = min(max(wanted_speed_kt, low), config.speed_kt + config.speed_spread_kt)
        else:
            speed = leg_speed_kt(config, rng)  # a new leg: new speed within +- spread
    v = speed * KNOT_TO_MPS
    omega_max = G * math.tan(math.radians(config.max_bank_deg)) / v
    radius_turn = v / omega_max
    heading = math.radians(state.heading_deg)
    position = state.position
    arrived = False
    eta = None
    center = hold_center if mode == "HOLD" else datum
    orbit_radius = HOLD_RADIUS_TURNS * radius_turn if mode == "HOLD" else orbit_radius_m(config, speed)

    if mode == "TRANSIT":
        east, north, _ = local_offset_m(position, target)
        distance = math.hypot(east, north)
        if distance <= config.capture_radius_yd * YD_TO_M:
            arrived = True
            turn = 0.0
            eta = 0.0
        else:
            side, arc, straight = dubins_turn(east, north, heading, radius_turn)
            if side == 0:
                turn = 0.0  # inside both turning circles: fly on first
                eta = distance / v + math.pi / omega_max
            else:
                turn = side * min(omega_max * dt, arc)
                eta = (arc * radius_turn + straight) / v
    else:
        # vector field onto a clockwise circle around the centre
        east, north, _ = local_offset_m(center, position)
        distance = math.hypot(east, north)
        bearing = math.atan2(east, north)
        desired = bearing + math.pi / 2.0 + math.atan(2.0 * (distance - orbit_radius) / orbit_radius)
        error = _wrap(desired - heading)
        turn = max(-omega_max * dt, min(omega_max * dt, error))

    new_heading = heading + turn
    mid = heading + 0.5 * turn
    moved = _offset(position, v * dt * math.sin(mid), v * dt * math.cos(mid), 0.0)
    if mode == "TRANSIT" and not arrived:
        # closest approach within this step (steps are ~100 m long at 200 kt)
        px, py, _ = local_offset_m(position, target)
        sx, sy = v * dt * math.sin(mid), v * dt * math.cos(mid)
        along = max(0.0, min(1.0, (px * sx + py * sy) / max(sx * sx + sy * sy, 1e-9)))
        if math.hypot(px - along * sx, py - along * sy) <= config.capture_radius_yd * YD_TO_M:
            arrived = True
    rate = turn / dt
    bank = math.degrees(math.atan(v * rate / G))
    return LayerState(
        tick=state.tick + int(dt),
        position=moved,
        heading_deg=math.degrees(new_heading) % 360.0,
        speed_kt=speed,
        bank_deg=bank,
        mode=mode,
        task_id=task_id if mode in ("TRANSIT", "HOLD") else None,
        orbit_center=center if mode != "TRANSIT" else None,
        orbit_radius_yd=orbit_radius / YD_TO_M if mode != "TRANSIT" else 0.0,
        eta_s=eta,
    ), arrived


def drift(position: Position, east_kt: float, north_kt: float, dt: float = 1.0) -> Position:
    """A drop point planned in the water frame moves with the (estimated) current."""
    return _offset(position, east_kt * KNOT_TO_MPS * dt, north_kt * KNOT_TO_MPS * dt, position.depth_ft)


DEPART_MARGIN_S = 10.0  # leave this much earlier than the flight time at the slowest speed
ON_TIME_TOLERANCE_S = 15.0  # a drop up to this early counts as on time
HOLD_RADIUS_TURNS = 2.5  # holding circle around an early drop point, in turn radii


def hold_inbound_s(speed_kt: float, config: LayerConfig) -> float:
    """From the holding circle (tangent heading, point abeam) to the point: a quarter turn
    towards it and the rest straight."""
    v = speed_kt * KNOT_TO_MPS
    radius = v * v / (G * math.tan(math.radians(config.max_bank_deg)))
    return (0.5 * math.pi * radius + (HOLD_RADIUS_TURNS - 1.0) * radius) / v


def loop_s(speed_kt: float, config: LayerConfig) -> float:
    v = speed_kt * KNOT_TO_MPS
    return 2.0 * math.pi * v / (G * math.tan(math.radians(config.max_bank_deg)))


def advance(feed: LayerFeed, state: LayerState, rng: random.Random) -> tuple[LayerState, LayerUpdate]:
    """Advance the layer from state.tick to feed.tick (1 s steps).

    Tasks are flown in the order of their planned drop time. The layer keeps circling the
    estimated target until it must leave to arrive on time (planned time - flight time -
    margin); if it gets to the point early it circles the point (HOLD) and comes in so as to
    drop at the planned time. Tasks without a planned time are flown at once. Drop points
    drift with the estimated current."""
    tasks = sorted(feed.tasks, key=lambda t: (t.planned_tick if t.planned_tick is not None else -1, t.task_id))
    positions = {t.task_id: t.position for t in tasks}
    completed: dict[int, Position] = {}
    tick = state.tick
    for _ in range(max(min(feed.tick - state.tick, 30), 0)):
        tick += 1
        positions = {k: drift(p, feed.current_east_kt, feed.current_north_kt) for k, p in positions.items()}
        open_tasks = [t for t in tasks if t.task_id not in completed]
        task = open_tasks[0] if open_tasks else None
        target = hold = wanted = None
        task_id = None
        if task is not None:
            point = positions[task.task_id]
            planned = task.planned_tick if task.planned_tick is not None else tick
            # leave as late as possible: when no speed of the band (150..250 kt) could still arrive
            # later than the planned time; the leg speed is then the one whose (simulated) flight
            # time is closest to the time left
            committed = state.task_id == task.task_id and state.mode == "TRANSIT"
            holding = state.mode == "HOLD" and state.task_id == task.task_id
            left = planned - tick
            times: dict[float, float] = {}
            if not committed and not holding and left > 0:
                fastest = feed.config.speed_kt + feed.config.speed_spread_kt
                if eta_to(state.model_copy(update={"speed_kt": fastest}), feed.config, point) <= left + 600:
                    times = {sp: flight_time_s(state, feed.config, point, sp) for sp in speed_band(feed.config)}
                    # prefer speeds that fly in directly (no extra loop): loops at a larger speed
                    # are not a reliable way to spend time (the holding pattern is)
                    east, north, _ = local_offset_m(state.position, point)
                    direct = {
                        sp: t for sp, t in times.items()
                        if t <= math.hypot(east, north) / (sp * KNOT_TO_MPS) + loop_s(sp, feed.config) / 2 + 10
                    }
                    times = direct or times
            latest = max(times.values()) if times else -math.inf
            if holding:
                # circle the drop point until the (simulated) flight back in takes the time left
                if left > flight_time_s(state, feed.config, point, state.speed_kt) + 5.0:
                    hold, task_id = point, task.task_id
                else:
                    target, task_id = point, task.task_id  # turn in to drop on time
            elif committed or left <= 0 or left - DEPART_MARGIN_S <= latest:
                target, task_id = point, task.task_id
                if times:
                    wanted = min(times, key=lambda sp: abs(times[sp] - left))
        state, arrived = step(state, feed.config, target, task_id, feed.datum, rng, hold_center=hold,
                              wanted_speed_kt=wanted)
        if arrived and task is not None:
            planned = task.planned_tick if task.planned_tick is not None else tick
            early = planned - tick
            # drop now when on time; otherwise come round again (HOLD) if that ends closer to the
            # planned time than dropping early now (coming round takes at least one flight back)
            back = loop_s(state.speed_kt, feed.config)  # once past the point: at least one loop
            if early <= ON_TIME_TOLERANCE_S or max(back - early, 0.0) >= early:
                completed[task.task_id] = positions[task.task_id]
            else:
                state = state.model_copy(update={"mode": "HOLD", "task_id": task.task_id})
    state.tick = max(state.tick, feed.tick)
    eta: dict[int, float] = {}
    previous, elapsed = state.position, 0.0
    for task in (t for t in tasks if t.task_id not in completed):
        point = positions[task.task_id]
        if state.task_id == task.task_id and state.eta_s is not None:
            elapsed = state.eta_s
        else:
            east, north, _ = local_offset_m(previous, point)
            elapsed += math.hypot(east, north) / (feed.config.speed_kt * KNOT_TO_MPS)
        planned = task.planned_tick
        # expected drop time from now: the planned time when it can be met, else the arrival
        eta[task.task_id] = round(max(elapsed, (planned - feed.tick) if planned is not None else 0.0), 1)
        elapsed = eta[task.task_id]
        previous = point
    return state, LayerUpdate(
        state=state,
        task_positions={k: v for k, v in positions.items() if k not in completed},
        task_eta_s=eta,
        completed=completed,
    )
