"""設標者 (layer) kinematics: the craft that lays additional observers.

* moves over the sea surface at a speed drawn for every leg from speed_kt +- speed_spread_kt
* turns with bank <= max_bank_deg: turn rate omega = g tan(bank) / V, turn radius V^2 / (g tan(bank))
* TRANSIT: flies to the drop point (pure pursuit with the turn-rate limit). When the point is
  inside the turning circle on the side it would turn to, it flies straight on first (otherwise
  it would circle the point forever) and comes back. The observer is laid when the layer is
  within capture_radius_yd of the point.
* Turns: left turn is the standard (circling counter-clockwise); a right turn is taken only when
  the route is clearly more efficient that way (shorter by more than turn_margin_s of flight).
* ORBIT: without a task it circles the estimated target position (vector-field
  guidance onto a circle whose radius is at least 1.2 x the turn radius, so the bank limit
  holds on the circle).
* Scheduling: every drop has a planned time (the optimal drop time from the planner). Time is
  adjusted with the flight path, not the speed: the layer leaves the orbit shortly before the
  direct flight time (along the same turn-then-straight path the guidance flies) and takes up
  the rest on the way by a detour with left turns, re-planned every second. The speed is
  changed only when no path can arrive on time (e.g. the point is inside the turning circle).
* 3-D flight in the wind (feed.wind): the layer cruises at cruise_altitude_ft, descends to
  drop_altitude_ft on the way to a drop point (climb_rate_fpm) and flies through the air mass:
  its ground velocity is the air velocity plus the wind at its altitude. The guidance runs in
  the air-mass frame (a ground point moves there with current - wind), aiming at the
  interception point. The observer is released at the release point (the drop point less the
  predicted fall displacement: the throw of the layer's ground speed and the drift of the
  estimated mean wind), the exact release instant within the 1 s step is where the path passes
  closest to it, and it falls freely through the true wind profile to the sea surface
  (aqua_drift.wind).
"""
from __future__ import annotations

import math
import random

import numpy as np

from aqua_drift import wind as windlib
from aqua_drift.deployment import _offset
from aqua_drift.models import DropRelease, LayerConfig, LayerFeed, LayerState, LayerUpdate, Position
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
    start = _offset(datum, 0.0, radius, 0.0)  # north of the datum, tangent to the orbit
    heading = 270.0 if preferred_side(config) < 0 else 90.0  # left turn: counter-clockwise
    return LayerState(
        tick=tick, position=start, heading_deg=heading, speed_kt=speed, mode="ORBIT",
        orbit_center=datum, orbit_radius_yd=radius / YD_TO_M, altitude_ft=config.cruise_altitude_ft,
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


def dubins_turn(east: float, north: float, heading: float, radius: float,
                prefer: int = -1, margin_m: float = 0.0) -> tuple[int, float, float]:
    """Turn-then-straight path (Dubins CS) to a point at (east, north) metres from the craft
    heading `heading` (rad, clockwise from north) with turn radius `radius`.

    The preferred side (default -1 = left turn) is taken unless the other side is shorter by
    more than margin_m (the route is clearly more efficient that way).
    Returns (side: +1 right / -1 left / 0 none possible, turn angle rad, straight length m)."""
    paths: dict[int, tuple[float, float, float]] = {}
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
        paths[side] = (arc * radius + straight, arc, straight)
    if not paths:
        return 0, 0.0, 0.0
    other = -prefer
    if prefer in paths and (other not in paths or paths[other][0] >= paths[prefer][0] - margin_m):
        side = prefer
    else:
        side = other if other in paths else prefer
    return side, paths[side][1], paths[side][2]


def preferred_side(config: LayerConfig) -> int:
    """-1: left turn (the standard), +1: right turn."""
    return 1 if config.preferred_turn == "right" else -1


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
    turn_rate: float | None = None,
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
    elif wanted_speed_kt is not None and mode == "TRANSIT":
        # timed leg: the speed is adjusted on the way (rate-limited) to arrive at the planned time
        low = max(config.speed_kt - config.speed_spread_kt, 1.0)
        wanted = min(max(wanted_speed_kt, low), config.speed_kt + config.speed_spread_kt)
        speed += max(-SPEED_RATE_KT_S * dt, min(SPEED_RATE_KT_S * dt, wanted - speed))
    v = speed * KNOT_TO_MPS
    omega_max = G * math.tan(math.radians(config.max_bank_deg)) / v
    radius_turn = v / omega_max
    heading = math.radians(state.heading_deg)
    position = state.position
    arrived = False
    eta = None
    center = hold_center if mode == "HOLD" else datum
    if mode == "HOLD":
        # circle the drop point at the distance the layer is at when it starts holding (time is
        # spent without going further away), at least HOLD_RADIUS_TURNS turn radii so that the
        # turn in is a quarter turn
        if state.mode == "HOLD" and state.task_id == task_id and state.orbit_radius_yd > 0:
            orbit_radius = state.orbit_radius_yd * YD_TO_M
        else:
            east, north, _ = local_offset_m(hold_center, position)
            orbit_radius = max(HOLD_RADIUS_TURNS * radius_turn, math.hypot(east, north))
    else:
        orbit_radius = orbit_radius_m(config, speed)

    if mode == "TRANSIT":
        east, north, _ = local_offset_m(position, target)
        distance = math.hypot(east, north)
        if distance <= config.capture_radius_yd * YD_TO_M:
            arrived = True
            turn = 0.0
            eta = 0.0
        else:
            side, arc, straight = dubins_turn(
                east, north, heading, radius_turn, preferred_side(config), config.turn_margin_s * v
            )
            if side == 0:
                turn = 0.0  # inside both turning circles: fly on first
                eta = distance / v + math.pi / omega_max
            else:
                turn = side * min(omega_max * dt, arc)
                eta = (arc * radius_turn + straight) / v
            if turn_rate is not None:  # timed leg: the turn chosen to arrive at the planned time
                turn = max(-omega_max, min(omega_max, turn_rate)) * dt
    else:
        # vector field onto a clockwise circle around the centre
        east, north, _ = local_offset_m(center, position)
        distance = math.hypot(east, north)
        bearing = math.atan2(east, north)
        # circling in the preferred direction (left turn = counter-clockwise by default)
        sense = float(preferred_side(config))  # -1 counter-clockwise, +1 clockwise
        desired = bearing + sense * (math.pi / 2.0 + math.atan(2.0 * (distance - orbit_radius) / orbit_radius))
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
        altitude_ft=state.altitude_ft,
    ), arrived


def drift(position: Position, east_kt: float, north_kt: float, dt: float = 1.0) -> Position:
    """A drop point planned in the water frame moves with the (estimated) current."""
    return _offset(position, east_kt * KNOT_TO_MPS * dt, north_kt * KNOT_TO_MPS * dt, position.depth_ft)


ON_TIME_TOLERANCE_S = 15.0  # a drop up to this early counts as on time
HOLD_RADIUS_TURNS = 2.5  # smallest holding circle around an early drop point, in turn radii
SPEED_RATE_KT_S = 5.0  # speed change on a timed leg, kt per second
LOOKAHEAD_S = 90  # how far ahead the layer checks that waiting on its circle keeps it on time
TURN_STEPS = 4  # detour turns compared on a timed leg: k/TURN_STEPS of the bank limit, k = 0..TURN_STEPS
HORIZON_S = 240  # longest detour turn compared
DEPART_SLACK_S = 10.0  # leave this much before the direct flight time (taken up by the path)
PATH_TOLERANCE_S = 5.0  # the speed is changed only when no path arrives this close to the planned time


def loop_s(speed_kt: float, config: LayerConfig) -> float:
    v = speed_kt * KNOT_TO_MPS
    return 2.0 * math.pi * v / (G * math.tan(math.radians(config.max_bank_deg)))


def _arrival_s(state: LayerState, config: LayerConfig, point: Position) -> float:
    """Flight time at the current speed along the guidance path (cheap when the point is
    inside both turning circles: coarse simulation)."""
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    radius = v * v / (G * math.tan(math.radians(config.max_bank_deg)))
    east, north, _ = local_offset_m(state.position, point)
    side, arc, straight = dubins_turn(
        east, north, math.radians(state.heading_deg), radius, preferred_side(config), config.turn_margin_s * v
    )
    if side == 0:
        return flight_time_s(state, config, point, state.speed_kt, limit_s=900.0)
    return max(arc * radius + straight - config.capture_radius_yd * YD_TO_M, 0.0) / v


def _dubins_times(east: np.ndarray, north: np.ndarray, heading: np.ndarray, v: float,
                  config: LayerConfig) -> np.ndarray:
    """dubins_turn for arrays of relative positions/headings: flight time (s) to within the
    capture radius at speed v (m/s); inf where the point is inside both turning circles."""
    radius = v * v / (G * math.tan(math.radians(config.max_bank_deg)))
    lengths = {}
    for side in (1, -1):
        cx, cy = side * radius * np.cos(heading), -side * radius * np.sin(heading)
        dx, dy = east - cx, north - cy
        d = np.hypot(dx, dy)
        straight = np.sqrt(np.maximum(d * d - radius * radius, 0.0))
        phi, theta0, tangent = np.arctan2(dx, dy), np.arctan2(-cx, -cy), np.arctan2(straight, radius)
        arc = np.mod(phi - tangent - theta0 if side == 1 else theta0 - phi - tangent, 2.0 * math.pi)
        lengths[side] = np.where(d >= radius, arc * radius + straight, np.inf)
    prefer = preferred_side(config)
    lp, lo = lengths[prefer], lengths[-prefer]
    length = np.where(np.isfinite(lp) & (lp <= lo + config.turn_margin_s * v), lp, lo)
    return np.maximum(length - config.capture_radius_yd * YD_TO_M, 0.0) / v


def _best_path(state: LayerState, config: LayerConfig, point: Position, left: float,
               speed_kt: float) -> tuple[float | None, float]:
    """Best path at speed_kt: the guidance path, or 'turn to the preferred side (left: 4.15) at
    k/TURN_STEPS of the bank limit, or fly straight, for T = 1..HORIZON_S s, then the guidance
    path'. The arrival closest to the planned time wins (ties, within 0.5 s: the shortest
    detour, then the gentlest turn). Returns (turn rate rad/s for this second, None = the
    guidance turn; arrival error s)."""
    v = max(speed_kt, 1.0) * KNOT_TO_MPS
    omega = G * math.tan(math.radians(config.max_bank_deg)) / v
    probe = state.model_copy(update={"speed_kt": speed_kt})
    direct = _arrival_s(probe, config, point) - left
    if direct >= -0.5:  # on time or late: nothing is quicker than the guidance path
        return None, direct
    east, north, _ = local_offset_m(state.position, point)
    h0 = math.radians(state.heading_deg)
    rates = preferred_side(config) * omega * np.arange(0, TURN_STEPS + 1) / TURN_STEPS
    t = np.arange(1, HORIZON_S + 1, dtype=float)
    rate, t = np.meshgrid(rates, t, indexing="ij")
    h = h0 + rate * t
    safe = np.where(rate == 0.0, 1.0, rate)
    dx = np.where(rate == 0.0, v * t * math.sin(h0), v / safe * (math.cos(h0) - np.cos(h)))
    dy = np.where(rate == 0.0, v * t * math.cos(h0), v / safe * (np.sin(h) - math.sin(h0)))
    error = t + _dubins_times(east - dx, north - dy, h, v, config) - left
    best = float(np.min(np.abs(error)))
    if best >= abs(direct) - 0.5:
        return None, direct
    close = np.abs(error) <= best + 0.5
    k, j = min(zip(*np.nonzero(close)), key=lambda kj: (t[kj], abs(rate[kj])))
    return float(rate[k, j]), float(error[k, j])


def timed_turn(state: LayerState, config: LayerConfig, point: Position,
               left: float) -> tuple[float | None, float]:
    """Path control of a timed leg, re-planned every second: the best path at the current
    speed (see _best_path). Only when no path arrives within PATH_TOLERANCE_S of the planned
    time is the speed changed: the smallest change of the band (5 kt steps) whose best path
    arrives within PATH_TOLERANCE_S, else the speed whose best path is closest. Returns (turn
    rate for this second or None = guidance, speed kt)."""
    rate, error = _best_path(state, config, point, left, state.speed_kt)
    if abs(error) <= PATH_TOLERANCE_S:
        return rate, state.speed_kt
    options = [(abs(e), abs(sp - state.speed_kt), r, sp)
               for sp in speed_band(config, 21) for r, e in [_best_path(state, config, point, left, sp)]]
    options.append((abs(error), 0.0, rate, state.speed_kt))
    within = [o for o in options if o[0] <= PATH_TOLERANCE_S]
    _, _, rate, speed = min(within, key=lambda o: o[1]) if within else min(options, key=lambda o: (round(o[0]), o[1]))
    return rate, speed


def departure(state: LayerState, config: LayerConfig, point: Position, left: float,
              datum: Position, hold: Position | None) -> str:
    """'leave' or 'wait' at the current speed: leaves DEPART_SLACK_S before the direct flight
    time (the path takes up the rest), or earlier when waiting on the circle (next LOOKAHEAD_S
    s) would make the direct path late (e.g. the point comes behind)."""
    if left <= _arrival_s(state, config, point) + DEPART_SLACK_S:
        return "leave"
    probe = state
    rng = random.Random(0)
    for tau in range(5, LOOKAHEAD_S + 1, 5):
        for _ in range(5):
            probe, _ = step(probe, config, None, probe.task_id, datum, rng, hold_center=hold,
                            wanted_speed_kt=state.speed_kt)
        if _arrival_s(probe, config, point) > left - tau - 2.0:
            return "leave"
    return "wait"


PATH_STEP_S = 5.0  # sampling of the planned flight path (about 500 m at 200 kt)
PATH_LEG_LIMIT_S = 1200.0  # longest leg drawn
PATH_MAX_POINTS = 400


def _shift(position: Position, east_m: float, north_m: float) -> Position:
    if not east_m and not north_m:
        return position
    return _offset(position, east_m, north_m, position.depth_ft)


def _intercept_s(east: float, north: float, drift_e: float, drift_n: float, v: float) -> float:
    """Time to meet a point at (east, north) m that moves at (drift_e, drift_n) m/s, flying
    straight at v m/s: |d + u t| = v t."""
    a = drift_e * drift_e + drift_n * drift_n - v * v
    b = 2.0 * (east * drift_e + north * drift_n)
    c = east * east + north * north
    if a >= -1e-9:
        return math.sqrt(c) / max(v, 1.0)
    return (b + math.sqrt(max(b * b - 4.0 * a * c, 0.0))) / (-2.0 * a)


def _lead(origin: Position, point: Position, drift_e: float, drift_n: float, v: float) -> Position:
    """Where a point moving at (drift_e, drift_n) m/s is met (air-mass frame interception)."""
    if not drift_e and not drift_n:
        return point
    east, north, _ = local_offset_m(origin, point)
    t = _intercept_s(east, north, drift_e, drift_n, v)
    return _shift(point, drift_e * t, drift_n * t)


def planned_path(state: LayerState, config: LayerConfig, points: list[Position],
                 wind_mps: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Planned flight path (飛行予定経路) from the current state through the drop points in
    order: the same turn-limited guidance as the flight (turn, then straight; a loop when a
    point is inside the turning circle), simulated at the current speed in PATH_STEP_S steps.
    In the wind (wind_mps at the current altitude) the path is flown in the air mass and drawn
    over the ground. The detours that take up early time on a timed leg are not predicted.
    Empty without points."""
    if not points:
        return []
    probe = state.model_copy(update={"mode": "TRANSIT", "task_id": -1, "planned_path": []})
    rng = random.Random(0)
    we, wn = wind_mps
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    total = 0.0  # ground = air mass + wind x total
    path = [(round(probe.position.latitude, 5), round(probe.position.longitude, 5))]
    for k, point in enumerate(points):
        elapsed = 0.0
        while elapsed < PATH_LEG_LIMIT_S and len(path) < PATH_MAX_POINTS:
            aim = _lead(probe.position, _shift(point, -we * total, -wn * total), -we, -wn, v)
            probe, arrived = step(probe, config, aim, -1 - k, aim, rng, dt=PATH_STEP_S,
                                  wanted_speed_kt=state.speed_kt)
            elapsed += PATH_STEP_S
            total += PATH_STEP_S
            if arrived:
                break
            over_ground = _shift(probe.position, we * total, wn * total)
            path.append((round(over_ground.latitude, 5), round(over_ground.longitude, 5)))
        path.append((round(point.latitude, 5), round(point.longitude, 5)))
        if len(path) >= PATH_MAX_POINTS:
            break
        probe = probe.model_copy(update={"position": _shift(point, -we * total, -wn * total), "task_id": -1 - k})
    return path


def _climb(state: LayerState, config: LayerConfig, wanted_ft: float, dt: float = 1.0) -> LayerState:
    change = config.climb_rate_fpm / 60.0 * dt
    altitude = state.altitude_ft + max(-change, min(change, wanted_ft - state.altitude_ft))
    return state if altitude == state.altitude_ft else state.model_copy(update={"altitude_ft": altitude})


def release_point(layer_position: Position, altitude_ft: float, speed_kt: float, config: LayerConfig,
                  point: Position, wind_mps: tuple[float, float], estimate_mps: tuple[float, float],
                  current_mps: tuple[float, float], terminal_mps: float) -> tuple[Position, float]:
    """Release point (投下点) for an observer that should enter the water at `point`.

    The layer approaches on the bearing to the point with the ground speed its airspeed
    (speed_kt) makes in the wind at its altitude (its navigation knows its own drift) and releases at the
    altitude it will have reached (it descends towards drop_altitude_ft on the way). The fall
    is predicted with the estimated mean wind from the drop altitude to the sea surface (0
    without an estimate: the no-wind free fall); the drop point drifts with the current during
    the fall. Returns (release point, predicted fall time s)."""
    east, north, _ = local_offset_m(layer_position, point)
    course = math.atan2(east, north)
    ge, gn = windlib.ground_velocity(course, max(speed_kt, 1.0) * KNOT_TO_MPS, *wind_mps)
    eta = math.hypot(east, north) / max(math.hypot(ge, gn), 1.0)
    change = config.climb_rate_fpm / 60.0 * eta
    altitude = altitude_ft + max(-change, min(change, config.drop_altitude_ft - altitude_ft))
    predicted = windlib.release_offset(altitude, ge, gn, *estimate_mps, terminal_mps)
    t = predicted.time_s
    release = _shift(point, current_mps[0] * t - predicted.east_m, current_mps[1] * t - predicted.north_m)
    return release, t


def advance(feed: LayerFeed, state: LayerState, rng: random.Random) -> tuple[LayerState, LayerUpdate]:
    """Advance the layer from state.tick to feed.tick (1 s steps).

    Tasks are flown in the order of their planned drop time (then the operator's drop order). The layer keeps circling the
    estimated target until it is time to leave (see departure) and flies in on the path that
    arrives at the planned time (see timed_turn; the speed is kept unless no path can make it).
    Arriving early anyway, it comes round again (HOLD) when that ends closer to the planned
    time. Tasks without a planned time are flown at once.
    Drop points drift with the estimated current.

    With feed.wind the layer flies in the air mass (the wind at its altitude carries it), heads
    for the release point of each drop (see release_point: corrected for the observer's fall in
    the estimated mean wind) and the released observer falls through the true wind profile;
    the completed drop is the point where it enters the water (releases: the fall)."""
    config = feed.config
    tasks = sorted(feed.tasks, key=lambda t: t.flight_key())
    positions = {t.task_id: t.position for t in tasks}
    completed: dict[int, Position] = {}
    releases: dict[int, DropRelease] = {}
    profile = windlib.WindProfile(feed.wind) if feed.wind is not None else None
    estimate = (0.0, 0.0)
    if profile is not None and feed.wind_estimate is not None and config.wind_correction:
        estimate = (feed.wind_estimate.east_kt * KNOT_TO_MPS, feed.wind_estimate.north_kt * KNOT_TO_MPS)
    current = (feed.current_east_kt * KNOT_TO_MPS, feed.current_north_kt * KNOT_TO_MPS)
    wind = (0.0, 0.0)
    offset = [0.0, 0.0]  # over the ground = air-mass frame + offset (the wind carries the layer)

    def air(position: Position) -> Position:
        return _shift(position, -offset[0], -offset[1])

    def over_ground(position: Position) -> Position:
        return _shift(position, offset[0], offset[1])

    tick = state.tick
    for _ in range(max(min(feed.tick - state.tick, 30), 0)):
        tick += 1
        positions = {k: drift(p, feed.current_east_kt, feed.current_north_kt) for k, p in positions.items()}
        if profile is not None:
            wind = profile.at(state.altitude_ft)
        open_tasks = [t for t in tasks if t.task_id not in completed]
        task = open_tasks[0] if open_tasks and not config.paused else None
        target = hold = wanted = rate = aim = None
        task_id = None
        datum = air(feed.datum)
        if task is not None:
            aim = positions[task.task_id]
            if profile is not None:
                aim, _ = release_point(over_ground(state.position), state.altitude_ft, state.speed_kt, config,
                                       aim, wind, estimate, current, profile.terminal_mps)
            point = air(aim)
            if profile is not None:  # in the air mass a ground point moves with current - wind
                point = _lead(state.position, point, current[0] - wind[0], current[1] - wind[1],
                              max(state.speed_kt, 1.0) * KNOT_TO_MPS)
            timed = task.planned_tick is not None
            left = task.planned_tick - tick if timed else 0
            committed = state.task_id == task.task_id and state.mode == "TRANSIT"
            holding = state.mode == "HOLD" and state.task_id == task.task_id
            decision = "leave"
            if timed and not committed:
                fastest = config.speed_kt + config.speed_spread_kt
                near = eta_to(state.model_copy(update={"speed_kt": fastest}), config, point) <= left + 600
                decision = departure(state, config, point, left, datum, point if holding else None) \
                    if near else ("hold" if holding else "wait")
            if decision == "leave":
                target, task_id = point, task.task_id
                if timed:
                    rate, wanted = timed_turn(state, config, point, left)
            elif holding:
                hold, task_id = point, task.task_id
                wanted = state.speed_kt
        to_drop = target is not None or hold is not None
        state = _climb(state, config, config.drop_altitude_ft if to_drop else config.cruise_altitude_ft)
        before = over_ground(state.position)
        state, arrived = step(state, config, target, task_id, datum, rng, hold_center=hold,
                              wanted_speed_kt=wanted, turn_rate=rate)
        offset[0] += wind[0]
        offset[1] += wind[1]
        if profile is not None and target is not None:
            # release at the instant the path passes the release point: abeam of it within this
            # second and inside the capture radius (the guidance' arrival can be up to the capture
            # radius early)
            arrived = _passes(before, over_ground(state.position), aim, config.capture_radius_yd * YD_TO_M)
        if arrived and task is not None:
            planned = task.planned_tick if task.planned_tick is not None else tick
            early = planned - tick
            # drop now when on time; otherwise come round again (HOLD) if that ends closer to the
            # planned time than dropping early now (coming round takes at least one flight back)
            back = loop_s(state.speed_kt, config)  # once past the point: at least one loop
            if early <= ON_TIME_TOLERANCE_S or max(back - early, 0.0) >= early:
                if profile is None:
                    completed[task.task_id] = positions[task.task_id]
                else:
                    drop = _release(task.task_id, tick, before, over_ground(state.position), aim,
                                    state.altitude_ft, positions[task.task_id], profile, current)
                    releases[task.task_id] = drop
                    completed[task.task_id] = drop.splash_position
            else:
                state = state.model_copy(update={"mode": "HOLD", "task_id": task.task_id})
    state = state.model_copy(update={"position": over_ground(state.position)})
    state.tick = max(state.tick, feed.tick)
    open_ids = [t.task_id for t in tasks if t.task_id not in completed]
    release_positions: dict[int, Position] = {}
    if profile is not None:
        wind = profile.at(state.altitude_ft)
        v = state.speed_kt * KNOT_TO_MPS
        heading = math.radians(state.heading_deg)
        ge, gn = v * math.sin(heading) + wind[0], v * math.cos(heading) + wind[1]
        direction, speed = windlib.wind_from(*wind)
        state = state.model_copy(update={
            "ground_speed_kt": math.hypot(ge, gn) / KNOT_TO_MPS,
            "track_deg": math.degrees(math.atan2(ge, gn)) % 360.0,
            "wind_direction_deg": direction,
            "wind_speed_kt": speed / KNOT_TO_MPS,
        })
        previous = state.position
        for task_id in open_ids:
            release_positions[task_id], _ = release_point(
                previous, state.altitude_ft, state.speed_kt, config, positions[task_id], wind, estimate,
                current, profile.terminal_mps)
            previous = positions[task_id]
    if not config.paused:
        open_points = [release_positions.get(k, positions[k]) for k in open_ids]
        state = state.model_copy(update={"planned_path": planned_path(state, config, open_points, wind)})
    eta: dict[int, float] = {}
    previous, elapsed = state.position, 0.0
    for task in (t for t in tasks if t.task_id not in completed):
        point = positions[task.task_id]
        if state.task_id == task.task_id and state.eta_s is not None:
            elapsed = state.eta_s
        else:
            east, north, _ = local_offset_m(previous, point)
            elapsed += math.hypot(east, north) / (config.speed_kt * KNOT_TO_MPS)
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
        releases=releases,
        release_positions=release_positions,
    )


def _passes(before: Position, after: Position, point: Position, radius_m: float) -> bool:
    """The step before -> after reaches the point abeam (or has passed it) within radius_m."""
    se, sn, _ = local_offset_m(before, after)
    pe, pn, _ = local_offset_m(before, point)
    along = (pe * se + pn * sn) / max(se * se + sn * sn, 1e-9)
    if along > 1.0:
        return False  # still ahead
    along = max(along, 0.0)
    return math.hypot(pe - along * se, pn - along * sn) <= radius_m


def _release(task_id: int, tick: int, before: Position, after: Position, aim: Position, altitude_ft: float,
             point: Position, profile: windlib.WindProfile, current: tuple[float, float]) -> DropRelease:
    """The observer leaves the layer where its path in this second passes closest to the
    release point (before -> after: the 1 s step over the ground) and falls freely through the
    true wind profile to the sea surface."""
    se, sn, _ = local_offset_m(before, after)  # 1 s step: also the ground velocity in m/s
    pe, pn, _ = local_offset_m(before, aim)
    along = max(0.0, min(1.0, (pe * se + pn * sn) / max(se * se + sn * sn, 1e-9)))
    release = _shift(before, along * se, along * sn).model_copy(update={"depth_ft": 0.0})
    result = windlib.fall(altitude_ft, se, sn, profile.at, profile.terminal_mps)
    splash = _shift(release, result.east_m, result.north_m).model_copy(update={"depth_ft": point.depth_ft})
    planned = _shift(point, current[0] * result.time_s, current[1] * result.time_s)
    return DropRelease(
        task_id=task_id, release_tick=tick, release_position=release, altitude_ft=altitude_ft,
        ground_east_kt=se / KNOT_TO_MPS, ground_north_kt=sn / KNOT_TO_MPS,
        splash_tick=tick + math.ceil(result.time_s), splash_position=splash,
        fall_time_s=result.time_s, planned_position=planned,
    )
