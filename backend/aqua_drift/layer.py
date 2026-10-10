"""設標者 (layer) kinematics: the craft that lays additional observers.

* moves over the sea surface at a speed drawn for every leg from speed_kt +- speed_spread_kt
* turns with bank <= max_bank_deg: turn rate omega = g tan(bank) / V, turn radius V^2 / (g tan(bank))
* TRANSIT: flies to the drop point along the shortest turn-limited (Dubins) path. When another
  drop follows, the point is crossed on the approach heading that makes the flight to this
  point and on through the following ones quickest (aqua_drift.route.route: lined up for the
  next drops instead of a loop after every drop); the last drop is flown to on any heading (turn,
  then straight; when the point is inside both turning circles it flies straight on first and
  comes back). The observer is laid when the layer is within capture_radius_yd of the point.
* Turns: left turn is the standard (circling counter-clockwise); a right turn is taken only when
  the route is clearly more efficient that way (shorter by more than turn_margin_s of flight).
* ORBIT: without a task it circles the estimated target position (vector-field
  guidance onto a circle whose radius is at least 1.2 x the turn radius, so the bank limit
  holds on the circle).
* Scheduling: a drop's planned time is the time the layer drops it flying the shortest path. The
  layer keeps circling the estimated target until the shortest flight (along the same path the
  guidance flies) takes up the time left, then leaves and flies that path at the speed it has;
  the observer is laid when it gets there. The path is never changed to keep the planned time
  (no detours, no circling at the point, no speed change on the way).
* 3-D flight in the wind (feed.wind): the layer cruises at cruise_altitude_ft, descends to
  drop_altitude_ft on the way to a drop point (climb_rate_fpm) and flies through the air mass:
  its ground velocity is the air velocity plus the wind at its altitude. The guidance runs in
  the air-mass frame (a ground point moves there with current - wind), aiming at the
  interception point. The observer is released at the release point (the drop point less the
  predicted fall displacement: the throw of the layer's ground speed and the drift of the
  estimated mean wind, or of the wind at the current altitude before there is an estimate), the exact release instant within the 1 s step is where the path passes
  closest to it, and it falls freely through the true wind profile to the sea surface
  (aqua_drift.wind).
"""
from __future__ import annotations

import math
import random

import numpy as np

from aqua_drift import route as routelib
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
                  limit_s: float = 1200.0, dt: float = 5.0, approach: float | None = None) -> float:
    """Flight time to a point by simulating the same guidance (coarse 5 s steps) at a fixed
    speed: accounts for the turn limit, loops and the fly-on-then-return manoeuvre."""
    probe = state.model_copy(update={"speed_kt": speed_kt, "mode": "TRANSIT", "task_id": -1})
    elapsed = 0.0
    rng = random.Random(0)
    while elapsed < limit_s:
        probe, arrived = step(probe, config, point, -1, point, rng, dt=dt, approach=approach)
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
    wanted_speed_kt: float | None = None,
    approach: float | None = None,
) -> tuple[LayerState, bool]:
    """Advance the layer by dt seconds. Returns (new state, arrived at the drop point).

    approach: heading (compass rad) to cross the drop point on (None: any heading).
    wanted_speed_kt: the speed of a new leg (within the band; None: drawn). A leg keeps its speed.

    target given -> TRANSIT to it; otherwise ORBIT around the datum (estimated target position)."""
    mode = "TRANSIT" if target is not None else "ORBIT"
    speed = state.speed_kt
    if mode != state.mode or task_id != state.task_id:
        if wanted_speed_kt is not None:
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
    path_step = None  # the motion along the guidance path over this step (east, north m)
    path_kept = ([], [])  # the rest of the path after this step
    center = datum
    orbit_radius = orbit_radius_m(config, speed)

    if mode == "TRANSIT":
        east, north, _ = local_offset_m(position, target)
        distance = math.hypot(east, north)
        if distance <= config.capture_radius_yd * YD_TO_M:
            arrived = True
            turn = 0.0
            eta = 0.0
        else:
            # the motion over this step along the path (also across a join of its segments)
            flown = state.path_sides if state.mode == "TRANSIT" and state.task_id == task_id else []
            turn, path_east, path_north, length, path = _guidance(
                east, north, heading, approach, radius_turn, config, v, dt,
                (flown, state.path_lengths_m) if flown and speed == state.speed_kt else None)
            if not math.isfinite(length):
                turn = 0.0  # inside both turning circles: fly on first
                eta = distance / v + math.pi / omega_max
            else:
                eta = length / v
                path_step = (path_east, path_north)
                path_kept = path
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
    if path_step is not None:  # along the path (it can change from one turn to the other within the step)
        sx, sy = path_step
    elif abs(turn) < 1e-12:
        sx, sy = v * dt * math.sin(heading), v * dt * math.cos(heading)
    else:  # an arc at a constant turn rate
        r = v * dt / turn
        sx, sy = r * (math.cos(heading) - math.cos(new_heading)), r * (math.sin(new_heading) - math.sin(heading))
    moved = _offset(position, sx, sy, 0.0)
    if mode == "TRANSIT" and not arrived:
        # closest approach within this step (steps are ~100 m long at 200 kt)
        px, py, _ = local_offset_m(position, target)
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
        task_id=task_id if mode == "TRANSIT" else None,
        orbit_center=center if mode != "TRANSIT" else None,
        orbit_radius_yd=orbit_radius / YD_TO_M if mode != "TRANSIT" else 0.0,
        eta_s=eta,
        altitude_ft=state.altitude_ft,
        approach_deg=state.approach_deg,
        approach_key=state.approach_key,
        path_sides=path_kept[0],
        path_lengths_m=path_kept[1],
    ), arrived


GUIDANCE_SLACK_M = 5.0  # a fresh plan up to this much longer than the path being flown replaces it


def _guidance(east: float, north: float, heading: float, approach: float | None, radius: float,
              config: LayerConfig, v: float, dt: float, flown: tuple[list[int], list[float]] | None
              ) -> tuple[float, float, float, float, tuple[list[int], list[float]]]:
    """(turn rad, east m, north m: the motion over this step along the path; path length m;
    the rest of the path after this step) to the point.

    Every second the shortest path is planned afresh (it follows the point as it drifts). The
    path being flown (`flown`: sides, remaining lengths m) is kept instead when the fresh plan
    is longer and the path still ends within half the capture radius of the point: the motion
    is exact along the path, but at the joins of its segments a tiny error can make the exact
    plan jump to another, longer path (a degenerate turn-turn path, or a last turn that has
    become a whisker too tight: a loop). Close to the point, when the fresh plan is a loop and
    no path is kept, the loose path (see route.path_to) instead."""
    args = (east, north, heading, approach, radius)
    options = {"prefer": preferred_side(config), "margin_m": config.turn_margin_s * v, "step_m": v * dt,
               "segments": True}
    turn, length, _, step_e, step_n, sides, lengths = routelib.path_to(*args, **options)
    path = ([int(x) for x in sides], [float(x) * radius for x in lengths])
    kept = False
    if flown is not None and sum(flown[1]) > v * dt and float(length) > sum(flown[1]) + GUIDANCE_SLACK_M:
        sides_n, lengths_n = tuple(flown[0]), tuple(x / radius for x in flown[1])
        _, end_e, end_n = routelib.motion(sides_n, lengths_n, heading, radius, sum(lengths_n))
        if math.hypot(float(end_e) - east, float(end_n) - north) <= config.capture_radius_yd * YD_TO_M / 2.0:
            path, length, kept = flown, sum(flown[1]), True
            turn, step_e, step_n = routelib.motion(sides_n, lengths_n, heading, radius, v * dt / radius)
    if not kept and approach is not None and float(length) > math.hypot(east, north) + math.pi * radius:
        loose = routelib.path_to(*args, loose=True, capture_m=config.capture_radius_yd * YD_TO_M, **options)
        if float(loose[1]) + math.pi * radius < float(length):
            turn, length, _, step_e, step_n, sides, lengths = loose
            path = ([int(x) for x in sides], [float(x) * radius for x in lengths])
    rest, left = [], v * dt
    for piece in path[1]:
        take = min(max(left, 0.0), piece)
        rest.append(piece - take)
        left -= take
    return float(turn), float(step_e), float(step_n), float(length), (path[0], rest)


def drift(position: Position, east_kt: float, north_kt: float, dt: float = 1.0) -> Position:
    """A drop point planned in the water frame moves with the (estimated) current."""
    return _offset(position, east_kt * KNOT_TO_MPS * dt, north_kt * KNOT_TO_MPS * dt, position.depth_ft)


LOOKAHEAD_S = 400  # longest stretch of the orbit over which the layer looks for the best departure
DEPART_TIE_S = 2.0  # a later departure must be this much closer to the planned time to wait for it
APPROACH_TOLERANCE_S = 5.0  # on the way, a new approach heading may make the drop this much later


def loop_s(speed_kt: float, config: LayerConfig) -> float:
    v = speed_kt * KNOT_TO_MPS
    return 2.0 * math.pi * v / (G * math.tan(math.radians(config.max_bank_deg)))


def _arrival_s(state: LayerState, config: LayerConfig, point: Position, approach: float | None = None) -> float:
    """Flight time at the current speed along the guidance path (coarse simulation when the
    point is inside both turning circles)."""
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    east, north, _ = local_offset_m(state.position, point)
    radius = v * v / (G * math.tan(math.radians(config.max_bank_deg)))
    _, length, _ = routelib.path_to(east, north, math.radians(state.heading_deg), approach, radius,
                                    preferred_side(config), config.turn_margin_s * v, loose=True)
    if not math.isfinite(float(length)):
        return flight_time_s(state, config, point, state.speed_kt, limit_s=900.0, approach=approach)
    return max(float(length) - config.capture_radius_yd * YD_TO_M, 0.0) / v


def _dubins_times(east: np.ndarray, north: np.ndarray, heading: np.ndarray, v: float,
                  config: LayerConfig, approach: float | np.ndarray | None = None) -> np.ndarray:
    """The guidance path for arrays of relative positions/headings: flight time (s) to within
    the capture radius at speed v (m/s); inf where the point is inside both turning circles."""
    radius = v * v / (G * math.tan(math.radians(config.max_bank_deg)))
    if approach is not None:
        east, north, heading, approach = np.broadcast_arrays(east, north, heading, approach)
    approaches = approach
    _, length, _ = routelib.path_to(east, north, heading, approaches, radius, preferred_side(config),
                                    config.turn_margin_s * v, loose=True)
    return np.maximum(length - config.capture_radius_yd * YD_TO_M, 0.0) / v


def _leg_times(state: LayerState, config: LayerConfig, point: Position, headings: np.ndarray) -> np.ndarray:
    """Flight time (s) to the point per approach heading (compass rad) along the guidance path
    at the current speed."""
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    east, north, _ = local_offset_m(state.position, point)
    headings = np.asarray(headings, dtype=float)
    return _dubins_times(np.full(len(headings), east), np.full(len(headings), north),
                         np.full(len(headings), math.radians(state.heading_deg)), v, config, headings)


def departure(state: LayerState, config: LayerConfig, point: Position, left: float,
              datum: Position, approach: float | None = None) -> str:
    """'leave' or 'wait' on the orbit at the current speed. For each departure time over the
    next round of the orbit (1 s steps, at most LOOKAHEAD_S s) the shortest flight to the
    point (the guidance path) gives an arrival; the layer leaves now when now is the departure
    that arrives closest to the planned time, or within DEPART_TIE_S of it (flying towards the
    point on the orbit only puts off a departure that is late anyway). Once on its way the
    layer flies that path: the planned time is not kept by changing it."""
    now = _arrival_s(state, config, point, approach)
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    radius = orbit_radius_m(config, state.speed_kt)
    window = min(math.ceil(2.0 * math.pi * radius / v), LOOKAHEAD_S)
    east, north, _ = local_offset_m(datum, point)
    longest = (math.hypot(east, north) + 2.0 * radius) / v + loop_s(state.speed_kt, config) + 60.0
    if left - window > longest:
        return "wait"
    probe = state
    rng = random.Random(0)
    poses = []
    for _ in range(window):
        probe, _ = step(probe, config, None, None, datum, rng, wanted_speed_kt=state.speed_kt)
        e, n, _ = local_offset_m(probe.position, point)
        poses.append((e, n, math.radians(probe.heading_deg)))
    e, n, h = (np.array(x) for x in zip(*poses, strict=True))
    error = np.arange(1, window + 1) + _dubins_times(e, n, h, v, config, approach) - left
    error = np.where(np.isfinite(error), np.abs(error), np.inf)
    return "leave" if abs(now - left) <= float(np.min(error)) + DEPART_TIE_S else "wait"


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


def _lead(origin: Position, point: Position, drift_e: float, drift_n: float, v: float,
          flight_s: float | None = None) -> Position:
    """Where a point moving at (drift_e, drift_n) m/s is met (air-mass frame interception):
    flying straight at v, or after flight_s s (on the way: the time left along the path)."""
    if not drift_e and not drift_n:
        return point
    east, north, _ = local_offset_m(origin, point)
    t = _intercept_s(east, north, drift_e, drift_n, v) if flight_s is None else flight_s
    return _shift(point, drift_e * t, drift_n * t)


def planned_path(state: LayerState, config: LayerConfig, points: list[Position],
                 wind_mps: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """Planned flight path (飛行予定経路) from the current state through the drop points in
    order: the same turn-limited guidance as the flight (each point crossed on the approach
    heading lined up for the next ones, see route.route; the last point turn, then straight),
    simulated at the current speed in PATH_STEP_S steps.
    In the wind (wind_mps at the current altitude) the path is flown in the air mass and drawn
    over the ground. Empty without points."""
    if not points:
        return []
    probe = state.model_copy(update={"mode": "TRANSIT", "task_id": -1, "planned_path": []})
    rng = random.Random(0)
    we, wn = wind_mps
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    total = 0.0  # ground = air mass + wind x total
    path = [(round(probe.position.latitude, 5), round(probe.position.longitude, 5))]
    legs = _route(state, config, points, [None] * len(points))
    for k, point in enumerate(points):
        elapsed = 0.0
        approach = legs[k].approach
        if k == 0 and len(points) > 1 and state.approach_deg is not None:
            approach = math.radians(state.approach_deg)  # the heading the layer is flying for
        while elapsed < PATH_LEG_LIMIT_S and len(path) < PATH_MAX_POINTS:
            aim = _lead(probe.position, _shift(point, -we * total, -wn * total), -we, -wn, v)
            probe, arrived = step(probe, config, aim, -1 - k, aim, rng, dt=PATH_STEP_S,
                                  wanted_speed_kt=state.speed_kt, approach=approach)
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
    return _thin(path)


PATH_THIN_M = 10.0  # a path point within this of the line through its neighbours is left out


def _thin(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The path without the points of its straight parts (each within PATH_THIN_M of the line
    from the last point kept to the next one): the same line, sent every second in fewer points."""
    if len(path) < 3:
        return path
    lat0 = math.radians(path[0][0])

    def xy(p: tuple[float, float]) -> tuple[float, float]:
        return p[1] * 111320.0 * math.cos(lat0), p[0] * 110540.0

    kept = [path[0]]
    for k in range(1, len(path) - 1):
        (ax, ay), (bx, by), (cx, cy) = xy(kept[-1]), xy(path[k]), xy(path[k + 1])
        span = math.hypot(cx - ax, cy - ay)
        off = abs((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / span if span > 0 else math.hypot(bx - ax, by - ay)
        if off > PATH_THIN_M:
            kept.append(path[k])
    kept.append(path[-1])
    return kept


def _climb(state: LayerState, config: LayerConfig, wanted_ft: float, dt: float = 1.0) -> LayerState:
    change = config.climb_rate_fpm / 60.0 * dt
    altitude = state.altitude_ft + max(-change, min(change, wanted_ft - state.altitude_ft))
    return state if altitude == state.altitude_ft else state.model_copy(update={"altitude_ft": altitude})


def release_point(layer_position: Position, altitude_ft: float, speed_kt: float, config: LayerConfig,
                  point: Position, wind_mps: tuple[float, float], estimate_mps: tuple[float, float],
                  current_mps: tuple[float, float], terminal_mps: float,
                  approach_deg: float | None = None) -> tuple[Position, float]:
    """Release point (投下点) for an observer that should enter the water at `point`.

    The layer approaches on the bearing to the point with the ground speed its airspeed
    (speed_kt) makes in the wind at its altitude (its navigation knows its own drift) and releases at the
    altitude it will have reached (it descends towards drop_altitude_ft on the way). The fall
    is predicted with the correction wind estimate_mps (see correction_wind: the estimated mean
    wind from the drop altitude to the sea surface, else the wind at the current altitude; 0: the
    no-wind free fall); the drop point drifts with the current during
    the fall. With approach_deg (the air-mass heading the layer crosses the point on, see
    _approach) the release is predicted on that heading rather than the bearing from the layer, so the release
    point stays put while the layer turns onto it. Returns (release point, predicted fall time s)."""
    east, north, _ = local_offset_m(layer_position, point)
    airspeed = max(speed_kt, 1.0) * KNOT_TO_MPS
    if approach_deg is None:
        ge, gn = windlib.ground_velocity(math.atan2(east, north), airspeed, *wind_mps)
    else:  # the heading in the air mass: the wind adds to the air velocity
        heading = math.radians(approach_deg)
        ge = airspeed * math.sin(heading) + wind_mps[0]
        gn = airspeed * math.cos(heading) + wind_mps[1]
    eta = math.hypot(east, north) / max(math.hypot(ge, gn), 1.0)
    change = config.climb_rate_fpm / 60.0 * eta
    altitude = altitude_ft + max(-change, min(change, config.drop_altitude_ft - altitude_ft))
    predicted = windlib.release_offset(altitude, ge, gn, *estimate_mps, terminal_mps)
    t = predicted.time_s
    release = _shift(point, current_mps[0] * t - predicted.east_m, current_mps[1] * t - predicted.north_m)
    return release, t


ROUTE_AHEAD = 8  # drops in the route when choosing the approach heading (the current one and up to 7 after it)


def _route(state: LayerState, config: LayerConfig, points: list[Position],
           planned: list[float | None], first_s: np.ndarray | None = None,
           first_headings: np.ndarray | None = None) -> list[routelib.Leg]:
    """The route from the current state through the points (planned drop times s from now;
    first_s, first_headings: see route.route)."""
    offsets = [local_offset_m(state.position, p) for p in points]
    v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
    return routelib.route([o[0] for o in offsets], [o[1] for o in offsets], math.radians(state.heading_deg),
                          state.speed_kt, config.max_bank_deg, planned, preferred_side(config),
                          config.turn_margin_s * v, capture_m=config.capture_radius_yd * YD_TO_M, first_s=first_s,
                          first_headings=first_headings)


def ready_pose(state: LayerState | None, config: LayerConfig, tasks: list, tick: int,
               default: Position) -> tuple[int, Position, float | None]:
    """When, where and on which heading (deg, None: unknown) the layer is free for a new drop:
    after its open tasks (planned time, else the expected one), on the heading it crosses the
    last of them along its route (see route.route)."""
    ready, where = tick, state.position if state is not None else default
    heading = state.heading_deg if state is not None else None
    tasks = sorted(tasks, key=lambda t: t.flight_key())
    last = None
    for k, task in enumerate(tasks):
        due = task.planned_tick if task.planned_tick is not None else tick + int(task.eta_s or 0)
        if due >= ready:
            ready, where, last = due, task.position, k
    if last is not None:
        heading = None
        if state is not None:
            legs = _route(state, config, [t.position for t in tasks[:last + 1]],
                          [None if t.planned_tick is None else float(t.planned_tick - tick) for t in tasks[:last + 1]])
            heading = math.degrees(legs[-1].heading) % 360.0
    return ready, where, heading


def _approach(state: LayerState, config: LayerConfig, tasks: list, points: list[Position], tick: int,
              committed: bool) -> LayerState:
    """The heading to cross the current drop point on, lined up for the following drops (see
    route.route). Chosen again every second until the layer is on its way, then kept while the
    drops stay the same. A new following drop on the way: chosen again among the headings that
    do not make the current drop later (or further off its planned time) than the heading flown."""
    key = ">".join(str(t.task_id) for t in tasks)
    if len(tasks) == 1:
        if state.approach_deg is None and state.approach_key == key:
            return state
        return state.model_copy(update={"approach_deg": None, "approach_key": key, "path_sides": [],
                                        "path_lengths_m": []})
    if committed and state.approach_key == key:
        return state
    planned = [None if t.planned_tick is None else float(t.planned_tick - tick) for t in tasks]
    offsets = [local_offset_m(state.position, p) for p in points]
    headings = routelib.approach_headings([o[0] for o in offsets], [o[1] for o in offsets], 0)
    # the heading flown stays a candidate (the last one): the approach, or on the way to a point
    # on any heading (it was the last drop) the heading that path arrives on
    flown = None if state.approach_deg is None else math.radians(state.approach_deg)
    if flown is None and committed:
        v = max(state.speed_kt, 1.0) * KNOT_TO_MPS
        _, _, final = routelib.path_to(offsets[0][0], offsets[0][1], math.radians(state.heading_deg), None,
                                       turn_radius_m(state.speed_kt, config.max_bank_deg), preferred_side(config),
                                       config.turn_margin_s * v)
        flown = float(final) if math.isfinite(float(final)) else None
    if flown is not None:
        headings = np.append(headings, flown)
    # on the way the drop is not delayed: the path flown arrives in eta_s, and a new approach
    # heading may not make it later
    first = None
    if committed and state.eta_s is not None and math.isfinite(state.eta_s):
        first = _leg_times(state, config, points[0], headings)
        kept = np.where(first > state.eta_s + APPROACH_TOLERANCE_S, np.inf, first)
        if flown is not None:  # the heading flown: as the guidance flies it
            kept[-1] = state.eta_s
        if np.isfinite(kept).any():
            first = kept
    legs = _route(state, config, points, planned, first, headings)
    approach = math.degrees(legs[0].approach) % 360.0
    if approach == state.approach_deg:
        return state.model_copy(update={"approach_key": key})
    return state.model_copy(update={"approach_deg": approach, "approach_key": key, "path_sides": [],
                                    "path_lengths_m": []})


# where the wind of the release correction comes from (LayerState.correction_source)
ESTIMATED_WIND = "estimate"  # mean wind estimated from the last drop (drop altitude to the sea surface)
FLIGHT_ALTITUDE_WIND = "flight_altitude"  # no estimate yet: the wind at the layer's current altitude
NO_CORRECTION = "none"  # correction off or no wind: the no-wind free fall


def correction_source(feed: LayerFeed) -> str:
    """The wind the release points are corrected with: the estimated mean wind when there is
    one, else the wind at the layer's current altitude (its navigation knows its own drift)."""
    if feed.wind is None or not feed.config.wind_correction:
        return NO_CORRECTION
    return ESTIMATED_WIND if feed.wind_estimate is not None else FLIGHT_ALTITUDE_WIND


def correction_wind(feed: LayerFeed, source: str,
                    flight_wind_mps: tuple[float, float]) -> tuple[float, float]:
    """(east, north) m/s towards of the correction wind from `source` (flight_wind_mps: the wind
    at the layer's current altitude)."""
    if source == ESTIMATED_WIND:
        return feed.wind_estimate.east_kt * KNOT_TO_MPS, feed.wind_estimate.north_kt * KNOT_TO_MPS
    if source == FLIGHT_ALTITUDE_WIND:
        return flight_wind_mps
    return 0.0, 0.0


def advance(feed: LayerFeed, state: LayerState, rng: random.Random) -> tuple[LayerState, LayerUpdate]:
    """Advance the layer from state.tick to feed.tick (1 s steps).

    Tasks are flown in the order of their planned drop time (then the operator's drop order). The layer keeps circling the
    estimated target until it is time to leave (see departure), then flies the shortest path
    (the guidance path) at the speed it has and drops when it gets there: the path and the
    speed are not changed to keep the planned time. Tasks without a planned time are flown at once.
    Drop points drift with the estimated current.

    With feed.wind the layer flies in the air mass (the wind at its altitude carries it), heads
    for the release point of each drop (see release_point: corrected for the observer's fall in
    the estimated mean wind, or the wind at its altitude before there is an estimate) and the released observer falls through the true wind profile;
    the completed drop is the point where it enters the water (releases: the fall)."""
    config = feed.config
    tasks = sorted(feed.tasks, key=lambda t: t.flight_key())
    positions = {t.task_id: t.position for t in tasks}
    completed: dict[int, Position] = {}
    releases: dict[int, DropRelease] = {}
    profile = windlib.WindProfile(feed.wind) if feed.wind is not None else None
    source = correction_source(feed)
    current = (feed.current_east_kt * KNOT_TO_MPS, feed.current_north_kt * KNOT_TO_MPS)
    wind = (0.0, 0.0)
    estimate = correction_wind(feed, source, wind)
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
            estimate = correction_wind(feed, source, wind)
        open_tasks = [t for t in tasks if t.task_id not in completed]
        task = open_tasks[0] if open_tasks and not config.paused else None
        target = wanted = aim = approach = None
        task_id = None
        datum = air(feed.datum)
        if task is not None:
            aim = positions[task.task_id]
            if profile is not None:
                aim, _ = release_point(over_ground(state.position), state.altitude_ft, state.speed_kt, config,
                                       aim, wind, estimate, current, profile.terminal_mps,
                                       state.approach_deg if state.task_id == task.task_id else None)
            point = air(aim)
            if profile is not None:  # in the air mass a ground point moves with current - wind
                # on the way: met at the arrival along the path flown (the straight-line time
                # would move the point as the path bends, and the path would never close on it)
                on_way = (state.task_id == task.task_id and state.mode == "TRANSIT" and state.eta_s is not None
                          and math.isfinite(state.eta_s))
                point = _lead(state.position, point, current[0] - wind[0], current[1] - wind[1],
                              max(state.speed_kt, 1.0) * KNOT_TO_MPS, max(state.eta_s - 1.0, 0.0) if on_way else None)
            timed = task.planned_tick is not None
            left = task.planned_tick - tick if timed else 0
            committed = state.task_id == task.task_id and state.mode == "TRANSIT"
            ahead = open_tasks[1:ROUTE_AHEAD]
            state = _approach(state, config, [task, *ahead], [point, *(air(positions[t.task_id]) for t in ahead)],
                              tick, committed)
            approach = None if state.approach_deg is None else math.radians(state.approach_deg)
            decision = "leave"
            if timed and not committed:
                fastest = config.speed_kt + config.speed_spread_kt
                near = eta_to(state.model_copy(update={"speed_kt": fastest}), config, point) <= left + 600
                decision = departure(state, config, point, left, datum, approach) if near else "wait"
            if decision == "leave":
                target, task_id = point, task.task_id
                if timed:  # left when the flight at this speed takes up the time: keep the speed
                    wanted = state.speed_kt
        to_drop = target is not None
        state = _climb(state, config, config.drop_altitude_ft if to_drop else config.cruise_altitude_ft)
        before = over_ground(state.position)
        state, arrived = step(state, config, target, task_id, datum, rng, wanted_speed_kt=wanted,
                              approach=approach)
        offset[0] += wind[0]
        offset[1] += wind[1]
        if profile is not None and target is not None:
            # release at the instant the path passes the release point: abeam of it within this
            # second and inside the capture radius (the guidance' arrival can be up to the capture
            # radius early)
            arrived = _passes(before, over_ground(state.position), aim, config.capture_radius_yd * YD_TO_M)
        if arrived and task is not None:  # dropped when it gets there, early or late
            if profile is None:
                completed[task.task_id] = positions[task.task_id]
            else:
                drop = _release(task.task_id, tick, before, over_ground(state.position), aim,
                                state.altitude_ft, positions[task.task_id], profile, current)
                releases[task.task_id] = drop
                completed[task.task_id] = drop.splash_position
    state = state.model_copy(update={"position": over_ground(state.position)})
    state.tick = max(state.tick, feed.tick)
    open_ids = [t.task_id for t in tasks if t.task_id not in completed]
    release_positions: dict[int, Position] = {}
    if profile is not None:
        wind = profile.at(state.altitude_ft)
        estimate = correction_wind(feed, source, wind)
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
                current, profile.terminal_mps, state.approach_deg if task_id == state.task_id else None)
            previous = positions[task_id]
    corrected = source != NO_CORRECTION
    direction, speed = windlib.wind_from(*estimate)
    state = state.model_copy(update={
        "correction_source": source,
        "correction_wind_direction_deg": direction if corrected else None,
        "correction_wind_speed_kt": speed / KNOT_TO_MPS if corrected else None,
    })
    if not config.paused:
        open_points = [release_positions.get(k, positions[k]) for k in open_ids]
        state = state.model_copy(update={"planned_path": planned_path(state, config, open_points, wind)})
    # expected drop time from now along the route (see route.route): the planned time when the
    # layer can wait for it on its orbit, else the arrival
    eta: dict[int, float] = {}
    remaining = [t for t in tasks if t.task_id not in completed]
    legs = _route(state, config, [positions[t.task_id] for t in remaining],
                  [None if t.planned_tick is None else float(t.planned_tick - feed.tick) for t in remaining])
    shift = 0.0
    for task, leg in zip(remaining, legs, strict=True):
        drop = leg.drop_s + shift
        if state.task_id == task.task_id and state.mode == "TRANSIT" and state.eta_s is not None:
            drop = state.eta_s  # flying it: the guidance' arrival (the path is not changed)
            shift = drop - leg.drop_s
        eta[task.task_id] = round(max(drop, 0.0), 1)
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
