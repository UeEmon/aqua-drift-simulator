"""Flight route of the layer (設標者) through its drop points.

The layer turns with a fixed radius r = V^2 / (g tan(bank)), so the shortest path between two
poses is a Dubins path: turn - straight - turn (LSL, RSR, LSR, RSL) or three turns (RLR, LRL).

An observer can be dropped on any heading, so the old guidance flew "turn, then straight" to
the point and arrived on whatever heading that gave. With drop points closer together than a
turning diameter (typical: a pair either side of the target track, 2-4 km apart, at 200 kt a
turn radius of about 4 km) the next point was then behind or beside the layer and every drop
cost a loop of 2 pi r (about 4 min). The route therefore chooses the heading the layer crosses
each drop point on (the approach heading): the one that makes the flight to this point plus
the flight on through the following points shortest, with the planned drop times kept (see
route: the approach headings of all the points are chosen together).

Angles: headings are compass radians (clockwise from north), east / north in metres; a left
turn is counter-clockwise (heading decreasing), side -1; a right turn is side +1.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

G = 9.80665
KNOT_TO_MPS = 0.5144444444444445
TWO_PI = 2.0 * math.pi
LOOP_TOLERANCE = math.radians(20.0)  # a turn-then-straight path this close to the approach heading instead of a loop
APPROACH_CANDIDATES = 36  # approach headings compared at each point (every 10 degrees)
LATE_WEIGHT = 5.0  # a second late at a drop costs this many seconds of flight
ORDER_LIMIT = 5  # drops of one plan ordered by trying every order (more: by drop time)

# Dubins words: (first turn, middle, last turn) with +1 right / -1 left / 0 straight
WORDS = {"LSL": (-1, 0, -1), "RSR": (1, 0, 1), "LSR": (-1, 0, 1), "RSL": (1, 0, -1),
         "RLR": (1, -1, 1), "LRL": (-1, 1, -1)}


def turn_radius_m(speed_kt: float, bank_deg: float) -> float:
    v = max(speed_kt, 1.0) * KNOT_TO_MPS
    return v * v / (G * math.tan(math.radians(bank_deg)))


def _mod(angle):
    return np.mod(angle, TWO_PI)


EPS = 1e-6  # numerical slack of the Dubins formulas (normalised lengths / radians)


def _arc(angle):
    """Turn angle in [0, 2 pi), a whisker short of a full turn read as none (a craft exactly
    at the end of a turn must not be sent round again)."""
    angle = _mod(angle)
    return np.where(angle > TWO_PI - 1e-4, 0.0, angle)


def dubins_words(east, north, heading, approach, radius: float) -> dict[str, tuple]:
    """Every Dubins path from (0, 0) on `heading` to (east, north) on `approach` (arrays or
    scalars, compass radians). Returns {word: (first arc, middle, last arc)} in radians of
    turn / radius-normalised straight length; inf where the word does not exist."""
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    # to the usual mathematical frame: x east, y north, angles counter-clockwise from east
    th0, th1 = math.pi / 2.0 - np.asarray(heading, dtype=float), math.pi / 2.0 - np.asarray(approach, dtype=float)
    d = np.hypot(east, north) / radius
    phi = np.arctan2(north, east)
    a, b = _mod(th0 - phi), _mod(th1 - phi)
    sa, sb, ca, cb, cab = np.sin(a), np.sin(b), np.cos(a), np.cos(b), np.cos(a - b)
    inf = np.inf
    out = {}
    with np.errstate(invalid="ignore"):
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb)
        tmp = np.arctan2(cb - ca, d + sa - sb)
        out["LSL"] = (np.where(p2 >= -EPS, _arc(tmp - a), inf), np.sqrt(np.maximum(p2, 0)), _arc(b - tmp))
        p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sb - sa)
        tmp = np.arctan2(ca - cb, d - sa + sb)
        out["RSR"] = (np.where(p2 >= -EPS, _arc(a - tmp), inf), np.sqrt(np.maximum(p2, 0)), _arc(tmp - b))
        p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb)
        p = np.sqrt(np.maximum(p2, 0))
        tmp = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
        out["LSR"] = (np.where(p2 >= -EPS, _arc(tmp - a), inf), p, _arc(tmp - b))
        p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb)
        p = np.sqrt(np.maximum(p2, 0))
        tmp = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
        out["RSL"] = (np.where(p2 >= -EPS, _arc(a - tmp), inf), p, _arc(b - tmp))
        c = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0
        p = _mod(TWO_PI - np.arccos(np.clip(c, -1.0, 1.0)))
        t = _arc(a - np.arctan2(ca - cb, d - sa + sb) + p / 2.0)
        out["RLR"] = (np.where(np.abs(c) <= 1.0 + EPS, t, inf), p, _arc(a - b - t + p))
        c = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sb - sa)) / 8.0
        p = _mod(TWO_PI - np.arccos(np.clip(c, -1.0, 1.0)))
        t = _arc(-a - np.arctan2(ca - cb, d + sa - sb) + p / 2.0)
        out["LRL"] = (np.where(np.abs(c) <= 1.0 + EPS, t, inf), p, _arc(b - a - t + p))
    return out


def turn_then_straight(east, north, heading, radius: float, prefer: int = -1, margin_m: float = 0.0):
    """Turn-then-straight path (Dubins CS) to a point on any heading (arrays or scalars).

    The preferred side (default -1 = left turn) is taken unless the other side is shorter by
    more than margin_m. Returns (side +1 / -1, 0 where none is possible: the point is inside
    both turning circles; turn angle rad; straight length m; length m, inf where none)."""
    east, north, heading = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (east, north, heading)))
    paths = {}
    for side in (1, -1):
        cx, cy = side * radius * np.cos(heading), -side * radius * np.sin(heading)
        dx, dy = east - cx, north - cy
        dist = np.hypot(dx, dy)
        straight = np.sqrt(np.maximum(dist * dist - radius * radius, 0.0))
        phi, theta0, tangent = np.arctan2(dx, dy), np.arctan2(-cx, -cy), np.arctan2(straight, radius)
        arc = _arc(phi - tangent - theta0 if side == 1 else theta0 - phi - tangent)
        paths[side] = (arc, straight, np.where(dist >= radius, arc * radius + straight, np.inf))
    lp, lo = paths[prefer][2], paths[-prefer][2]
    take = np.isfinite(lp) & (lp <= lo + margin_m)
    side = np.where(take, prefer, np.where(np.isfinite(lo), -prefer, 0))
    arc = np.where(take, paths[prefer][0], paths[-prefer][0])
    straight = np.where(take, paths[prefer][1], paths[-prefer][1])
    return side, arc, straight, np.where(take, lp, lo)


def motion(sides, lengths, heading, radius: float, step):
    """Heading change and displacement (east, north m) over the first `step` (radius-normalised
    distance) along a path of three segments (turn sides, radius-normalised lengths)."""
    h = np.asarray(heading, dtype=float)
    east = north = 0.0
    left = step
    for side, length in zip(sides, lengths, strict=True):
        take = np.clip(left, 0.0, np.where(np.isfinite(length), length, 0.0))
        turn = side * take
        safe = np.where(side == 0, 1.0, side)
        # straight: along h; turn: the chord of the arc
        east = east + radius * np.where(side == 0, take * np.sin(h), (np.cos(h) - np.cos(h + turn)) / safe)
        north = north + radius * np.where(side == 0, take * np.cos(h), (np.sin(h + turn) - np.sin(h)) / safe)
        h = h + turn
        left = left - take
    return h - np.asarray(heading, dtype=float), east, north


def path_to(east, north, heading, approach, radius: float, prefer: int = -1, margin_m: float = 0.0,
            step_m: float = 0.0, loose: bool = False, capture_m: float = 0.0, segments: bool = False):
    """Path to a point crossed on the approach heading (None or nan: any heading).

    The shortest Dubins path, the preferred side first unless a path starting with the other
    turn is shorter by more than margin_m. Arrays or scalars.

    loose: instead of a loop (close to the point a small offset cannot be flown out any more),
    a turn-then-straight path within LOOP_TOLERANCE of the approach heading, or -- when the
    craft is turning onto the point and a small error has put the point just inside its turning
    circle -- a full-rate turn on that side that passes within capture_m / 2 of the point near
    the approach heading.

    Returns (signed heading change rad over the first step_m metres of the path, +right / -left;
    length m, inf where no path: any heading and the point inside both turning circles;
    heading at the point), and with segments=True also the displacement (east, north m) over
    the first step_m metres along the path and the path itself (three turn sides, three
    radius-normalised lengths)."""
    heading = np.asarray(heading, dtype=float)
    side, arc, straight, length = turn_then_straight(east, north, heading, radius, prefer, margin_m)
    zero = np.zeros_like(length)
    sides = [side + zero, zero, zero]
    lengths = [arc + zero, straight / radius + zero, zero]
    final = heading + side * arc
    if approach is not None:
        approach = np.asarray(approach, dtype=float)
        free = np.isnan(approach)
        off = np.abs(np.mod(final - approach + math.pi, TWO_PI) - math.pi)
        goal = np.where(free, 0.0, approach)
        words = dubins_words(east, north, heading, goal, radius)
        best = {}
        for start in (prefer, -prefer):
            chosen = None
            for w, parts in words.items():
                if WORDS[w][0] != start:
                    continue
                here = (parts[0] + parts[1] + parts[2]) * radius
                values = [here, *(np.full_like(here, x) for x in WORDS[w]), *parts]
                if chosen is None:
                    chosen = values
                else:
                    better = here < chosen[0]
                    chosen = [np.where(better, a, b) for a, b in zip(values, chosen, strict=True)]
            best[start] = chosen
        take = best[prefer][0] <= best[-prefer][0] + margin_m
        dubins = [np.where(take, a, b) for a, b in zip(best[prefer], best[-prefer], strict=True)]
        within = (off <= LOOP_TOLERANCE) & (dubins[0] > length + math.pi * radius) if loose else False
        lined_up = free | (np.isfinite(length) & within)
        sides = [np.where(lined_up, a, b) for a, b in zip(sides, dubins[1:4], strict=True)]
        lengths = [np.where(lined_up, a, b) for a, b in zip(lengths, dubins[4:7], strict=True)]
        length = np.where(lined_up, length, dubins[0])
        final = np.where(lined_up, final, goal)
        if loose and capture_m > 0.0:
            for turn_side in (1, -1):
                cx, cy = turn_side * radius * np.cos(heading), -turn_side * radius * np.sin(heading)
                dx, dy = np.asarray(east, dtype=float) - cx, np.asarray(north, dtype=float) - cy
                arc = _mod(turn_side * (np.arctan2(dx, dy) - np.arctan2(-cx, -cy)))
                on_arc = np.abs(np.hypot(dx, dy) - radius) <= capture_m / 2.0
                arrive = heading + turn_side * arc
                near = free | (np.abs(np.mod(arrive - approach + math.pi, TWO_PI) - math.pi) <= LOOP_TOLERANCE)
                use = on_arc & near & (arc * radius + math.pi * radius < length)  # instead of a loop
                sides = [np.where(use, turn_side, sides[0]), np.where(use, 0, sides[1]), np.where(use, 0, sides[2])]
                lengths = [np.where(use, arc, lengths[0]), np.where(use, 0.0, lengths[1]),
                           np.where(use, 0.0, lengths[2])]
                length = np.where(use, arc * radius, length)
                final = np.where(use, arrive, final)
    turn, step_east, step_north = motion(tuple(sides), tuple(lengths), heading, radius, step_m / radius)
    if segments:
        return turn, length, final, step_east, step_north, sides, lengths
    return turn, length, final


def _bearing(east: float, north: float) -> float:
    return math.atan2(east, north)


@dataclass
class Leg:
    approach: float | None  # compass rad, None: any heading
    arrive_s: float  # flight time from the start of the route to the point
    drop_s: float  # drop time: the arrival, or the planned time when the layer is early
    heading: float = math.nan  # heading at the point (the approach; the last point: as flown in)


def approach_headings() -> np.ndarray:
    """The approach headings compared at each point (compass rad)."""
    return np.arange(APPROACH_CANDIDATES) * TWO_PI / APPROACH_CANDIDATES


def route(east: list[float], north: list[float], heading: float, speed_kt: float, bank_deg: float,
          planned_s: list[float | None] | None = None, prefer: int = -1, margin_m: float = 0.0,
          start_s: float = 0.0, capture_m: float = 0.0, first_s: np.ndarray | None = None) -> list[Leg]:
    """The route through points (east, north m from the start) in order: the approach heading,
    arrival and drop time (s) of each.

    The approach headings (APPROACH_CANDIDATES per point; the last point on any heading) are
    chosen together by dynamic programming over the points: least LATE_WEIGHT x (seconds late
    after the planned times) + the drop time of the last point. A layer early at a point waits
    (detours) for its planned time; the next leg starts at the point on the approach heading.
    The first leg is flown as the guidance flies it (path_to loose, capture_m): close to the point
    a small offset from the path is not a loop.

    first_s: the drop time of the first point per approach heading (approach_headings()) as
    the layer can really fly it. A timed leg cannot always wait: close to the point the paths
    on one approach heading arrive either at once or about a loop later. An early drop there
    costs as much as a late one."""
    n = len(east)
    first = first_s is not None and n > 1
    if n == 0:
        return []
    radius = turn_radius_m(speed_kt, bank_deg)
    v = max(speed_kt, 1.0) * KNOT_TO_MPS
    planned_s = planned_s or [None] * n
    headings = approach_headings()
    # per state (approach heading at the point): cost, drop time, arrival, parent
    cost = np.zeros(1)
    drop = np.full(1, float(start_s))
    pose = np.array([heading])  # heading at the previous point (one start pose)
    xs, ys = 0.0, 0.0
    parents, arrivals, drops = [], [], []
    for k in range(n):
        dx, dy = east[k] - xs, north[k] - ys
        last = k == n - 1
        if last:
            _, length, final = path_to(dx, dy, pose, None, radius, prefer, margin_m, loose=k == 0,
                                       capture_m=capture_m)
            final = np.where(np.isfinite(final), final, pose + math.pi)
            length = np.where(np.isfinite(length), length, math.hypot(dx, dy) + math.pi * radius)[:, None]
        else:
            _, length, _ = path_to(np.full((len(pose), len(headings)), dx), np.full((len(pose), len(headings)), dy),
                                   np.repeat(pose[:, None], len(headings), axis=1), headings[None, :], radius,
                                   prefer, margin_m, loose=k == 0, capture_m=capture_m)
        arrive = drop[:, None] + length / v
        if k == 0 and first:
            arrive = drop[:, None] + np.asarray(first_s, dtype=float)[None, :]
            late = np.abs(arrive - planned_s[0]) if planned_s[0] is not None else np.zeros_like(arrive)
            here = arrive
        elif planned_s[k] is not None:
            late, here = np.maximum(arrive - planned_s[k], 0.0), np.maximum(arrive, planned_s[k])
        else:
            late, here = np.zeros_like(arrive), arrive
        with np.errstate(invalid="ignore"):  # inf - inf after an excluded heading (first_s inf)
            total = cost[:, None] + LATE_WEIGHT * late + (here - drop[:, None])
        total = np.where(np.isnan(total), np.inf, total)
        best = np.argmin(total, axis=0)
        cols = np.arange(total.shape[1])
        cost, drop = total[best, cols], here[best, cols]
        parents.append(best)
        arrivals.append(arrive[best, cols])
        drops.append(drop)
        pose = headings if not last else pose
        xs, ys = east[k], north[k]
    # back along the parents
    state = int(np.argmin(cost))
    legs: list[Leg] = []
    for k in range(n - 1, -1, -1):
        approach = None if k == n - 1 else float(headings[state])
        heading_in = float(final[parents[k][state]]) % TWO_PI if approach is None else approach
        legs.append(Leg(approach, float(arrivals[k][state]), float(drops[k][state]), heading_in))
        state = int(parents[k][state])
    return legs[::-1]


def schedule(east: list[float], north: list[float], heading: float | None, speed_kt: float, bank_deg: float,
             wanted_s: list[float], start_s: float = 0.0,
             prefix: tuple[list[float], list[float], list[float | None]] | None = None) -> tuple[list[int], list[float]]:
    """Order and drop times for the drops of one plan: wanted_s are the optimal drop times
    (s from now; the layer is free at start_s at the origin on `heading`, None: unknown).

    Every order is tried (up to ORDER_LIMIT drops, else by wanted time): the one with the least
    total delay after the wanted times, then the earliest end. A drop is at its wanted time, or
    when the layer can be there after the previous drop (route along the order), whichever is
    later. Returns (order: indices into the inputs, drop times in that order)."""
    n = len(east)
    if n == 0:
        return [], []
    by_time = sorted(range(n), key=lambda i: wanted_s[i])
    orders = itertools.permutations(range(n)) if n <= ORDER_LIMIT else [by_time]
    best = None
    for order in orders:
        # unknown heading: the layer is assumed to be on its way towards the first point
        pe, pn, ps = prefix or ([], [], [])
        h = heading if heading is not None else _bearing(*((pe[0], pn[0]) if pe else (east[order[0]], north[order[0]])))
        legs = route([*pe, *(east[i] for i in order)], [*pn, *(north[i] for i in order)], h, speed_kt, bank_deg,
                     [*ps, *(wanted_s[i] for i in order)], start_s=start_s)
        drops = [leg.drop_s for leg in legs[len(pe):]]
        delay = sum(max(t - wanted_s[i], 0.0) for t, i in zip(drops, order, strict=True)) + sum(
            max(leg.drop_s - s, 0.0) for leg, s in zip(legs, ps) if s is not None)
        key = (round(delay, 1), drops[-1], list(order) != by_time[:len(order)])
        if best is None or key < best[0]:
            best = (key, list(order), drops)
    return best[1], best[2]
