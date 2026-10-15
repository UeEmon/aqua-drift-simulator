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
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

G = 9.80665
KNOT_TO_MPS = 0.5144444444444445
TWO_PI = 2.0 * math.pi
LOOP_TOLERANCE = math.radians(20.0)  # a turn-then-straight path this close to the approach heading instead of a loop
LINE_TOLERANCE_M = 5.0  # a point this close across the heading is ahead on the line (loose: capture / 2) ...
LINE_TOLERANCE = math.radians(2.0)  # ... with its approach this close to the heading
LOOSE_PENALTY_M = 1.0  # a loose path (not on the approach heading) only when strictly shorter
APPROACH_CANDIDATES = 36  # approach headings compared at each point (every 10 degrees) ...
# ... and the bearings from the previous point and to the next one (a line of close drops is
# crossed straight along it: drops a few hundred metres apart cannot take up a 5 degree offset
# from the 10 degree grid within a turn radius of ~4 km and would cost a loop each)
LATE_WEIGHT = 5.0  # a second late at a drop costs this many seconds of flight
ORDER_LIMIT = 5  # drops of one plan ordered by trying every order (more: by drop time)
# among routes as costly (the layer waits for a planned time on every approach): the approach
# to the first point kept (keep)
KEEP_TIE_S = 1e-3
# a moving point is met where it is when the layer gets there: fixed-point steps from the
# straight-line meeting (each cuts the error ~6x at 30 kt of wind) ...
MEET_STEPS = 2
# ... and a leg still this far off after them has no shortest path that meets the point (the
# path jumps from going round to turning straight in): it costs a turn round more
MEET_MISS_S = 5.0

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

    The shorter side is taken (prefer: the side taken when both are as long, or when the other
    is shorter by no more than margin_m). Returns (side +1 / -1, 0 where none is possible: the point is inside
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
    if not np.any(step):  # nothing flown yet
        zero = np.zeros(np.broadcast(h, *sides, *lengths).shape)
        return zero, zero, zero
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
            step_m: float = 0.0, loose: bool = False, capture_m: float = 0.0, segments: bool = False,
            before_m=0.0, hold_m=0.0, relax: bool = False):
    """Path to a point crossed on the approach heading (None or nan: any heading), wings level
    over the last before_m metres to it (on the approach heading) and straight on for the first
    hold_m metres from the craft (the straight flight after a drop that is still to be flown).

    With a straight part (before_m or hold_m > 0) a point straight ahead on the heading (within
    LINE_TOLERANCE_M across, loose: capture_m / 2; the approach within LINE_TOLERANCE of the heading) is flown to
    straight on: the straight parts of two drops on one line are one straight flight. The rest
    is the path below from the end of the hold to the start of the straight approach; loose
    (the guidance): close to the point, when that is a loop, a turn-then-straight path to the
    point with at least before_m (less capture_m / 2) of straight flight and within
    LOOP_TOLERANCE of the approach; relax (the guidance has gone round once already, a point
    moving across in the wind): when that is a loop too, the loose path below without the
    straight approach.

    Returns as below; the path (segments=True) is then the hold, the three parts below and the
    straight approach (five turn sides and radius-normalised lengths)."""
    heading = np.asarray(heading, dtype=float)
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    hold = np.asarray(hold_m, dtype=float)
    straight = np.any(hold > 0.0) or np.any(np.asarray(before_m, dtype=float) > 0.0)
    if not straight:
        return _path_core(east, north, heading, approach, radius, prefer, margin_m, step_m, loose, capture_m,
                          segments)
    he, hn = east - hold * np.sin(heading), north - hold * np.cos(heading)
    if approach is None:
        shape = np.broadcast(east, north, heading, hold).shape
        before, goal, free = np.zeros(shape), np.zeros(shape), np.ones(shape, dtype=bool)
    else:
        approach = np.asarray(approach, dtype=float)
        free = np.isnan(approach)
        goal = np.where(free, 0.0, approach)
        before = np.where(free, 0.0, before_m)
    te, tn = he - before * np.sin(goal), hn - before * np.cos(goal)
    _, length, final, _, _, sides, lengths = _path_core(te, tn, heading, approach, radius, prefer, margin_m, 0.0,
                                                        False, 0.0, True)  # exactly onto the straight approach
    zero = np.zeros_like(length)
    sides = [zero, *(s + zero for s in sides), zero]
    lengths = [hold / radius + zero, *(x + zero for x in lengths), before / radius + zero]
    length = length + hold + before
    if loose:  # close to the point: turn, then straight to it (wings level over before_m at least,
        # within LOOP_TOLERANCE of the approach) rather than a loop back to the straight approach
        _, direct, d_final, _, _, d_sides, d_lengths = _path_core(he, hn, heading, None, radius, prefer, margin_m,
                                                                  0.0, False, 0.0, True)
        off = np.abs(np.mod(d_final - goal + math.pi, TWO_PI) - math.pi)
        use = (np.isfinite(direct) & (d_lengths[1] * radius >= before - capture_m / 2.0) & (free | (off <= LOOP_TOLERANCE))
               & (direct + hold + math.pi * radius < length))
        direct = direct + LOOSE_PENALTY_M  # the exact approach first where it is as short
        sides = [sides[0], *(np.where(use, a, b) for a, b in zip(d_sides, sides[1:4], strict=True)),
                 np.where(use, 0, sides[4])]
        lengths = [lengths[0], *(np.where(use, a, b) for a, b in zip(d_lengths, lengths[1:4], strict=True)),
                   np.where(use, 0.0, lengths[4])]
        length = np.where(use, direct + hold, length)
        final = np.where(use, d_final, final)
    if loose and relax:  # still a loop after one more round (a point moving across in the wind):
        # the loose path without the straight approach rather than going round again
        _, near, n_final, _, _, n_sides, n_lengths = _path_core(he, hn, heading, approach, radius, prefer, margin_m,
                                                                0.0, True, capture_m, True)
        use = np.isfinite(near) & (near + hold + math.pi * radius < length)
        sides = [sides[0], *(np.where(use, a, b) for a, b in zip(n_sides, sides[1:4], strict=True)),
                 np.where(use, 0, sides[4])]
        lengths = [lengths[0], *(np.where(use, a, b) for a, b in zip(n_lengths, lengths[1:4], strict=True)),
                   np.where(use, 0.0, lengths[4])]
        length = np.where(use, near + hold + 2.0 * LOOSE_PENALTY_M, length)
        final = np.where(use, n_final, final)
    # straight on to a point ahead on the line
    along = east * np.sin(heading) + north * np.cos(heading)
    across = np.abs(east * np.cos(heading) - north * np.sin(heading))
    aligned = free | (np.abs(np.mod(goal - heading + math.pi, TWO_PI) - math.pi) <= LINE_TOLERANCE)
    lined = (along >= 0.0) & (across <= max(LINE_TOLERANCE_M, capture_m / 2.0 if loose else 0.0)) & aligned
    sides = [np.where(lined, 0, x) for x in sides]
    lengths = [np.where(lined, along / radius, lengths[0]), *(np.where(lined, 0.0, x) for x in lengths[1:])]
    length = np.where(lined, along, length)
    final = np.where(lined, heading, final)
    turn, step_east, step_north = motion(tuple(sides), tuple(lengths), heading, radius, step_m / radius)
    if segments:
        return turn, length, final, step_east, step_north, sides, lengths
    return turn, length, final


def _path_core(east, north, heading, approach, radius: float, prefer: int, margin_m: float, step_m: float,
               loose: bool, capture_m: float, segments: bool):
    """Path to a point crossed on the approach heading (None or nan: any heading).

    The shortest Dubins path, starting with either turn (prefer: the first turn taken when both
    are as short, or the other is shorter by no more than margin_m). Arrays or scalars.

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


def intercept_s(east, north, drift: tuple[float, float], speed_mps: float):
    """Time (s, arrays) to meet a point at (east, north) m moving at drift (m/s) flying straight at
    speed_mps: |d + u t| = v t."""
    east, north = np.asarray(east, dtype=float), np.asarray(north, dtype=float)
    a = drift[0] * drift[0] + drift[1] * drift[1] - speed_mps * speed_mps
    b = 2.0 * (east * drift[0] + north * drift[1])
    c = east * east + north * north
    if a >= -1e-9:
        return np.sqrt(c) / max(speed_mps, 1.0)
    return (b + np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))) / (-2.0 * a)


def _leg_to(east, north, xs, ys, pose, headings, drift, radius, options, t):
    """Length (m) of the shortest path from the previous points (xs, ys on pose) to the points
    (east, north at the start, moving at drift) crossed on headings, where they are at t."""
    with np.errstate(invalid="ignore"):  # inf * 0 after an excluded heading (an inf time)
        east, north = east + drift[0] * t - xs, north + drift[1] * t - ys
    return path_to(east, north, pose, headings, radius, **options)[1]


def _last_leg_to(east, north, xs, ys, pose, drift, radius, release, options, t):
    """_leg_to for the last point, on any heading: (length m, inf -> straight plus half a turn;
    heading at the point)."""
    with np.errstate(invalid="ignore"):  # inf * 0 after an excluded heading (an inf time)
        dx, dy = east + drift[0] * t - xs, north + drift[1] * t - ys
    if release is not None:
        de, dn = release(np.arctan2(dx, dy))
        dx, dy = dx + de, dy + dn
    _, length, final = path_to(dx, dy, pose, None, radius, **options)
    return np.where(np.isfinite(length), length, np.hypot(dx, dy) + math.pi * radius), final


def crab(tracks, speed_mps: float, wind_mps: tuple[float, float]):
    """The heading in the air (compass rad) that makes good the track over the ground (compass
    rad, arrays) at airspeed speed_mps in the wind (east, north m/s towards): the track plus the
    wind correction angle (偏流修正角)."""
    tracks = np.asarray(tracks, dtype=float)
    if not wind_mps[0] and not wind_mps[1]:
        return tracks
    ue, un = np.sin(tracks), np.cos(tracks)
    along = wind_mps[0] * ue + wind_mps[1] * un
    cross = -wind_mps[0] * un + wind_mps[1] * ue
    ground = np.maximum(along + np.sqrt(np.maximum(speed_mps * speed_mps - cross * cross, 0.0)), 1.0)
    return np.mod(np.arctan2(ground * ue - wind_mps[0], ground * un - wind_mps[1]), TWO_PI)


def release_table(east_m, north_m) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """route's release from a table of the release point's offset from the drop point (east,
    north m) over headings evenly spaced round the compass, the first repeated at the end
    (interpolated between)."""
    east, north = np.asarray(east_m, dtype=float), np.asarray(north_m, dtype=float)
    grid = np.arange(len(east)) * TWO_PI / max(len(east) - 1, 1)

    def offsets(headings):
        h = np.mod(np.asarray(headings, dtype=float), TWO_PI)
        return np.interp(h, grid, east), np.interp(h, grid, north)
    return offsets


@dataclass
class Leg:
    approach: float | None  # compass rad, None: any heading
    arrive_s: float  # flight time from the start of the route to the point
    drop_s: float  # drop time: the arrival, or the planned time when the layer is early
    heading: float = math.nan  # heading at the point (the approach; the last point: as flown in)
    track: float | None = None  # track over the ground at the point (the approach less the wind correction)


def approach_headings(east: list[float] | None = None, north: list[float] | None = None, k: int = 0) -> np.ndarray:
    """The approach headings compared at point k of the points (east, north m from the start;
    compass rad): APPROACH_CANDIDATES evenly spaced, then the bearing from the previous point
    (the start for the first) and the bearing to the next point. Without points: the even ones."""
    grid = np.arange(APPROACH_CANDIDATES) * TWO_PI / APPROACH_CANDIDATES
    if east is None or north is None:
        return grid
    px, py = (east[k - 1], north[k - 1]) if k > 0 else (0.0, 0.0)
    extra = []
    if math.hypot(east[k] - px, north[k] - py) > 0.0:
        extra.append(_bearing(east[k] - px, north[k] - py))
    if k + 1 < len(east) and math.hypot(east[k + 1] - east[k], north[k + 1] - north[k]) > 0.0:
        extra.append(_bearing(east[k + 1] - east[k], north[k + 1] - north[k]))
    return np.concatenate([grid, np.mod(extra, TWO_PI)])


def route(east: list[float], north: list[float], heading: float, speed_kt: float, bank_deg: float,
          planned_s: list[float | None] | None = None, prefer: int = -1, margin_m: float = 0.0,
          start_s: float = 0.0, capture_m: float = 0.0, first_s: np.ndarray | None = None,
          first_headings: np.ndarray | None = None, before_s: float = 0.0, after_s: float = 0.0,
          hold_s: float = 0.0,
          release: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
          keep: float | None = None, drift: tuple[float, float] = (0.0, 0.0),
          wind: tuple[float, float] = (0.0, 0.0)) -> list[Leg]:
    """The route through points (east, north m from the start) in order: the approach heading,
    arrival and drop time (s) of each.

    The approach headings (approach_headings per point; the last point on any heading) are
    chosen together by dynamic programming over the points: least LATE_WEIGHT x (seconds late
    after the planned times) + the drop time of the last point. A drop is at its planned time
    when the layer can wait for it on its orbit (it leaves when the shortest flight takes up the
    time, see layer.departure), else at the arrival; the path itself is never lengthened to keep a
    planned time. The next leg starts at the point on the approach heading.
    The first leg is flown as the guidance flies it (path_to loose, capture_m): close to the point
    a small offset from the path is not a loop.

    first_s: the flight time to the first point per approach heading (first_headings, default
    approach_headings(east, north, 0)) when the layer is already on its way to it: it is dropped
    on arrival, without waiting.

    Wings level at the drops (see path_to): straight on the approach heading for before_s before
    each point and for after_s after it (hold_s: what is left of that after the last drop, from
    the start). With before_s the last point is crossed on one of its approach headings too.

    release: the release point of a drop crossed on a heading lies this far (east, north m, per
    compass rad heading) from its point (the observer's fall: thrown forward along the heading,
    carried by the wind): every point is then flown to its release point for the approach
    heading of the state, so the headings are chosen with the release points they make. The
    point of a drop on any heading (the last one): the release point on the bearing to it.

    Routes as costly (waiting for planned times, any approach is in time) are told apart by the
    approach to the first point: keep (compass rad, the one being flown) if it is among them.

    In a wind the points are in the air mass (the frame the layer flies Dubins paths in) as they
    are at the start: they move at drift (east, north m/s; a point fixed over the ground moves
    with the wind's opposite, plus the current) and each is met where it is when the layer gets
    there (a few fixed-point steps on the arrival). The approaches compared are tracks over the
    ground (approach_headings: on the bearings between the points, a line is laid along it) and
    flown on the heading that makes them good in the wind (crab: the track plus the wind
    correction angle); Leg.approach is that heading, Leg.track the track. first_headings are
    headings in the air."""
    n = len(east)
    first = first_s is not None and n > 1
    if n == 0:
        return []
    radius = turn_radius_m(speed_kt, bank_deg)
    v = max(speed_kt, 1.0) * KNOT_TO_MPS
    planned_s = planned_s or [None] * n
    # the observer is released up to capture_m before the point
    straight = {"before_m": before_s * v + capture_m if before_s > 0.0 else 0.0}
    # per state (approach heading at the point): cost, drop time, arrival, parent
    cost = np.zeros(1)
    drop = np.full(1, float(start_s))
    pose = np.array([heading])  # heading at the previous point (one start pose)
    xs, ys = np.zeros(1), np.zeros(1)  # the previous point per state (its release point)
    moving = bool(drift[0] or drift[1])
    parents, arrivals, drops, candidates, tracks_at = [], [], [], [], []
    for k in range(n):
        last = k == n - 1 and before_s <= 0.0
        straight["hold_m"] = hold_s * v if k == 0 else after_s * v
        tracks = approach_headings(east, north, k)
        headings = crab(tracks, v, wind)
        if k == 0 and first_headings is not None:
            headings = np.asarray(first_headings, dtype=float)
            tracks = np.full(len(headings), np.nan)
        candidates.append(headings)
        tracks_at.append(tracks)
        start = drop[:, None] - start_s  # s from the start: where the points are
        if last:
            last_leg = partial(_last_leg_to, east[k], north[k], xs, ys, pose, drift, radius, release,
                               {"prefer": prefer, "margin_m": margin_m, "loose": k == 0, "capture_m": capture_m,
                                **straight})

            if moving:  # from the straight-line meeting (a path along a line closes on it at once)
                t0 = start[:, 0] + intercept_s(east[k] + drift[0] * start[:, 0] - xs,
                                               north[k] + drift[1] * start[:, 0] - ys, drift, v)
                when = t0
                for _ in range(MEET_STEPS):
                    when = start[:, 0] + last_leg(when)[0] / v
            else:
                when = start[:, 0]
            length, final = last_leg(when)
            final = np.where(np.isfinite(final), final, pose + math.pi)
            length = length[:, None]
            here_e, here_n = np.full(1, east[k]), np.full(1, north[k])  # not used further
        else:
            de, dn = release(headings) if release is not None else (np.zeros(len(headings)), np.zeros(len(headings)))
            base_e, base_n = east[k] + de, north[k] + dn  # the release point per approach heading, at the start
            leg = partial(_leg_to, base_e[None, :], base_n[None, :], xs[:, None], ys[:, None],
                          np.repeat(pose[:, None], len(headings), axis=1), headings[None, :], drift, radius,
                          {"prefer": prefer, "margin_m": margin_m, "loose": k == 0, "capture_m": capture_m, **straight})
            when = np.broadcast_to(start, (len(pose), len(headings)))
            if moving:  # from the straight-line meeting (a path along a line closes on it at once)
                when = when + intercept_s(base_e[None, :] + drift[0] * when - xs[:, None],
                                          base_n[None, :] + drift[1] * when - ys[:, None], drift, v)
                for _ in range(MEET_STEPS):
                    length = leg(when)
                    when = np.where(np.isfinite(length), start + length / v, when)
            length = leg(when)
            if moving:  # no shortest path meets it: a turn round more
                length = np.where(np.abs(start + length / v - when) > MEET_MISS_S, length + TWO_PI * radius, length)
        arrive = drop[:, None] + length / v
        if k == 0 and first:  # on its way: dropped on arrival
            arrive = drop[:, None] + np.asarray(first_s, dtype=float)[None, :]
            late = np.maximum(arrive - planned_s[0], 0.0) if planned_s[0] is not None else np.zeros_like(arrive)
            here = arrive
        elif planned_s[k] is not None:
            late, here = np.maximum(arrive - planned_s[k], 0.0), np.maximum(arrive, planned_s[k])
        else:
            late, here = np.zeros_like(arrive), arrive
        with np.errstate(invalid="ignore"):  # inf - inf after an excluded heading (first_s inf)
            total = cost[:, None] + LATE_WEIGHT * late + (here - drop[:, None])
            if k == 0 and keep is not None:
                other = np.abs(np.mod(headings - keep + math.pi, TWO_PI) - math.pi) > 1e-9
                total = total + KEEP_TIE_S * other[None, :]
        total = np.where(np.isnan(total), np.inf, total)
        best = np.argmin(total, axis=0)
        cols = np.arange(total.shape[1])
        cost, drop = total[best, cols], here[best, cols]
        parents.append(best)
        arrivals.append(arrive[best, cols])
        drops.append(drop)
        pose = headings if not last else pose
        if not last:  # the release point where it is at the drop
            here_e = base_e + drift[0] * np.where(np.isfinite(drop), drop - start_s, 0.0)
            here_n = base_n + drift[1] * np.where(np.isfinite(drop), drop - start_s, 0.0)
        xs, ys = np.broadcast_to(here_e, cost.shape).copy(), np.broadcast_to(here_n, cost.shape).copy()
    # back along the parents
    state = int(np.argmin(cost))
    legs: list[Leg] = []
    for k in range(n - 1, -1, -1):
        approach = None if k == n - 1 and before_s <= 0.0 else float(candidates[k][state]) % TWO_PI
        heading_in = float(final[parents[k][state]]) % TWO_PI if approach is None else approach
        track = None if approach is None or math.isnan(tracks_at[k][state]) else float(tracks_at[k][state]) % TWO_PI
        legs.append(Leg(approach, float(arrivals[k][state]), float(drops[k][state]), heading_in, track))
        state = int(parents[k][state])
    return legs[::-1]


def schedule(east: list[float], north: list[float], heading: float | None, speed_kt: float, bank_deg: float,
             wanted_s: list[float], start_s: float = 0.0,
             prefix: tuple[list[float], list[float], list[float | None]] | None = None,
             orders: list[list[int]] | None = None, before_s: float = 0.0,
             after_s: float = 0.0,
             release: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
             drift: tuple[float, float] = (0.0, 0.0),
             wind: tuple[float, float] = (0.0, 0.0)) -> tuple[list[int], list[float]]:
    """Order and drop times for the drops of one plan: wanted_s are the optimal drop times
    (s from now; the layer is free at start_s at the origin on `heading`, None: unknown).

    Every order is tried (up to ORDER_LIMIT drops, else by wanted time): the one with the least
    total delay after the wanted times, then the earliest end. A drop is at its wanted time, or
    when the layer can be there after the previous drop (route along the order), whichever is
    later. `orders`: only these orders (e.g. along a line, either way). before_s, after_s: wings
    level at the drops (see route). release, drift, wind: the observer's fall and the wind, as
    the layer flies (see route). Returns (order: indices into the inputs, drop times in that
    order)."""
    n = len(east)
    if n == 0:
        return [], []
    by_time = sorted(range(n), key=lambda i: wanted_s[i])
    if orders is None:
        orders = itertools.permutations(range(n)) if n <= ORDER_LIMIT else [by_time]
    best = None
    for order in orders:
        # unknown heading: the layer is assumed to be on its way towards the first point
        pe, pn, ps = prefix or ([], [], [])
        h = heading if heading is not None else _bearing(*((pe[0], pn[0]) if pe else (east[order[0]], north[order[0]])))
        legs = route([*pe, *(east[i] for i in order)], [*pn, *(north[i] for i in order)], h, speed_kt, bank_deg,
                     [*ps, *(wanted_s[i] for i in order)], start_s=start_s, before_s=before_s, after_s=after_s,
                     release=release, drift=drift, wind=wind)
        drops = [leg.drop_s for leg in legs[len(pe):]]
        delay = sum(max(t - wanted_s[i], 0.0) for t, i in zip(drops, order, strict=True)) + sum(
            max(leg.drop_s - s, 0.0) for leg, s in zip(legs, ps) if s is not None)
        key = (round(delay, 1), drops[-1], list(order) != by_time[:len(order)])
        if best is None or key < best[0]:
            best = (key, list(order), drops)
    return best[1], best[2]
