import itertools
import math
import random

import numpy as np
import pytest

from aqua_drift.deployment import _offset
from aqua_drift.layer import advance, initial_state
from aqua_drift.models import DropTask, LayerConfig, LayerFeed, Position
from aqua_drift.physics import local_offset_m
from aqua_drift.route import (
    WORDS,
    dubins_words,
    motion,
    path_to,
    release_table,
    route,
    schedule,
    turn_radius_m,
)

DATUM = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)


def _fly(heading: float, radius: float, segments) -> tuple[float, float, float]:
    """End pose after (side, angle or normalised length) segments from the origin."""
    x = y = 0.0
    for side, amount in segments:
        if side == 0:
            x, y = x + amount * radius * math.sin(heading), y + amount * radius * math.cos(heading)
        else:
            cx, cy = x + side * radius * math.cos(heading), y - side * radius * math.sin(heading)
            heading += side * amount
            x, y = cx - side * radius * math.cos(heading), cy + side * radius * math.sin(heading)
    return x, y, heading


def test_dubins_paths_end_on_the_point_and_the_approach_heading() -> None:
    rng = np.random.default_rng(0)
    radius, checked = 1000.0, 0
    for _ in range(300):
        east, north = rng.uniform(-5000, 5000, 2)
        heading, approach = rng.uniform(0, 2 * math.pi, 2)
        for word, parts in dubins_words(east, north, heading, approach, radius).items():
            if not np.isfinite(parts[0]):
                continue
            x, y, h = _fly(heading, radius, [(s, float(p)) for s, p in zip(WORDS[word], parts, strict=True)])
            assert math.hypot(x - east, y - north) < 1e-3
            assert abs((h - approach + math.pi) % (2 * math.pi) - math.pi) < 1e-6
            checked += 1
    assert checked > 1000
    # the shortest is chosen: straight ahead on the same heading is a straight line
    _, length, final = path_to(0.0, 3000.0, 0.0, 0.0, radius)
    assert abs(float(length) - 3000.0) < 1e-6 and abs(float(final)) < 1e-9


def test_route_lines_up_for_the_next_drop_instead_of_a_loop() -> None:
    """Two drops 2 km apart abeam (well inside a turning diameter of ~8 km): crossing the first
    on the heading towards the second makes the second a straight 2 km, not a loop."""
    speed, bank = 200.0, 15.0
    v = speed * 0.5144444444444445
    radius = turn_radius_m(speed, bank)
    legs = route([0.0, 2000.0], [10000.0, 10000.0], 0.0, speed, bank)
    assert legs[0].approach is not None and legs[1].approach is None
    assert abs(math.degrees(legs[0].approach) - 90.0) <= 15.0  # lined up eastwards for the second
    assert legs[1].arrive_s - legs[0].arrive_s < (2000.0 + 0.2 * radius) / v
    # free heading at the first (the old guidance): arriving northbound, the second is a loop away
    _, free_first, _ = path_to(0.0, 10000.0, 0.0, None, radius)
    _, onward, _ = path_to(2000.0, 0.0, 0.0, None, radius)
    assert float(onward) > 2000.0 + math.pi * radius
    assert legs[1].arrive_s < 0.85 * (float(free_first) + float(onward)) / v


def test_schedule_orders_the_drops_and_keeps_times_the_layer_can_meet() -> None:
    speed, bank = 200.0, 15.0
    # three drops on a line east-west, wanted at the same time: flown along the line, not back
    east, north = [4000.0, -4000.0, 0.0], [20000.0, 20000.0, 20000.0]
    order, times = schedule(east, north, math.radians(90.0), speed, bank, [0.0, 0.0, 0.0])
    assert order in ([1, 2, 0], [0, 2, 1])
    assert all(b > a for a, b in itertools.pairwise(times))
    # the times follow the route: each drop no earlier than the layer can be there
    legs = route([east[i] for i in order], [north[i] for i in order], math.radians(90.0), speed, bank,
                 [0.0] * 3)
    assert [round(t, 6) for t in times] == [round(leg.drop_s, 6) for leg in legs]
    # wanted times far apart: kept as they are
    order, times = schedule(east, north, None, speed, bank, [600.0, 900.0, 1200.0])
    assert order == [0, 1, 2] and times == [600.0, 900.0, 1200.0]


def _lay(points: list[Position], seed: int = 3) -> tuple[dict[int, int], float, list[float]]:
    """Fly the layer through drops (as soon as possible). Returns (drop tick by task, total
    heading change in degrees, the route's predicted drop times at the start)."""
    config = LayerConfig()
    rng = random.Random(seed)
    state = initial_state(config, DATUM, 0, rng)
    tasks = [DropTask(task_id=k + 1, created_tick=0, source="forward", position=p, status="APPROVED")
             for k, p in enumerate(points)]
    offsets = [local_offset_m(state.position, p) for p in points]
    predicted = [leg.drop_s for leg in route([o[0] for o in offsets], [o[1] for o in offsets],
                                             math.radians(state.heading_deg), state.speed_kt, config.max_bank_deg)]
    done: dict[int, int] = {}
    turned, worst = 0.0, 0.0
    for tick in range(1, 1500):
        open_tasks = [t for t in tasks if t.task_id not in done]
        if not open_tasks:
            break
        before = state.heading_deg
        state, update = advance(LayerFeed(tick=tick, config=config, tasks=open_tasks, datum=DATUM), state, rng)
        turned += abs((state.heading_deg - before + 180.0) % 360.0 - 180.0)
        worst = max(worst, abs(state.bank_deg))
        for task_id in update.completed:
            done[task_id] = tick
    assert worst <= config.max_bank_deg + 1e-6
    return done, turned, predicted


def test_layer_lays_close_drops_in_one_pass_as_the_route_predicts() -> None:
    """A pair of drops 3 km apart either side of the target track ahead: the layer turns
    in once, lays one and flies straight on to the other (no loop after the first drop), at the
    times the route predicts."""
    a = _offset(DATUM, -1500.0, 12000.0, 300.0)
    b = _offset(DATUM, 1500.0, 12000.0, 300.0)
    done, turned, predicted = _lay([a, b])
    assert set(done) == {1, 2}
    assert done[2] - done[1] < 3000.0 / (200.0 * 0.5144444444444445) + 15  # straight on
    assert turned < 360.0  # never a full circle
    for task_id, time in zip((1, 2), predicted, strict=True):
        assert abs(done[task_id] - time) <= 10.0


def test_schedule_routes_new_drops_on_after_the_queued_ones() -> None:
    speed, bank = 200.0, 15.0
    prefix = ([0.0], [8000.0], [None])
    east, north = [2000.0], [8000.0]
    order, times = schedule(east, north, 0.0, speed, bank, [0.0], prefix=prefix)
    legs = route([0.0, 2000.0], [8000.0, 8000.0], 0.0, speed, bank, [None, 0.0])
    assert order == [0] and abs(times[0] - legs[1].drop_s) < 1e-6
    # lined up at the queued drop for the new one: about a straight 2 km after it
    assert times[0] - legs[0].drop_s < (2000.0 + 0.2 * turn_radius_m(speed, bank)) / (speed * 0.5144444444444445)


def test_route_crosses_a_line_of_close_drops_straight_along_it() -> None:
    """Drops 450 m apart on a line whose bearing is off the 10 degree grid (43 deg): the line is
    flown straight through (the bearing to the next drop is an approach heading), not a loop
    at every drop (a 3 degree offset cannot be taken up within 450 m at a ~4 km turn radius)."""
    speed, bank = 200.0, 15.0
    v = speed * 0.5144444444444445
    bearing = math.radians(43.0)
    east = [8000.0 * math.sin(bearing) + 450.0 * k * math.sin(bearing) for k in range(6)]
    north = [8000.0 * math.cos(bearing) + 450.0 * k * math.cos(bearing) for k in range(6)]
    legs = route(east, north, bearing, speed, bank)
    assert all(abs(leg.approach - bearing) < 1e-9 for leg in legs[:-1])
    assert legs[-1].arrive_s - legs[0].arrive_s < 5 * 450.0 / v + 1.0


def test_layer_lays_a_line_of_close_drops_in_one_pass() -> None:
    """Four drops 450 m apart on a line at 43 deg ahead of the layer: one pass, no loop."""
    bearing = math.radians(43.0)
    points = [_offset(DATUM, (6000.0 + 450.0 * k) * math.sin(bearing) - 3000.0,
                      (6000.0 + 450.0 * k) * math.cos(bearing) + 6000.0, 300.0) for k in range(4)]
    done, _, predicted = _lay(points)
    assert set(done) == {1, 2, 3, 4}
    assert done[4] - done[1] < 3 * 450.0 / (200.0 * 0.5144444444444445) + 10
    for task_id, time in zip((1, 2, 3, 4), predicted, strict=True):
        assert abs(done[task_id] - time) <= 10.0


def _lay_with(points: list[Position], config: LayerConfig) -> dict[int, int]:
    rng = random.Random(3)
    state = initial_state(config, DATUM, 0, rng)
    tasks = [DropTask(task_id=k + 1, created_tick=0, source="forward", position=p, status="APPROVED")
             for k, p in enumerate(points)]
    done: dict[int, int] = {}
    for tick in range(1, 1500):
        open_tasks = [t for t in tasks if t.task_id not in done]
        if not open_tasks:
            break
        state, update = advance(LayerFeed(tick=tick, config=config, tasks=open_tasks, datum=DATUM), state, rng)
        for task_id in update.completed:
            done[task_id] = tick
    return done


def test_one_drop_releases_the_observers_at_one_point_up_to_the_maximum() -> None:
    """Three observers at one drop point (different depths) and one elsewhere: one release lays
    those at the point (no second pass over it), up to max_per_release; the others there wait for
    another pass."""
    stacked = [_offset(DATUM, 0.0, 12000.0, depth) for depth in (60.0, 500.0, 1500.0)]
    other = _offset(DATUM, 9000.0, 20000.0, 300.0)
    done = _lay_with([*stacked, other], LayerConfig())
    assert set(done) == {1, 2, 3, 4}
    assert done[1] == done[2] == done[3] < done[4]  # one release at the point
    done = _lay_with(stacked, LayerConfig(max_per_release=2))
    assert done[1] == done[2] < done[3]  # the third one needs another pass
    done = _lay_with(stacked, LayerConfig(max_per_release=1))
    assert len(set(done.values())) == 3


def test_path_to_flies_wings_level_before_and_after_a_drop() -> None:
    """The path ends with before_m straight on the approach heading and starts with the hold
    (straight on after the last drop); a point straight ahead on the line is flown to straight."""
    radius = turn_radius_m(200.0, 15.0)
    _, length, final, _, _, sides, lengths = path_to(5000.0, 8000.0, 0.0, math.pi / 2, radius, before_m=500.0,
                                                     hold_m=300.0, segments=True)
    assert int(sides[0]) == 0 and float(lengths[0]) * radius == pytest.approx(300.0)
    assert int(sides[-1]) == 0 and float(lengths[-1]) * radius == pytest.approx(500.0)
    _, x, y = motion(tuple(sides), tuple(lengths), 0.0, radius, float(length) / radius)
    assert math.hypot(float(x) - 5000.0, float(y) - 8000.0) < 1e-3 and float(final) == pytest.approx(math.pi / 2)
    _, along, _ = path_to(0.0, 450.0, 0.0, 0.0, radius, before_m=500.0, hold_m=500.0)
    assert float(along) == pytest.approx(450.0)


def test_layer_drops_wings_level() -> None:
    """Wings level (no bank) for straight_before_s before every drop and straight_after_s after
    it, also on a line of close drops (one straight flight) and at a drop off to the side."""
    config = LayerConfig(straight_before_s=5.0, straight_after_s=5.0)
    bearing = math.radians(43.0)
    points = [_offset(DATUM, (6000.0 + 450.0 * k) * math.sin(bearing) - 3000.0,
                      (6000.0 + 450.0 * k) * math.cos(bearing) + 6000.0, 300.0) for k in range(3)]
    points.append(_offset(DATUM, 9000.0, -2000.0, 300.0))
    rng = random.Random(3)
    state = initial_state(config, DATUM, 0, rng)
    tasks = [DropTask(task_id=k + 1, created_tick=0, source="forward", position=p, status="APPROVED")
             for k, p in enumerate(points)]
    done: dict[int, int] = {}
    banks: dict[int, float] = {}
    for tick in range(1, 1500):
        open_tasks = [t for t in tasks if t.task_id not in done]
        if not open_tasks:
            break
        state, update = advance(LayerFeed(tick=tick, config=config, tasks=open_tasks, datum=DATUM), state, rng)
        banks[tick] = state.bank_deg
        for task_id in update.completed:
            done[task_id] = tick
    assert set(done) == {1, 2, 3, 4}
    assert done[3] - done[1] < 2 * 450.0 / (200.0 * 0.5144444444444445) + 5  # the line in one pass
    for tick in done.values():
        assert all(abs(banks.get(t, 0.0)) < 1e-6 for t in range(tick - 4, min(tick + 5, max(banks) + 1)))


def test_schedule_routes_with_the_observers_fall_and_the_wind() -> None:
    """The planner's drop times are the layer's: route with the same release points and wind."""
    speed, bank = 200.0, 15.0
    table = release_table([150.0 * math.sin(math.radians(5 * i)) for i in range(73)],
                          [150.0 * math.cos(math.radians(5 * i)) for i in range(73)])
    de, dn = table(np.array([math.radians(42.5)]))
    assert abs(de[0] - 150.0 * math.sin(math.radians(42.5))) < 1.0
    assert abs(dn[0] - 150.0 * math.cos(math.radians(42.5))) < 1.0
    east, north = [0.0, 3000.0], [9000.0, 9000.0]
    wind, drift = (-15.0, 10.0), (15.0, -10.0)
    _, times = schedule(east, north, 0.0, speed, bank, [0.0, 0.0], orders=[[0, 1]], before_s=2.0, after_s=5.0,
                        release=table, drift=drift, wind=wind)
    legs = route(east, north, 0.0, speed, bank, [0.0, 0.0], before_s=2.0, after_s=5.0, release=table,
                 drift=drift, wind=wind)
    assert times == [leg.drop_s for leg in legs]
    _, still = schedule(east, north, 0.0, speed, bank, [0.0, 0.0], orders=[[0, 1]], before_s=2.0, after_s=5.0)
    assert still != times


def test_schedule_starts_with_the_leg_being_flown() -> None:
    """The layer is on its way to a queued drop (51 s ahead): a line planned after it is routed on
    from that leg as it is flown. Re-planned from the layer's position the first leg came out a
    loop for a heading a fraction of a degree off, and the line ~3 min late (closed loop S0)."""
    from aqua_drift.route import schedule

    east = [4159.0, 4388.0, 5074.0, 5302.0, 5645.0, 5874.0, 6103.0]
    north = [-3396.0] * 7
    prefix = ([3662.6], [-3264.4], [51.0])
    common = {"start_s": 0.0, "prefix": prefix, "before_s": 2.0, "after_s": 5.0,
              "orders": [list(range(7)), list(range(7))[::-1]]}
    _, free = schedule(east, north, math.radians(169.205), 200.0, 15.0, [280.0] * 7, **common)
    _, flown = schedule(east, north, math.radians(169.205), 200.0, 15.0, [280.0] * 7, **common,
                        first_s=np.array([51.0]), first_headings=np.array([math.radians(105.0)]))
    assert min(free) > 400.0  # the first leg re-planned as a loop
    assert min(flown) < 360.0  # on from the drop the layer is flying to


def test_schedule_flies_every_leg_as_the_guidance_does() -> None:
    """A drop after a queued line, behind its end and off its heading: the exact path of that leg
    is a loop, but the guidance (loose path within the capture radius) gets there ~4 min sooner --
    the layer did, and the drop timed on the exact path was laid behind the target (closed loop S0)."""
    from aqua_drift.route import schedule

    prefix = ([4213.6, 4442.1, 4670.7], [-2658.8, -2658.3, -2657.9], [49.0, 52.0, 54.0])
    common = {"start_s": 0.0, "prefix": prefix, "before_s": 2.0, "after_s": 5.0, "orders": [[0]],
              "first_s": np.array([52.7]), "first_headings": np.array([math.radians(90.0)])}
    _, exact = schedule([4220.0], [-2893.0], math.radians(160.3), 200.0, 15.0, [317.0], **common)
    _, guided = schedule([4220.0], [-2893.0], math.radians(160.3), 200.0, 15.0, [317.0], **common,
                         capture_m=150 * 0.9144, loose_all=True)
    assert exact[0] > 500.0
    assert guided[0] < 360.0
