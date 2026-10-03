"""Default observer placement patterns (used when an observer container has no explicit
position). Index 0, 1, 2, ... alternate around the pattern centre."""
from __future__ import annotations

import math

from aqua_drift.models import Position, ScenarioConfig, Velocity
from aqua_drift.physics import YD_TO_M, move_position

KNOT_TO_MPS = 0.5144444444444445


def _offset(origin: Position, east_m: float, north_m: float, depth_ft: float) -> Position:
    # move_position works with knots over seconds; use 1 s at the equivalent speed
    moved = move_position(
        origin,
        Velocity(east_kt=east_m / KNOT_TO_MPS, north_kt=north_m / KNOT_TO_MPS, vertical_fps=0.0),
        1.0,
    )
    return Position(latitude=moved.latitude, longitude=moved.longitude, depth_ft=depth_ft)


def default_position(config: ScenarioConfig, index: int, angle_hint: float = 0.0) -> Position:
    d = config.deployment
    origin = config.target.initial_position
    hdg = math.radians(config.target.desired_hdg_deg)
    ahead = d.offset_ahead_yd * YD_TO_M
    centre_e, centre_n = ahead * math.sin(hdg), ahead * math.cos(hdg)
    spacing = d.spacing_yd * YD_TO_M
    if d.pattern == "grid":
        # staggered 2-row field across the expected track: not collinear, so the
        # Doppler-only mirror ambiguity about a single line of observers is broken
        column = index // 2
        row = index % 2
        step = (column + 1) // 2 * (1 if column % 2 else -1)
        bearing = math.radians(d.line_bearing_deg)
        along = step * spacing + (0.5 * spacing if row else 0.0)
        across = (row - 0.5) * spacing
        east = centre_e + along * math.sin(bearing) + across * math.cos(bearing)
        north = centre_n + along * math.cos(bearing) - across * math.sin(bearing)
    elif d.pattern == "line":
        step = (index + 1) // 2 * (1 if index % 2 else -1)
        bearing = math.radians(d.line_bearing_deg)
        east = centre_e + step * spacing * math.sin(bearing)
        north = centre_n + step * spacing * math.cos(bearing)
    elif d.pattern == "ring":
        angle = index * 2.399963229728653  # golden angle
        radius = spacing * (1 + index // 6)
        east = centre_e + radius * math.sin(angle)
        north = centre_n + radius * math.cos(angle)
    else:
        radius = spacing * 2.0
        east = centre_e + radius * math.sin(angle_hint)
        north = centre_n + radius * math.cos(angle_hint)
    depth = d.depth_ft + (index % 3) * d.depth_step_ft
    return _offset(origin, east, north, depth)
