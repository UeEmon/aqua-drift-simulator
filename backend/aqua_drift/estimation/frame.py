"""Local tangent-plane frame used by the estimator.

Coordinates are (east_m, north_m, down_m); down is positive with depth, so a depth in feet
maps directly to the third axis. The frame is anchored at the first observer fix and is
accurate to well below a yard over the tens of nautical miles a scenario covers.
"""
from __future__ import annotations

import math

import numpy as np

from aqua_drift.models import Position

EARTH_RADIUS_M = 6_371_008.8
KNOT_TO_MPS = 0.5144444444444445
FT_TO_M = 0.3048
YD_TO_M = 0.9144
NM_TO_M = 1852.0


class LocalFrame:
    def __init__(self, origin: Position) -> None:
        self.lat0 = origin.latitude
        self.lon0 = origin.longitude
        self.cos_lat0 = max(math.cos(math.radians(self.lat0)), 1e-8)

    def to_local(self, position: Position) -> np.ndarray:
        north = math.radians(position.latitude - self.lat0) * EARTH_RADIUS_M
        east = math.radians(position.longitude - self.lon0) * EARTH_RADIUS_M * self.cos_lat0
        return np.array([east, north, position.depth_ft * FT_TO_M])

    def to_geo(self, east: float, north: float, down: float) -> Position:
        lat = self.lat0 + math.degrees(north / EARTH_RADIUS_M)
        lon = self.lon0 + math.degrees(east / (EARTH_RADIUS_M * self.cos_lat0))
        lon = ((lon + 180.0) % 360.0) - 180.0
        return Position(latitude=lat, longitude=lon, depth_ft=max(0.0, down / FT_TO_M))

    def to_lonlat_array(self, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lat = self.lat0 + np.degrees(north / EARTH_RADIUS_M)
        lon = self.lon0 + np.degrees(east / (EARTH_RADIUS_M * self.cos_lat0))
        return ((lon + 180.0) % 360.0) - 180.0, lat
