"""Wind (風向風速) and the free fall of the observers laid by the layer (設標者).

* Profile: direction (from) and speed every 1,000 ft from the sea surface to 30,000 ft
  (WindConfig), interpolated linearly in the east / north components; above the top level the
  top value holds.
* Free fall: the observer leaves the layer with the layer's ground velocity and falls to the sea
  surface under gravity and quadratic air drag on its velocity relative to the local wind,
  a = g z - k |v - w(h)| (v - w(h)), k = g / v_t^2 (v_t: terminal velocity). It drifts with
  the wind of every altitude it falls through.
* Mean wind (投下高度から海面までの平均風向風速): the uniform wind that, in the same fall from
  the same release (altitude, ground velocity), moves the entry point from the predicted
  no-wind entry point to the actual one. First guess: the offset divided by the wind
  sensitivity of the fall (about the fall time less the drag lag); then Newton steps.
* Release correction: the next observer is released so that, in the estimated mean wind, it
  enters the water at its drop point: release = drop point - fall displacement(estimated wind).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aqua_drift.deployment import _offset
from aqua_drift.models import DropRelease, WindConfig, WindEstimate
from aqua_drift.physics import local_offset_m

G = 9.80665
FT_TO_M = 0.3048
KNOT_TO_MPS = 0.5144444444444445
M_TO_YD = 1.0 / 0.9144
FALL_DT_S = 0.1

WindFunction = Callable[[float], tuple[float, float]]  # altitude ft -> (east, north) m/s towards


def wind_vector_kt(direction_deg: float, speed_kt: float) -> tuple[float, float]:
    """(east, north) of the wind blowing FROM direction_deg."""
    rad = math.radians(direction_deg)
    return -speed_kt * math.sin(rad), -speed_kt * math.cos(rad)


def wind_from(east: float, north: float) -> tuple[float, float]:
    """(direction it blows FROM in degrees, speed) of a wind vector (towards)."""
    speed = math.hypot(east, north)
    if speed < 1e-9:
        return 0.0, 0.0
    return math.degrees(math.atan2(-east, -north)) % 360.0, speed


class WindProfile:
    """Wind truth as a function of altitude (ft) -> (east, north) m/s."""

    def __init__(self, config: WindConfig) -> None:
        levels = config.levels
        self.enabled = config.enabled
        self.altitudes = np.array([level.altitude_ft for level in levels], dtype=float)
        vectors = [wind_vector_kt(level.direction_deg, level.speed_kt) for level in levels]
        self.east = np.array([v[0] for v in vectors]) * KNOT_TO_MPS
        self.north = np.array([v[1] for v in vectors]) * KNOT_TO_MPS
        self.terminal_mps = config.terminal_velocity_fps * FT_TO_M

    def at(self, altitude_ft: float) -> tuple[float, float]:
        if not self.enabled:
            return 0.0, 0.0
        return (float(np.interp(altitude_ft, self.altitudes, self.east)),
                float(np.interp(altitude_ft, self.altitudes, self.north)))

    def mean(self, altitude_ft: float) -> tuple[float, float]:
        """Vector mean of the wind from the sea surface to altitude_ft (m/s, towards)."""
        if altitude_ft <= 0:
            return self.at(0.0)
        heights = np.linspace(0.0, altitude_ft, 201)
        winds = np.array([self.at(h) for h in heights])
        return float(winds[:, 0].mean()), float(winds[:, 1].mean())


@dataclass
class Fall:
    east_m: float  # entry point - release point
    north_m: float
    time_s: float


def uniform(east_mps: float, north_mps: float) -> WindFunction:
    return lambda _altitude: (east_mps, north_mps)


def fall(altitude_ft: float, east_mps: float, north_mps: float, wind: WindFunction | None,
         terminal_mps: float, dt: float = FALL_DT_S) -> Fall:
    """Free fall from altitude_ft with initial horizontal velocity (east, north) m/s through
    the wind (None: still air) to the sea surface."""
    k = G / (terminal_mps * terminal_mps)
    height = max(altitude_ft, 0.0) * FT_TO_M
    ve, vn, vd = east_mps, north_mps, 0.0  # vd: downwards
    x = y = t = 0.0
    while height > 0.0:
        we, wn = wind(height / FT_TO_M) if wind is not None else (0.0, 0.0)
        re, rn = ve - we, vn - wn
        speed = math.sqrt(re * re + rn * rn + vd * vd)
        ve -= k * speed * re * dt
        vn -= k * speed * rn * dt
        vd += (G - k * speed * vd) * dt
        step = dt
        if vd * dt >= height:  # the sea surface is reached within this step
            step = height / vd
        x += ve * step
        y += vn * step
        height -= vd * step
        t += step
    return Fall(x, y, t)


@dataclass
class MeanWind:
    east_mps: float  # towards
    north_mps: float
    fall_time_s: float
    no_wind: Fall  # predicted fall without wind


def estimate_mean_wind(altitude_ft: float, east_mps: float, north_mps: float,
                       actual_east_m: float, actual_north_m: float, terminal_mps: float,
                       iterations: int = 8) -> MeanWind:
    """Uniform (mean) wind from the release altitude to the sea surface that explains the actual
    entry point (relative to the release point) of a fall with the given initial velocity."""
    still = fall(altitude_ft, east_mps, north_mps, None, terminal_mps)
    # sensitivity of the entry point to a uniform wind (m per m/s, about the fall time less
    # the drag lag); the fall is nearly linear in the wind, Newton steps remove the rest
    probe = fall(altitude_ft, east_mps, north_mps, uniform(1.0, 0.0), terminal_mps)
    gain = max(probe.east_m - still.east_m, 1e-3)
    we = (actual_east_m - still.east_m) / gain
    wn = (actual_north_m - still.north_m) / gain
    for _ in range(iterations):
        sim = fall(altitude_ft, east_mps, north_mps, uniform(we, wn), terminal_mps)
        de, dn = actual_east_m - sim.east_m, actual_north_m - sim.north_m
        if math.hypot(de, dn) < 0.01:
            break
        we += de / gain
        wn += dn / gain
    final = fall(altitude_ft, east_mps, north_mps, uniform(we, wn), terminal_mps)
    return MeanWind(we, wn, final.time_s, still)


def ground_velocity(course_rad: float, airspeed_mps: float, wind_east: float,
                    wind_north: float) -> tuple[float, float]:
    """Ground velocity (east, north) flying the course course_rad at airspeed in the wind:
    the heading is corrected for the cross wind (crab)."""
    ue, un = math.sin(course_rad), math.cos(course_rad)
    along = wind_east * ue + wind_north * un
    cross = -wind_east * un + wind_north * ue
    speed = along + math.sqrt(max(airspeed_mps * airspeed_mps - cross * cross, 0.0))
    speed = max(speed, 1.0)
    return speed * ue, speed * un


def release_offset(altitude_ft: float, east_mps: float, north_mps: float,
                   wind_east: float, wind_north: float, terminal_mps: float) -> Fall:
    """Displacement from the release point to the entry point predicted with a uniform wind
    (the estimated mean wind; 0 = no correction for the wind)."""
    wind = uniform(wind_east, wind_north) if (wind_east or wind_north) else None
    return fall(altitude_ft, east_mps, north_mps, wind, terminal_mps)


def yd(metres: float) -> float:
    return metres * M_TO_YD


def wind_estimate(drop: DropRelease, profile: WindProfile) -> WindEstimate:
    """Mean wind from the drop altitude to the sea surface of one drop, once the observer is in
    the water (its entry point is known exactly), with the profile's vector mean as the truth
    for comparison."""
    release = drop.release_position
    east, north, _ = local_offset_m(release, drop.splash_position)
    mean = estimate_mean_wind(drop.altitude_ft, drop.ground_east_kt * KNOT_TO_MPS,
                              drop.ground_north_kt * KNOT_TO_MPS, east, north, profile.terminal_mps)
    still = mean.no_wind
    no_wind = _offset(release, still.east_m, still.north_m, release.depth_ft)
    east_kt, north_kt = mean.east_mps / KNOT_TO_MPS, mean.north_mps / KNOT_TO_MPS
    direction, speed = wind_from(east_kt, north_kt)
    true_e, true_n = profile.mean(drop.altitude_ft)
    true_direction, true_speed = wind_from(true_e / KNOT_TO_MPS, true_n / KNOT_TO_MPS)
    miss_e, miss_n, _ = local_offset_m(drop.planned_position, drop.splash_position)
    return WindEstimate(
        task_id=drop.task_id, tick=drop.splash_tick, altitude_ft=round(drop.altitude_ft, 1),
        fall_time_s=round(mean.fall_time_s, 1), release_position=release, no_wind_position=no_wind,
        splash_position=drop.splash_position,
        offset_yd=round(yd(math.hypot(east - still.east_m, north - still.north_m)), 1),
        east_kt=east_kt, north_kt=north_kt, direction_deg=round(direction, 1), speed_kt=round(speed, 2),
        miss_yd=round(yd(math.hypot(miss_e, miss_n)), 1),
        true_direction_deg=round(true_direction, 1), true_speed_kt=round(true_speed, 2),
    )
