from __future__ import annotations

import math
import random

from aqua_drift.models import (
    CurrentFieldConfig,
    DopplerObservation,
    DopplerTruth,
    ObserverState,
    Position,
    ScenarioConfig,
    TargetState,
    Velocity,
)

EARTH_RADIUS_M = 6_371_008.8
KNOT_TO_MPS = 0.5144444444444445
FT_TO_M = 0.3048
YD_TO_M = 0.9144
M_TO_YD = 1.0 / YD_TO_M
NM_TO_M = 1852.0


def clamp_rate(current: float, desired: float, max_change: float) -> float:
    delta = desired - current
    return current + max(-max_change, min(max_change, delta))


def clamp_heading(current: float, desired: float, max_change: float) -> float:
    delta = (desired - current + 180.0) % 360.0 - 180.0
    return (current + max(-max_change, min(max_change, delta))) % 360.0


def heading_velocity(speed_kt: float, heading_deg: float, vertical_fps: float = 0.0) -> Velocity:
    radians = math.radians(heading_deg)
    return Velocity(
        east_kt=speed_kt * math.sin(radians),
        north_kt=speed_kt * math.cos(radians),
        vertical_fps=vertical_fps,
    )


def speed_and_course(velocity: Velocity) -> tuple[float, float]:
    speed = math.hypot(velocity.east_kt, velocity.north_kt)
    if speed < 1e-12:
        return 0.0, 0.0
    course = math.degrees(math.atan2(velocity.east_kt, velocity.north_kt)) % 360.0
    return speed, course


def add_velocity(first: Velocity, second: Velocity) -> Velocity:
    return Velocity(
        east_kt=first.east_kt + second.east_kt,
        north_kt=first.north_kt + second.north_kt,
        vertical_fps=first.vertical_fps + second.vertical_fps,
    )


def move_position(position: Position, velocity: Velocity, seconds: float) -> Position:
    north_m = velocity.north_kt * KNOT_TO_MPS * seconds
    east_m = velocity.east_kt * KNOT_TO_MPS * seconds
    latitude_rad = math.radians(position.latitude)
    dlat = north_m / EARTH_RADIUS_M
    longitude_scale = max(math.cos(latitude_rad), 1e-8)
    dlon = east_m / (EARTH_RADIUS_M * longitude_scale)
    depth = max(0.0, position.depth_ft + velocity.vertical_fps * seconds)
    return Position(
        latitude=position.latitude + math.degrees(dlat),
        longitude=((position.longitude + math.degrees(dlon) + 180.0) % 360.0) - 180.0,
        depth_ft=depth,
    )


def local_offset_m(origin: Position, other: Position) -> tuple[float, float, float]:
    mean_lat = math.radians((origin.latitude + other.latitude) / 2.0)
    north = math.radians(other.latitude - origin.latitude) * EARTH_RADIUS_M
    east = math.radians(other.longitude - origin.longitude) * EARTH_RADIUS_M * math.cos(mean_lat)
    down = (other.depth_ft - origin.depth_ft) * FT_TO_M
    return east, north, down


def current_at(field: CurrentFieldConfig, position: Position) -> Velocity:
    east_m, north_m, down_m = local_offset_m(field.reference_position, position)
    vector_nm = [east_m / NM_TO_M, north_m / NM_TO_M, down_m / NM_TO_M]
    gradient = field.gradient_per_nm
    delta = [sum(gradient[row][col] * vector_nm[col] for col in range(3)) for row in range(3)]
    return Velocity(
        east_kt=field.base_velocity.east_kt + delta[0],
        north_kt=field.base_velocity.north_kt + delta[1],
        vertical_fps=field.base_velocity.vertical_fps + delta[2],
    )


def initial_target(config: ScenarioConfig) -> TargetState:
    through_water = heading_velocity(
        config.target.initial_through_water_speed_kt, config.target.initial_hdg_deg
    )
    current = current_at(config.current_field, config.target.initial_position)
    ground = add_velocity(through_water, current)
    ground_speed, cog = speed_and_course(ground)
    return TargetState(
        tick=0,
        position=config.target.initial_position,
        hdg_deg=config.target.initial_hdg_deg,
        cog_deg=cog,
        through_water_speed_kt=config.target.initial_through_water_speed_kt,
        ground_speed_kt=ground_speed,
        through_water_velocity=through_water,
        ground_velocity=ground,
    )


def advance_target(
    config: ScenarioConfig,
    state: TargetState,
    seconds: float,
    current_velocity: Velocity | None = None,
) -> TargetState:
    target = config.target
    heading = clamp_heading(state.hdg_deg, target.desired_hdg_deg, target.hdg_rate_deg_per_sec * seconds)
    speed = clamp_rate(
        state.through_water_speed_kt,
        target.desired_through_water_speed_kt,
        target.speed_rate_kt_per_sec * seconds,
    )
    depth = clamp_rate(
        state.position.depth_ft,
        target.desired_depth_ft,
        target.depth_rate_ft_per_sec * seconds,
    )
    vertical = (depth - state.position.depth_ft) / seconds if seconds else 0.0
    through_water = heading_velocity(speed, heading, vertical)
    current = current_velocity or current_at(config.current_field, state.position)
    ground = add_velocity(through_water, current)
    position = move_position(state.position, ground, seconds)
    ground_speed, cog = speed_and_course(ground)
    return TargetState(
        tick=state.tick + int(seconds),
        position=position,
        hdg_deg=heading,
        cog_deg=cog,
        through_water_speed_kt=speed,
        ground_speed_kt=ground_speed,
        through_water_velocity=through_water,
        ground_velocity=ground,
    )


def advance_observer(
    config: ScenarioConfig,
    state: ObserverState,
    seconds: float,
    current_velocity: Velocity | None = None,
) -> ObserverState:
    current = current_velocity or current_at(config.current_field, state.position)
    return ObserverState(
        observer_id=state.observer_id,
        tick=state.tick + int(seconds),
        position=move_position(state.position, current, seconds),
        ground_velocity=current,
        status=state.status,
    )


def doppler_observation(
    config: ScenarioConfig,
    target: TargetState,
    observer: ObserverState,
    rng: random.Random | None = None,
) -> tuple[DopplerObservation, DopplerTruth]:
    """Synthesize one error-free Doppler sample (or a non-detection), a noisy horizontal
    bearing every `bearing.interval_s` while detected, and the truth record."""
    east_m, north_m, down_m = local_offset_m(observer.position, target.position)
    slant_m = math.sqrt(east_m**2 + north_m**2 + down_m**2)
    rel_e = (target.ground_velocity.east_kt - observer.ground_velocity.east_kt) * KNOT_TO_MPS
    rel_n = (target.ground_velocity.north_kt - observer.ground_velocity.north_kt) * KNOT_TO_MPS
    rel_d = (target.ground_velocity.vertical_fps - observer.ground_velocity.vertical_fps) * FT_TO_M
    relative_speed_mps = math.sqrt(rel_e**2 + rel_n**2 + rel_d**2)
    if slant_m < 1e-9:
        radial_away_mps = 0.0
    else:
        radial_away_mps = (east_m * rel_e + north_m * rel_n + down_m * rel_d) / slant_m
    source = config.source.source_frequency_hz
    observed = source * (1.0 - radial_away_mps / config.source.sound_speed_mps)
    recognized = source + config.source.shared_recognition_bias_hz
    slant_yd = slant_m * M_TO_YD
    detected = slant_yd <= config.max_slant_range_yd
    true_bearing = math.degrees(math.atan2(east_m, north_m)) % 360.0
    bearing = None
    b = config.bearing
    if b.enabled and detected and target.tick % b.interval_s == 0:
        noise = (rng or random).gauss(0.0, b.sigma_deg)
        bearing = (true_bearing + noise) % 360.0
    observation = DopplerObservation(
        observer_id=observer.observer_id,
        tick=target.tick,
        observer_position=observer.position,
        detected=detected,
        observed_frequency_hz=observed if detected else None,
        recognized_frequency_hz=recognized,
        bearing_deg=bearing,
    )
    truth = DopplerTruth(
        observer_id=observer.observer_id,
        tick=target.tick,
        slant_range_yd=slant_yd,
        relative_speed_kt=relative_speed_mps / KNOT_TO_MPS,
        relative_radial_speed_kt=-radial_away_mps / KNOT_TO_MPS,
        true_bearing_deg=true_bearing,
    )
    return observation, truth
