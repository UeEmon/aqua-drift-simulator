from __future__ import annotations

import cmath
import math
import random
from collections import deque

from aqua_drift.models import (
    CurrentFieldConfig,
    DopplerObservation,
    DopplerTruth,
    ObserverState,
    Position,
    ScenarioConfig,
    TargetState,
    TonalObservation,
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
        session=state.session,
    )


def lloyd_mirror_level_db(
    config: ScenarioConfig,
    target_position: Position,
    observer_position: Position,
    frequency_hz: float,
) -> tuple[float, float]:
    """Received level [dB] of the tonal via the direct and the surface-reflected path.

    Image-source model (isovelocity, pressure-release surface: reflection coefficient -mu):
        p = exp(i k r1) / r1 - mu exp(i k r2) / r2
        r1 = sqrt(R^2 + (zs - zr)^2), r2 = sqrt(R^2 + (zs + zr)^2)
        mu = exp(-2 (k sigma_h sin g)^2)  coherent reflection of a rough surface (Rayleigh),
             sin g = (zs + zr) / r2 the grazing angle at the reflection point
    The true path difference can be scaled (path_difference_error_pct) to emulate a sound-speed
    structure that the estimator's isovelocity model does not know. Returns (level, mu)."""
    lloyd = config.lloyd
    east_m, north_m, _ = local_offset_m(observer_position, target_position)
    horizontal = math.hypot(east_m, north_m)
    zs = target_position.depth_ft * FT_TO_M
    zr = observer_position.depth_ft * FT_TO_M
    r1 = math.hypot(horizontal, zs - zr)
    r2 = math.hypot(horizontal, zs + zr)
    k = 2.0 * math.pi * frequency_hz / config.source.sound_speed_mps
    sin_grazing = (zs + zr) / max(r2, 1e-6)
    mu = math.exp(-2.0 * (k * lloyd.wave_height_rms_m * sin_grazing) ** 2)
    r2_true = r1 + (r2 - r1) * (1.0 + lloyd.path_difference_error_pct / 100.0)
    pressure = cmath.exp(1j * k * r1) / max(r1, 1.0) - mu * cmath.exp(1j * k * r2_true) / max(r2_true, 1.0)
    return lloyd.source_level_db + 20.0 * math.log10(max(abs(pressure), 1e-12)), mu


class LevelNoise:
    """Received-level fluctuation: AR(1) per observer, sigma level_noise_db, correlation time
    noise_correlation_s (independent between observers)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.state: dict[str, float] = {}

    def sample(self, observer_id: str, sigma_db: float, correlation_s: float) -> float:
        if sigma_db <= 0:
            return 0.0
        a = math.exp(-1.0 / correlation_s) if correlation_s > 0 else 0.0
        previous = self.state.get(observer_id)
        if previous is None:
            value = self.rng.gauss(0.0, sigma_db)
        else:
            value = a * previous + math.sqrt(1.0 - a * a) * self.rng.gauss(0.0, sigma_db)
        self.state[observer_id] = value
        return value


class SourceSignal:
    """What the target has emitted recently: its states (one per tick) and, per tonal, the
    emitted-frequency fluctuation (first-order Gauss-Markov with stability_hz and
    stability_correlation_s, one value per tick). With the propagation delay an observer
    receives at t what the target emitted at t_e = t - r(t_e) / c."""

    HISTORY_S = 180

    def __init__(self, seed: int = 13) -> None:
        self.rng = random.Random(seed)
        self.states: deque[TargetState] = deque()
        self.drift: dict[int, list[float]] = {}  # tick -> fluctuation per tonal [Hz]

    def add(self, config: ScenarioConfig, target: TargetState) -> None:
        if self.states and target.tick <= self.states[-1].tick:
            return
        tonals = config.source.tonals()
        previous = self.drift.get(self.states[-1].tick) if self.states else None
        steps = target.tick - self.states[-1].tick if self.states else 1
        values = []
        for index, tonal in enumerate(tonals):
            sigma = tonal.stability_hz
            if sigma <= 0:
                values.append(0.0)
                continue
            if previous is None or index >= len(previous):
                values.append(self.rng.gauss(0.0, sigma))
                continue
            a = math.exp(-steps / tonal.stability_correlation_s)
            values.append(a * previous[index] + math.sqrt(1.0 - a * a) * self.rng.gauss(0.0, sigma))
        self.states.append(target)
        self.drift[target.tick] = values
        while self.states and target.tick - self.states[0].tick > self.HISTORY_S:
            self.drift.pop(self.states.popleft().tick, None)

    def _bracket(self, t: float) -> tuple[TargetState, TargetState, float]:
        states = self.states
        if t <= states[0].tick:
            return states[0], states[0], t - states[0].tick
        if t >= states[-1].tick:
            return states[-1], states[-1], t - states[-1].tick
        index = min(int(t - states[0].tick), len(states) - 1)
        while index > 0 and states[index].tick > t:
            index -= 1
        while index + 1 < len(states) and states[index + 1].tick <= t:
            index += 1
        first, second = states[index], states[index + 1]
        return first, second, (t - first.tick) / (second.tick - first.tick)

    def at(self, origin: Position, t: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Target offset from `origin` (east, north, down m) and ground velocity (m/s) at time t
        (linear between ticks; extrapolated with the velocity outside the history)."""
        first, second, f = self._bracket(t)
        v0, v1 = _velocity_mps(first.ground_velocity), _velocity_mps(second.ground_velocity)
        o0 = local_offset_m(origin, first.position)
        if first is second:  # f = seconds beyond the history
            return tuple(o + v * f for o, v in zip(o0, v0, strict=True)), v0
        o1 = local_offset_m(origin, second.position)
        offset = tuple(a + (b - a) * f for a, b in zip(o0, o1, strict=True))
        velocity = tuple(a + (b - a) * f for a, b in zip(v0, v1, strict=True))
        return offset, velocity

    def fluctuation(self, t: float, index: int) -> float:
        first, second, f = self._bracket(t)
        a = self.drift.get(first.tick, [])
        b = self.drift.get(second.tick, [])
        va = a[index] if index < len(a) else 0.0
        vb = b[index] if index < len(b) else 0.0
        return va if first is second else va + (vb - va) * f


def _velocity_mps(velocity: Velocity) -> tuple[float, float, float]:
    return (velocity.east_kt * KNOT_TO_MPS, velocity.north_kt * KNOT_TO_MPS, velocity.vertical_fps * FT_TO_M)


def doppler_observation(
    config: ScenarioConfig,
    target: TargetState,
    observer: ObserverState,
    rng: random.Random | None = None,
    level_noise: LevelNoise | None = None,
    signal: SourceSignal | None = None,
) -> tuple[DopplerObservation, DopplerTruth]:
    """Synthesize one Doppler sample per tonal (or a non-detection), a noisy horizontal
    bearing every `bearing.interval_s` while detected, and the truth record.

    With `signal` and source.propagation_delay the observer hears what the target emitted at
    t_e = t - r/c (position, velocity and frequency fluctuation at t_e), so a change of course
    or speed reaches each observer after its own delay. Without a signal history the geometry
    is instantaneous. Each tonal's line centre is measured with a normal error of
    bandwidth / sqrt(12)."""
    source = config.source
    c = source.sound_speed_mps
    delay = 0.0
    if signal is not None and signal.states and source.propagation_delay:
        for _ in range(4):
            offset, velocity = signal.at(observer.position, target.tick - delay)
            delay = math.sqrt(sum(v * v for v in offset)) / c
        offset, velocity = signal.at(observer.position, target.tick - delay)
    else:
        offset = local_offset_m(observer.position, target.position)
        velocity = _velocity_mps(target.ground_velocity)
    east_m, north_m, down_m = offset
    slant_m = math.sqrt(east_m**2 + north_m**2 + down_m**2)
    observer_v = _velocity_mps(observer.ground_velocity)
    rel_e, rel_n, rel_d = (a - b for a, b in zip(velocity, observer_v, strict=True))
    relative_speed_mps = math.sqrt(rel_e**2 + rel_n**2 + rel_d**2)
    if slant_m < 1e-9:
        radial_away_mps = 0.0
    else:
        radial_away_mps = (east_m * rel_e + north_m * rel_n + down_m * rel_d) / slant_m
    slant_yd = slant_m * M_TO_YD
    detected = slant_yd <= config.max_slant_range_yd
    scale = 1.0 + source.shared_recognition_bias_hz / source.source_frequency_hz
    tonals = []
    for index, tonal in enumerate(source.tonals()):
        observed = None
        if detected:
            emitted = tonal.frequency_hz
            if signal is not None and tonal.stability_hz > 0:
                emitted += signal.fluctuation(target.tick - delay, index)
            observed = emitted * (1.0 - radial_away_mps / c)
            if tonal.bandwidth_hz > 0:
                observed += (rng or random).gauss(0.0, tonal.bandwidth_hz / math.sqrt(12.0))
        tonals.append(TonalObservation(
            recognized_frequency_hz=tonal.frequency_hz * scale,
            observed_frequency_hz=observed,
            bandwidth_hz=tonal.bandwidth_hz,
        ))
    true_bearing = math.degrees(math.atan2(east_m, north_m)) % 360.0
    bearing = None
    b = config.bearing
    if b.enabled and detected and target.tick % b.interval_s == 0:
        noise = (rng or random).gauss(0.0, b.sigma_deg)
        bearing = (true_bearing + noise) % 360.0
    level = None
    if config.lloyd.enabled and detected:
        level, _ = lloyd_mirror_level_db(config, target.position, observer.position, tonals[0].observed_frequency_hz)
        if level_noise is not None:
            level += level_noise.sample(
                observer.observer_id, config.lloyd.level_noise_db, config.lloyd.noise_correlation_s
            )
    observation = DopplerObservation(
        observer_id=observer.observer_id,
        tick=target.tick,
        observer_position=observer.position,
        detected=detected,
        observed_frequency_hz=tonals[0].observed_frequency_hz,
        recognized_frequency_hz=tonals[0].recognized_frequency_hz,
        bearing_deg=bearing,
        tonals=tonals,
        received_level_db=level,
    )
    truth = DopplerTruth(
        observer_id=observer.observer_id,
        tick=target.tick,
        slant_range_yd=slant_yd,
        relative_speed_kt=relative_speed_mps / KNOT_TO_MPS,
        relative_radial_speed_kt=-radial_away_mps / KNOT_TO_MPS,
        true_bearing_deg=true_bearing,
        propagation_delay_s=delay,
    )
    return observation, truth
