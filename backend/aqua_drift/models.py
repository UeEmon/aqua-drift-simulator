"""Domain models shared by every AQUA-DRIFT container.

Truth (target state, true source frequency, true current field, true slant range) and
observations (Doppler frequency, detection flag, observer position/depth/time) are kept in
separate models so that the estimator can be fed observations only.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ObserverStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    EXPIRED = "EXPIRED"
    EVICTED = "EVICTED"
    ARCHIVED = "ARCHIVED"


class EstimateMode(StrEnum):
    ONLINE = "ONLINE"  # past track is NOT updated (filter output frozen when produced)
    SMOOTHED = "SMOOTHED"  # past track IS updated within the recomputation window


class Position(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    depth_ft: float = Field(ge=0)


class Velocity(BaseModel):
    east_kt: float = 0.0
    north_kt: float = 0.0
    vertical_fps: float = 0.0  # positive = deeper


# --------------------------------------------------------------------------- configuration


class TargetMotionConfig(BaseModel):
    """Target truth. Base motion is constant HDG / constant through-water speed."""

    initial_position: Position = Position(latitude=35.0, longitude=140.0, depth_ft=500.0)
    initial_hdg_deg: float = Field(default=90.0, ge=0, lt=360)
    initial_through_water_speed_kt: float = Field(default=8.0, ge=0)
    desired_hdg_deg: float = Field(default=90.0, ge=0, lt=360)
    desired_through_water_speed_kt: float = Field(default=8.0, ge=0)
    desired_depth_ft: float = Field(default=500.0, ge=0)
    hdg_rate_deg_per_sec: float = Field(default=1.0, gt=0)
    speed_rate_kt_per_sec: float = Field(default=0.1, gt=0)
    depth_rate_ft_per_sec: float = Field(default=2.0, gt=0)


class CurrentFieldConfig(BaseModel):
    """Truth current field: v(p) = base + G (p - p_ref), p in NM (east, north, down)."""

    reference_position: Position = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)
    base_velocity: Velocity = Velocity(east_kt=1.0, north_kt=0.3, vertical_fps=0.0)
    gradient_per_nm: list[list[float]] = Field(
        default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    @field_validator("gradient_per_nm")
    @classmethod
    def validate_gradient(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("gradient_per_nm must be a 3x3 matrix")
        return value


class Tonal(BaseModel):
    """One line (tonal) of the target's acoustic signature.

    * bandwidth_hz: width of the spectral line. The observer measures the line centre with a
      normal error of 1 sigma = bandwidth / sqrt(12) (peak of a line of that width), independent
      between observers and seconds. 0 = error-free.
    * stability_hz: 1 sigma of the emitted-frequency fluctuation (first-order Gauss-Markov with
      stability_correlation_s). It is emitted, so every observer receives the same fluctuation
      (shifted by its propagation delay). 0 = constant frequency."""

    frequency_hz: float = Field(gt=0)
    bandwidth_hz: float = Field(default=0.0, ge=0)
    stability_hz: float = Field(default=0.0, ge=0)
    stability_correlation_s: float = Field(default=60.0, gt=0)


class SourceFrequencyConfig(BaseModel):
    """Source truth. The primary tonal is source_frequency_hz (with bandwidth_hz /
    stability_hz); additional_tonals are further lines of the same source. The recognition bias
    is common to every observer and does not change with time; it is given on the primary tonal
    and applies to every tonal in proportion to its frequency (the source rate is misjudged).
    With propagation_delay the sound reaching an observer at t left the target at t - r/c."""

    source_frequency_hz: float = Field(default=400.0, gt=0)
    bandwidth_hz: float = Field(default=0.0, ge=0)
    stability_hz: float = Field(default=0.0, ge=0)
    stability_correlation_s: float = Field(default=60.0, gt=0)
    additional_tonals: list[Tonal] = Field(default_factory=list, max_length=15)
    shared_recognition_bias_hz: float = 0.0
    sound_speed_mps: float = Field(default=1500.0, gt=0)
    propagation_delay: bool = True
    random_seed: int = 13  # frequency fluctuation (stability)

    def tonals(self) -> list[Tonal]:
        """Every tonal, the primary first."""
        primary = Tonal(
            frequency_hz=self.source_frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            stability_hz=self.stability_hz,
            stability_correlation_s=self.stability_correlation_s,
        )
        return [primary, *self.additional_tonals]


class BearingConfig(BaseModel):
    """Horizontal true bearing observation (observer -> target), inside the max slant range.

    Errors: normal, zero mean, sigma_deg, independent in time and between observers."""

    enabled: bool = True
    sigma_deg: float = Field(default=15.0, gt=0, le=90)
    interval_s: int = Field(default=15, ge=1, le=600)
    random_seed: int = 11


class LloydMirrorConfig(BaseModel):
    """Surface-reflection (Lloyd's mirror) interference on the received level of the tonal.

    Optional because the depth fit is computationally heavy: only when `enabled` does the
    acoustic container simulate the received level and does the estimator fit the target depth
    to the interference pattern. Truth-side parameters (the estimator does not see them):

    * level_noise_db / noise_correlation_s : level fluctuation (AR(1), dB, seconds)
    * wave_height_rms_m : sea surface roughness -> coherent reflection exp(-2 (k s sin g)^2)
    * path_difference_error_pct : sound-speed structure not represented by the isovelocity
      image-source model (scales the true path difference; 0 = isovelocity water)
    """

    enabled: bool = False
    level_noise_db: float = Field(default=2.0, ge=0, le=20)
    noise_correlation_s: float = Field(default=20.0, ge=0, le=600)
    wave_height_rms_m: float = Field(default=0.3, ge=0, le=5)
    path_difference_error_pct: float = Field(default=0.0, ge=-50, le=50)
    source_level_db: float = 140.0
    random_seed: int = 23


WIND_TOP_FT = 30000
WIND_STEP_FT = 1000


class WindLevel(BaseModel):
    """Wind at one altitude: direction it blows FROM (true, degrees) and speed."""

    altitude_ft: int = Field(ge=0, le=WIND_TOP_FT)
    direction_deg: float = Field(ge=0, le=360)
    speed_kt: float = Field(ge=0, le=300)

    @field_validator("altitude_ft")
    @classmethod
    def validate_altitude(cls, value: int) -> int:
        if value % WIND_STEP_FT:
            raise ValueError(f"altitude_ft must be a multiple of {WIND_STEP_FT} ft")
        return value


def default_wind_levels() -> list[WindLevel]:
    """Default profile: 10 kt from 250 deg at the sea surface, veering and strengthening with
    height to 70 kt from 290 deg at 30,000 ft."""
    count = WIND_TOP_FT // WIND_STEP_FT
    return [
        WindLevel(altitude_ft=k * WIND_STEP_FT, direction_deg=round(250.0 + 40.0 * k / count, 1),
                  speed_kt=10.0 + 2.0 * k)
        for k in range(count + 1)
    ]


class WindConfig(BaseModel):
    """Wind truth (風向風速) from the sea surface to 30,000 ft, set every 1,000 ft (linear in
    between, the top level above it). It acts on the layer (設標者) at its flight altitude and
    on every observer falling from the drop altitude to the sea surface.

    The observer falls freely: gravity and quadratic air drag on its velocity relative to the
    air (k = g / terminal_velocity^2), starting with the layer's ground velocity."""

    enabled: bool = True  # False: no wind at any altitude (the observers still fall)
    levels: list[WindLevel] = Field(default_factory=default_wind_levels)
    terminal_velocity_fps: float = Field(default=100.0, ge=5, le=1000)

    @field_validator("levels")
    @classmethod
    def validate_levels(cls, value: list[WindLevel]) -> list[WindLevel]:
        if not value:
            raise ValueError("at least one wind level is required")
        altitudes = [level.altitude_ft for level in value]
        if len(set(altitudes)) != len(altitudes):
            raise ValueError("wind levels must have different altitudes")
        return sorted(value, key=lambda level: level.altitude_ft)


class LayerConfig(BaseModel):
    """設標者 (layer): the craft that lays additional observers. It moves over the sea surface
    at speed_kt +- speed_spread_kt (a new speed for every leg) with bank <= max_bank_deg and,
    without a task, circles the estimated target position. An additional observer is in the
    water only when the layer reaches its drop point. Planned drops are proposed to the
    operator; approval is automatic (default) or manual."""

    enabled: bool = True  # False: additional observers appear at once (no layer)
    approval: str = Field(default="auto", pattern="^(auto|manual)$")
    speed_kt: float = Field(default=200.0, gt=0, le=600)
    speed_spread_kt: float = Field(default=50.0, ge=0, le=300)
    max_bank_deg: float = Field(default=15.0, gt=0, le=60)
    orbit_radius_yd: float = Field(default=5000.0, gt=0)  # raised to the turn radius if smaller
    capture_radius_yd: float = Field(default=150.0, gt=0)
    proposal_timeout_s: int = Field(default=600, ge=10, le=7200)  # unanswered proposals expire
    preferred_turn: str = Field(default="left", pattern="^(left|right)$")  # standard turn direction
    turn_margin_s: float = Field(default=10.0, ge=0, le=600)  # other side only if this much quicker
    paused: bool = False  # operator hold: the layer circles the target and does not leave
    random_seed: int = 31
    # 3-D flight: the layer cruises (orbit) at cruise_altitude_ft, descends to drop_altitude_ft
    # on the way to a drop point, and climbs / descends at climb_rate_fpm
    cruise_altitude_ft: float = Field(default=3000.0, ge=100, le=WIND_TOP_FT)
    drop_altitude_ft: float = Field(default=1000.0, ge=100, le=WIND_TOP_FT)
    climb_rate_fpm: float = Field(default=2000.0, gt=0, le=20000)
    # release the observer upwind by the drift of the estimated mean wind (from the last drop)
    wind_correction: bool = True


class ForwardDeploymentConfig(BaseModel):
    """Automatic deployment of observers ahead (前程) of the ESTIMATED target position.

    Uses only the estimate (position, HDG, through-water speed, status), never the truth.
    Observers and target drift with the same water, so the geometry is planned in the water
    frame: the target's predicted relative position after lead_time_s is
    estimate + through-water velocity x lead_time_s. When fewer than min_coverage observers
    (active or pending) lie within coverage_fraction x R_max of that point, observers_per_drop
    observers are placed ahead_distance_yd ahead on the estimated heading, lateral_offset_yd to
    either side, using standby observer containers."""

    enabled: bool = True
    lead_time_s: int = Field(default=600, ge=30, le=7200)
    coverage_fraction: float = Field(default=0.8, gt=0, le=1.0)
    min_coverage: int = Field(default=2, ge=1, le=10)
    observers_per_drop: int = Field(default=2, ge=1, le=6)
    ahead_distance_yd: float = Field(default=4000.0, gt=0)
    lateral_offset_yd: float = Field(default=2000.0, ge=0)
    cooldown_s: int = Field(default=120, ge=0, le=3600)
    min_speed_kt: float = Field(default=0.5, ge=0)
    max_sigma_fraction: float = Field(default=0.5, gt=0)  # skip if 1σ major > fraction x R_max
    # "optimal": positions, number and depths chosen by the Doppler-tracking information
    # (aqua_drift.optimal_deployment); "fixed": the two-sided pattern above
    strategy: str = Field(default="optimal", pattern="^(optimal|fixed)$")
    horizon_s: int = Field(default=1800, ge=120, le=7200)
    max_per_drop: int = Field(default=4, ge=1, le=8)
    min_relative_gain: float = Field(default=0.10, ge=0, le=1)  # stop adding below this gain
    trigger_gain: float = Field(default=0.30, ge=0, le=1)  # deploy without a coverage gap above this
    target_error_yd: float = Field(default=25.0, gt=0)  # stop adding once the predicted error is below
    depth_options_ft: list[float] = Field(default_factory=lambda: [60.0, 200.0, 500.0, 1000.0, 1500.0])
    depth_weight: float = Field(default=1.0, ge=0, le=10)  # depth error weight in the criterion
    optimal_max_sigma_fraction: float = Field(default=0.15, gt=0)  # placement needs a converged track
    schedule_drops: bool = True  # plan the optimal drop time (else: as soon as possible)
    drop_lead_s: int = Field(default=60, ge=0, le=1800)  # in the water this long before detection starts
    # maneuver-robust planning (optimal strategy): weighted course / speed / depth change
    # hypotheses, process noise in the information recursion, risk (CVaR) and coverage terms
    maneuver_weight: float = Field(default=0.4, ge=0, le=0.9)  # total weight of the maneuver hypotheses
    maneuver_time_s: int = Field(default=600, ge=0, le=7200)  # hypothetical maneuver starts this far ahead
    maneuver_depth_change_ft: float = Field(default=500.0, ge=0, le=1500)
    process_velocity_sigma_kt: float = Field(default=0.5, ge=0, le=20)  # velocity drift per 10 min
    process_depth_rate_sigma_fps: float = Field(default=0.2, ge=0, le=20)  # depth-rate drift per 10 min
    risk_weight: float = Field(default=0.5, ge=0, le=5)  # weight of the worst-case (CVaR) term
    risk_quantile: float = Field(default=0.2, gt=0, le=1)  # share of hypotheses in the worst case
    coverage_loss_yd: float = Field(default=300.0, ge=0)  # error equivalent of a step with < min_coverage
    use_detection_gate: bool = True  # value the detection start / end (slant range = R_max)
    # replanning (aqua_drift.replanning): open automatic drops are replaced when the estimate has
    # moved away from what they were planned on and a fresh plan is clearly better; drops that are
    # being flown to, due within replan_freeze_s or revised replan_max_revisions times are kept
    replan_enabled: bool = True
    replan_min_interval_s: int = Field(default=180, ge=0, le=3600)  # between replacements (a new maneuver: once at once)
    replan_freeze_s: int = Field(default=240, ge=0, le=3600)  # drops due this soon are kept
    replan_shift_sigma: float = Field(default=2.0, ge=0)  # estimate shift trigger, x the 1 sigma major
    replan_shift_fraction: float = Field(default=0.15, ge=0, le=1)  # ... and at least this x R_max
    replan_heading_deg: float = Field(default=20.0, ge=0, le=180)  # heading change trigger (>= 2 sigma)
    replan_speed_kt: float = Field(default=2.0, ge=0)  # speed change trigger (>= 2 sigma)
    replan_min_improvement: float = Field(default=0.2, ge=0, le=1)  # replace only if the cost drops this much
    replan_max_revisions: int = Field(default=2, ge=0, le=10)  # a drop replaced this often is kept
    # a kept approved drop (flown to, due soon...) that no longer helps detection is proposed to the
    # operator for cancellation (the layer keeps flying it until the operator cancels): when the
    # motion hypotheses that bring the target within R_max (+2 sigma) of it after its drop weigh less
    # than this (the proposal is withdrawn again above 2.5 x this)
    cancel_suggest_weight: float = Field(default=0.1, ge=0, le=1)


class ObserverDeploymentConfig(BaseModel):
    """Default placement for observer containers that are not given an explicit position.

    `surround` (default): the first four observers surround the target's initial position
    (bearings HDG+45/135/225/315 deg at surround_radius_yd); further observers are added on a
    wider ring."""

    pattern: str = Field(default="surround", pattern="^(surround|grid|line|ring|random)$")
    initial_count: int = Field(default=4, ge=1, le=100)  # further containers wait as standby
    surround_radius_yd: float = Field(default=3000.0, gt=0)
    spacing_yd: float = Field(default=3000.0, gt=0)
    line_bearing_deg: float = Field(default=0.0, ge=0, lt=360)
    offset_ahead_yd: float = Field(default=9000.0)
    depth_ft: float = Field(default=200.0, ge=0)
    depth_step_ft: float = Field(default=150.0, ge=0)  # alternate depths break vertical mirror


class EstimatorConfig(BaseModel):
    """Assumptions used by the estimator (NOT truth)."""

    particle_count: int = Field(default=6000, ge=500, le=50000)
    model_frequency_sigma_hz: float = Field(default=0.03, gt=0)
    assumed_bias_sigma_hz: float = Field(default=0.5, ge=0)
    max_target_speed_kt: float = Field(default=25.0, gt=0)
    max_target_depth_ft: float = Field(default=1500.0, gt=0)
    horizontal_accel_sigma_mps2: float = Field(default=0.01, ge=0)
    vertical_accel_sigma_mps2: float = Field(default=0.002, ge=0)
    maneuver_fraction: float = Field(default=0.1, ge=0, le=0.5)
    maneuver_accel_sigma_mps2: float = Field(default=0.12, ge=0)
    maneuver_vertical_sigma_mps: float = Field(default=0.15, ge=0)
    move_min_window_s: int = Field(default=60, ge=20, le=3600)
    move_mismatch_chi2: float = Field(default=4.0, gt=1)
    use_bearing: bool = True
    bearing_sigma_deg: float = Field(default=15.0, gt=0, le=90)
    range_gate_softness_yd: float = Field(default=15.0, gt=0)
    current_gradient_ridge: float = Field(default=1e-6, ge=0)
    track_store_slots: int = Field(default=360, ge=10, le=2000)
    cpa_fit_half_window_s: int = Field(default=600, ge=30, le=3600)
    cpa_min_post_samples: int = Field(default=30, ge=5)
    move_interval_s: int = Field(default=10, ge=1, le=600)
    move_window_s: int = Field(default=600, ge=30, le=3600)
    move_epochs: int = Field(default=60, ge=5, le=600)
    move_starts: int = Field(default=8, ge=2, le=50)
    random_seed: int = 7
    # Lloyd's mirror depth fit (runs only while ScenarioConfig.lloyd.enabled)
    lloyd_fit_interval_s: int = Field(default=10, ge=1, le=600)
    lloyd_window_s: int = Field(default=600, ge=60, le=3600)
    lloyd_min_samples: int = Field(default=120, ge=20, le=3600)
    lloyd_depth_step_ft: float = Field(default=2.0, ge=0.5, le=50)
    lloyd_model_error_pct: float = Field(default=3.0, ge=0, le=50)  # assumed sound-speed model error
    lloyd_noise_correlation_s: float = Field(default=20.0, ge=0, le=600)  # assumed
    # assumed emitted-frequency fluctuation (1 sigma, on the primary tonal; the others in
    # proportion): common to every observer, so it is not averaged out by more observers
    assumed_stability_hz: float = Field(default=0.0, ge=0)
    # time differences between observers of the Doppler change when the target maneuvers
    # (the change reaches each observer after its own propagation delay r/c)
    use_maneuver_timing: bool = True
    maneuver_timing_window_s: int = Field(default=40, ge=15, le=300)  # kink fit window
    maneuver_timing_chi2: float = Field(default=25.0, gt=0)  # kink detection threshold
    maneuver_timing_min_slope_hz_s: float = Field(default=0.003, ge=0)  # smallest df/dt step
    maneuver_timing_sigma_s: float = Field(default=0.3, ge=0)  # added onset-time error


class ScenarioConfig(BaseModel):
    scenario_name: str = "AQUA-DRIFT default"
    observer_limit: int = Field(default=99, ge=1, le=99)  # observer slots obs-01 .. obs-99
    max_slant_range_yd: float = Field(default=6000.0, gt=0)
    max_observation_seconds: int = Field(default=10800, ge=1, le=10800)
    doppler_interval_seconds: int = Field(default=1, ge=1)
    smoothing_window_seconds: int = Field(default=900, ge=1, le=10800)
    presence_probability_pct: float = Field(default=90.0, gt=0, lt=100)
    target: TargetMotionConfig = TargetMotionConfig()
    current_field: CurrentFieldConfig = CurrentFieldConfig()
    source: SourceFrequencyConfig = SourceFrequencyConfig()
    bearing: BearingConfig = BearingConfig()
    forward: ForwardDeploymentConfig = ForwardDeploymentConfig()
    deployment: ObserverDeploymentConfig = ObserverDeploymentConfig()
    estimator: EstimatorConfig = EstimatorConfig()
    lloyd: LloydMirrorConfig = LloydMirrorConfig()
    layer: LayerConfig = LayerConfig()
    wind: WindConfig = WindConfig()


class EstimatorSettings(BaseModel):
    """Subset of the configuration the estimator is allowed to see (no truth)."""

    max_slant_range_yd: float
    smoothing_window_seconds: int
    presence_probability_pct: float
    sound_speed_mps: float
    estimator: EstimatorConfig
    lloyd_enabled: bool = False  # the on/off switch only; truth-side Lloyd parameters stay hidden
    propagation_delay: bool = False  # sound travel time is part of the measurement model

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> EstimatorSettings:
        return cls(
            max_slant_range_yd=config.max_slant_range_yd,
            smoothing_window_seconds=config.smoothing_window_seconds,
            presence_probability_pct=config.presence_probability_pct,
            sound_speed_mps=config.source.sound_speed_mps,
            estimator=config.estimator,
            lloyd_enabled=config.lloyd.enabled,
            propagation_delay=config.source.propagation_delay,
        )


# --------------------------------------------------------------------------- runtime state


class TickMessage(BaseModel):
    tick: int = Field(ge=0)
    # system time (UTC epoch seconds) the clock scheduled this tick for; fixes the time of day of
    # tick 0 (Snapshot.epoch_s) so every tick has a clock time, not just an elapsed count
    wall_s: float | None = None


MAX_TIME_SCALE = 100.0


class ClockSettings(BaseModel):
    """Simulation time per wall-clock second (1 = real time, 0 = paused)."""

    time_scale: float = Field(default=1.0, ge=0, le=MAX_TIME_SCALE)


class ClockStatus(BaseModel):
    """Polled by the clock container: synced is True once every worker finished the current tick."""

    tick: int
    time_scale: float
    synced: bool
    epoch_s: float | None = None


class TargetState(BaseModel):
    tick: int = Field(ge=0)
    position: Position
    hdg_deg: float = Field(ge=0, lt=360)
    cog_deg: float = Field(ge=0, lt=360)
    through_water_speed_kt: float = Field(ge=0)
    ground_speed_kt: float = Field(ge=0)
    through_water_velocity: Velocity
    ground_velocity: Velocity


class ObserverState(BaseModel):
    observer_id: str = Field(min_length=1, max_length=128)
    tick: int = Field(ge=0)
    position: Position
    ground_velocity: Velocity = Velocity()  # truth drift (display only)
    status: ObserverStatus = ObserverStatus.ACTIVE
    session: int = Field(default=0, ge=0)  # increments each time the observer slot is reused


class ObserverPlacement(BaseModel):
    """Optional explicit placement handed to the next observer container that starts."""

    position: Position
    observer_id: str | None = None
    source: str = "manual"  # manual (operator) | forward (automatic 前程 deployment)
    planned_tick: int | None = None  # drop time requested by the operator (None = as soon as possible)


class ObserverAssignment(Position):
    """Start position for an observer container plus its session for this slot."""

    observer_id: str
    session: int = 0


class OrchestratorFeed(BaseModel):
    """What the container orchestrator needs to decide how many observer containers to run."""

    tick: int
    limit: int
    active_ids: list[str]
    standby_ids: list[str]
    pending_placements: int
    initial_remaining: int


class TonalObservation(BaseModel):
    """One tonal as an observer measures it: the line centre (with the bandwidth error) and the
    line width it sees; recognized_frequency_hz is the observer's belief of the emitted line."""

    recognized_frequency_hz: float
    observed_frequency_hz: float | None = None  # None while not detected
    bandwidth_hz: float = 0.0


class DopplerObservation(BaseModel):
    """What an observer actually measures at a synchronized 1 s epoch.

    No bearing and no range: only the received frequency (error-free) while the target is
    inside the common maximum slant range, and an explicit non-detection otherwise.
    """

    observer_id: str
    tick: int = Field(ge=0)
    observer_position: Position  # exact position/depth at the observation time
    detected: bool
    observed_frequency_hz: float | None = None
    recognized_frequency_hz: float  # observer's belief of the source frequency (biased)
    bearing_deg: float | None = None  # horizontal true bearing with error (every interval_s)
    # every tonal (the primary first; observed_ / recognized_frequency_hz repeat the primary)
    tonals: list[TonalObservation] = Field(default_factory=list)
    # received level of the tonal [dB] (direct + surface-reflected path), only while detected
    # and only when the optional Lloyd's mirror calculation is enabled
    received_level_db: float | None = None


class DopplerTruth(BaseModel):
    """Simulator truth attached to an observation for evaluation/display only."""

    observer_id: str
    tick: int
    slant_range_yd: float
    relative_speed_kt: float
    relative_radial_speed_kt: float
    true_bearing_deg: float = 0.0
    propagation_delay_s: float = 0.0  # sound travel time from the target (emission) to here


class DopplerBatch(BaseModel):
    tick: int
    observations: list[DopplerObservation]
    truth: list[DopplerTruth] = Field(default_factory=list)


class ObserverFix(BaseModel):
    """Exact observer position/depth/time as seen by the estimator."""

    observer_id: str
    tick: int
    position: Position


class EstimationControl(BaseModel):
    """Operator start/stop of the estimator. A start begins a new run from the current tick."""

    running: bool = True
    run_id: int = 0
    started_tick: int = 0
    stopped_tick: int | None = None


class BearingReport(BaseModel):
    observer_id: str
    tick: int
    bearing_deg: float
    observer_position: Position


class DeploymentRecord(BaseModel):
    tick: int
    positions: list[Position]
    reason: str
    source: str = "forward"
    # with the layer: the drop task of each position (the map shows a planned point only while
    # its drop is open); empty without the layer (observers are queued at once)
    task_ids: list[int] = Field(default_factory=list)


class CancelSuggestions(BaseModel):
    """The replanner's cancellation proposals: task id -> reason (None withdraws a proposal)."""

    tick: int
    suggestions: dict[int, str | None]


class PlanBasis(BaseModel):
    """The estimate a planned drop was chosen on: the drop point relative to the estimated target
    (water frame, metres east / north) and its estimated heading and through-water speed at tick.
    Drop point and target drift with the same water, so the replanner can tell how far the
    estimate has moved since (aqua_drift.replanning)."""

    tick: int
    east_m: float
    north_m: float
    hdg_deg: float
    speed_kt: float


class DeploymentRequest(BaseModel):
    tick: int
    positions: list[Position]
    reason: str
    planned_ticks: list[int] | None = None  # optimal drop time of each position
    bases: list[PlanBasis] | None = None  # the estimate each position was planned on
    # replanning: open drop tasks this plan replaces (all or nothing: rejected with 409 when one
    # of them is no longer revisable) and the revision of the new drops
    replaces: list[int] = Field(default_factory=list)
    revision: int = 0


class DeploymentFeed(BaseModel):
    """What the deployer may see: estimates and observer positions only (no truth)."""

    tick: int
    generation: int = 0
    config: ForwardDeploymentConfig
    max_slant_range_yd: float
    depth_step_ft: float
    estimates: list[TrackEstimate]
    observer_positions: list[Position]
    pending_positions: list[Position]
    standby_count: int
    last_deploy_tick: int | None
    free_slots: int = 99
    # layer (設標者) availability for the drop-time planning: when / where it is free next
    layer_enabled: bool = False
    layer_ready_tick: int | None = None
    layer_ready_position: Position | None = None
    layer_ready_heading_deg: float | None = None
    # where it is now and its open drops in flight order (planned drop tick, None: at once):
    # a new plan is routed on after them
    layer_position: Position | None = None
    layer_heading_deg: float | None = None
    layer_queue: list[Position] = Field(default_factory=list)
    layer_queue_ticks: list[int | None] = Field(default_factory=list)
    layer_speed_kt: float = 200.0
    layer_max_bank_deg: float = 15.0
    source_frequency_hz: float = 400.0  # operator's (recognized) source frequency
    sound_speed_mps: float = 1500.0
    frequency_sigma_hz: float = 0.03
    # what the tracker measures besides the Doppler (the planner values the same measurements)
    use_bearing: bool = True
    bearing_sigma_deg: float = 15.0
    bearing_interval_s: float = 15.0
    range_gate_sigma_yd: float = 15.0
    max_target_depth_ft: float = 1500.0
    replan: ReplanFeed | None = None  # with the layer: what the replanner needs


class DropTask(BaseModel):
    """One additional observer to be laid by the layer.

    PROPOSED (waiting for the operator) -> APPROVED (the layer flies there) -> DONE (laid: the
    observer is in the water); or REJECTED / EXPIRED / CANCELLED / REPLACED (by a new plan after
    the estimate changed, aqua_drift.replanning). The drop point is planned in the water
    frame, so it drifts with the (estimated) current until the layer reaches it."""

    task_id: int
    created_tick: int
    source: str  # forward (automatic plan) | manual (operator placement) | operator (deploy now)
    reason: str = ""
    position: Position
    status: str = "PROPOSED"
    planned_tick: int | None = None  # optimal / requested drop time (None = as soon as possible)
    # the planner's optimal (or the operator's) drop time; planned_tick is moved from it only
    # when the layer cannot be there in the operator's drop order (設標順)
    requested_tick: int | None = None
    sequence: int = 0  # drop order among tasks with the same planned time (operator reorder)
    basis: PlanBasis | None = None  # the estimate an automatic drop was planned on (replanning)
    revision: int = 0  # how often this drop's plan was replaced (replanning keeps it from the limit on)
    # proposed to the operator for cancellation: the drop would no longer help detection (the layer
    # keeps flying it until the operator cancels it)
    cancel_suggestion: str | None = None
    cancel_suggested_tick: int | None = None
    approved_tick: int | None = None
    done_tick: int | None = None
    eta_s: float | None = None  # seconds from now to the expected drop
    # free fall from the layer (設標者): the drop point (position) is where the observer should
    # enter the water; the layer releases it at release_position (corrected upwind for the
    # estimated mean wind). After the release: where and when it entered the water.
    release_position: Position | None = None
    release_altitude_ft: float | None = None
    planned_position: Position | None = None  # the drop point when the observer entered the water
    splash_tick: int | None = None
    miss_yd: float | None = None  # entry point - drop point (horizontal)

    def flight_key(self) -> tuple[int, int, int]:
        """Order the layer flies the drops in: planned time (as soon as possible first), then
        the operator's drop order."""
        return (self.planned_tick if self.planned_tick is not None else -1,
                self.sequence or self.task_id, self.task_id)


class DropDecision(BaseModel):
    task_ids: list[int] | None = None  # None = every proposed drop


class DropReorder(BaseModel):
    task_ids: list[int]  # open drops in the new drop order (unlisted open drops follow in their order)


class DropReschedule(BaseModel):
    task_id: int
    planned_tick: int | None = None  # None = as soon as possible


class LayerState(BaseModel):
    tick: int
    position: Position  # depth 0 (sea surface)
    heading_deg: float
    speed_kt: float
    bank_deg: float = 0.0
    mode: str = "ORBIT"  # ORBIT (circling the estimated target) | TRANSIT (to a drop point) | HOLD (early at the point)
    task_id: int | None = None
    orbit_center: Position | None = None
    orbit_radius_yd: float = 0.0
    eta_s: float | None = None
    # heading the layer crosses its current drop point on (lined up for the next drop point;
    # None: any heading) and the drops it was chosen for ("task>next")
    approach_deg: float | None = None
    approach_key: str = ""
    # the Dubins path being flown to the current drop point (turn sides +1 right / -1 left / 0
    # straight and the remaining lengths in metres): followed as it is when a fresh plan comes out
    # longer (small errors at the joins of the path can make the exact plan jump to a detour)
    path_sides: list[int] = Field(default_factory=list)
    path_lengths_m: list[float] = Field(default_factory=list)
    # planned flight path ahead (飛行予定経路): [latitude, longitude] from the current position
    # through the open drop points in order, along the turn-limited guidance path
    planned_path: list[tuple[float, float]] = Field(default_factory=list)
    # 3-D flight and the wind at the flight altitude
    altitude_ft: float = 0.0
    ground_speed_kt: float | None = None
    track_deg: float | None = None
    wind_direction_deg: float | None = None  # from
    wind_speed_kt: float | None = None


class DropRelease(BaseModel):
    """One drop (投下): the observer leaves the layer and falls freely to the sea surface."""

    task_id: int
    release_tick: int
    release_position: Position  # depth 0; the layer was at altitude_ft above it
    altitude_ft: float
    ground_east_kt: float  # layer ground velocity at the release (the observer's initial velocity)
    ground_north_kt: float
    splash_tick: int
    splash_position: Position  # truth: where the observer enters the water (its hanging depth)
    fall_time_s: float
    planned_position: Position  # the drop point when the observer enters the water


class LayerUpdate(BaseModel):
    """Posted by the layer container every tick."""

    state: LayerState
    task_positions: dict[int, Position] = Field(default_factory=dict)  # drifted drop points
    task_eta_s: dict[int, float] = Field(default_factory=dict)
    # laid: task id -> point where the observer enters the water
    completed: dict[int, Position] = Field(default_factory=dict)
    releases: dict[int, DropRelease] = Field(default_factory=dict)  # free fall of the laid observers
    release_positions: dict[int, Position] = Field(default_factory=dict)  # corrected release points


class WindEstimate(BaseModel):
    """Mean wind from the drop altitude to the sea surface (投下高度から海面までの平均風向風速),
    from where an observer entered the water compared with the predicted entry point of the
    same free fall without wind. Known after the observer is in the water (splash_tick)."""

    task_id: int
    tick: int
    altitude_ft: float
    fall_time_s: float
    release_position: Position
    no_wind_position: Position  # predicted entry point without wind
    splash_position: Position
    offset_yd: float  # entry point - no-wind prediction
    east_kt: float  # mean wind vector (towards)
    north_kt: float
    direction_deg: float  # from
    speed_kt: float
    miss_yd: float | None = None  # entry point - drop point
    # truth for comparison: vector mean of the wind profile from the drop altitude to the sea
    true_direction_deg: float | None = None
    true_speed_kt: float | None = None


class LayerFeed(BaseModel):
    tick: int
    generation: int = 0
    config: LayerConfig
    tasks: list[DropTask]  # approved, in order
    datum: Position  # orbit centre: estimated target position (or the configured datum)
    current_east_kt: float = 0.0  # estimated current (drift of the planned drop points)
    current_north_kt: float = 0.0
    state: LayerState | None = None
    # wind truth (the layer flies in it and the observers fall through it); None: the observer
    # is in the water at the drop point at once (no free fall)
    wind: WindConfig | None = None
    wind_estimate: WindEstimate | None = None  # latest estimated mean wind (release correction)


class DeploymentStatus(BaseModel):
    standby_count: int = 0
    pending_placements: int = 0
    last_deploy_tick: int | None = None
    history: list[DeploymentRecord] = Field(default_factory=list)
    approval: str = "auto"
    tasks: list[DropTask] = Field(default_factory=list)  # recent drop tasks (all states)
    layer: LayerState | None = None
    falling: int = 0  # observers released and still falling
    wind_estimates: list[WindEstimate] = Field(default_factory=list)  # newest last


class EstimatorFeed(BaseModel):
    tick: int
    time_scale: float = 1.0
    generation: int = 0
    estimation: EstimationControl = EstimationControl()
    settings: EstimatorSettings
    observers: list[ObserverFix]
    archived_observer_ids: list[str]
    batches: list[DopplerBatch]


# --------------------------------------------------------------------------- estimates


class RegionComponent(BaseModel):
    """One connected part of the highest-density presence region."""

    probability_mass_pct: float
    centroid: Position
    polygon: list[list[float]]  # [[lon, lat], ...] horizontal hull
    min_depth_ft: float
    max_depth_ft: float
    voxels: list[list[float]] = Field(default_factory=list)  # [lon, lat, depth_ft]
    voxel_size_yd: float = 0.0
    voxel_height_ft: float = 0.0


class PresenceRegion(BaseModel):
    probability_pct: float
    components: list[RegionComponent] = Field(default_factory=list)
    disconnected: bool = False


class Uncertainty(BaseModel):
    horizontal_major_yd: float
    horizontal_minor_yd: float
    horizontal_major_axis_deg: float
    depth_sigma_ft: float
    ground_speed_sigma_kt: float
    through_water_speed_sigma_kt: float
    cog_sigma_deg: float
    hdg_sigma_deg: float
    bias_sigma_hz: float


class TrackPoint(BaseModel):
    tick: int
    latitude: float
    longitude: float
    depth_ft: float
    horizontal_sigma_yd: float
    depth_sigma_ft: float
    ground_speed_kt: float
    cog_deg: float


class RelativeKinematics(BaseModel):
    observer_id: str
    relative_speed_kt: float
    slant_range_yd: float
    detected: bool


class CpaResult(BaseModel):
    observer_id: str
    pass_index: int
    final: bool
    cpa_tick: float
    cpa_tick_sigma_s: float
    cpa_slant_range_yd: float
    cpa_slant_range_sigma_yd: float
    relative_speed_kt: float
    relative_speed_sigma_kt: float
    slope_hz_per_s: float
    method_note: str
    range_from_slope_yd: float | None = None
    bias_shift_tick_s: float = 0.0
    bias_range_sigma_yd: float = 0.0
    speed_range_sigma_yd: float = 0.0
    fit_source_frequency_hz: float | None = None
    fit_cpa_tick: float | None = None
    fit_cpa_slant_range_yd: float | None = None
    fit_relative_speed_kt: float | None = None


class CurrentEstimate(BaseModel):
    base_velocity: Velocity
    gradient_per_nm: list[list[float]]
    reference_position: Position | None = None
    observer_count: int
    sample_count: int
    window_seconds: float
    residual_kt: float


class TrackEstimate(BaseModel):
    mode: EstimateMode
    tick: int = Field(ge=0)
    observability_status: str
    current_position: Position | None = None
    depth_ft: float | None = None
    relative_speed_kt: float | None = None
    ground_speed_kt: float | None = None
    through_water_speed_kt: float | None = None
    hdg_deg: float | None = None
    cog_deg: float | None = None
    vertical_rate_fps: float | None = None
    source_bias_hz: float | None = None
    uncertainty: Uncertainty | None = None
    presence_region: PresenceRegion
    track: list[TrackPoint] = Field(default_factory=list)
    relative: list[RelativeKinematics] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LloydObserverFit(BaseModel):
    """Depth fitted to one observer's received-level interference pattern."""

    observer_id: str
    status: str  # OK | AMBIGUOUS | NO_FRINGES | NO_PATTERN | FEW_SAMPLES
    samples: int
    fringes: float  # change of the path difference over the window, in wavelengths
    depth_ft: float | None = None
    sigma_ft: float | None = None
    reflection: float | None = None  # fitted coherent surface reflection magnitude


class LloydDepthEstimate(BaseModel):
    """Target depth from the Lloyd's mirror (direct + surface-reflected path) interference."""

    enabled: bool
    tick: int
    status: str  # OFF | WAITING | OK | NO_RESULT
    depth_ft: float | None = None
    sigma_ft: float | None = None
    used_observers: int = 0
    observers: list[LloydObserverFit] = Field(default_factory=list)
    fit_ms: float = 0.0
    applied: bool = False  # fed to the particle filter as a depth measurement


class EstimatorOutput(BaseModel):
    tick: int
    estimates: list[TrackEstimate]
    cpa: list[CpaResult]
    current: CurrentEstimate | None = None
    lloyd: LloydDepthEstimate | None = None


class ObserverRecord(BaseModel):
    state: ObserverState
    registered_tick: int
    last_tick: int


class SimState(BaseModel):
    """Lightweight state polled by the simulation containers."""

    tick: int
    generation: int = 0
    time_scale: float = 1.0
    config: ScenarioConfig
    target: TargetState | None
    observers: list[ObserverRecord]


class Snapshot(BaseModel):
    tick: int
    generation: int = 0
    time_scale: float = 1.0
    epoch_s: float | None = None  # system time (UTC epoch s) of tick 0: tick t is at epoch_s + t
    deployment: DeploymentStatus = DeploymentStatus()
    estimation: EstimationControl = EstimationControl()
    bearings: list[BearingReport] = Field(default_factory=list)
    config: ScenarioConfig
    target: TargetState | None
    observers: list[ObserverRecord]
    doppler: DopplerBatch | None
    estimates: list[TrackEstimate]
    cpa: list[CpaResult]
    current_estimate: CurrentEstimate | None
    archived_observer_ids: list[str]
    lloyd: LloydDepthEstimate | None = None


class ReplanFeed(BaseModel):
    """What the replanner (aqua_drift.replanning) needs besides the deployment feed: every open
    drop (in flight order) and the layer, so it can tell which drops are kept and when / where the
    layer is free with only those."""

    tasks: list[DropTask] = Field(default_factory=list)
    layer_state: LayerState | None = None
    layer: LayerConfig
    datum: Position  # where the layer orbits before it has a state
    # observers on their way into the water that are not open drops (queued placements, falling)
    other_pending: list[Position] = Field(default_factory=list)
    last_replan_tick: int | None = None


DeploymentFeed.model_rebuild()
