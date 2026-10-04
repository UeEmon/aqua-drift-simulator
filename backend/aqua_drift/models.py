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


class SourceFrequencyConfig(BaseModel):
    """Source truth. The frequency is constant; the recognition bias is common to every
    observer and does not change with time (its magnitude is a free scenario parameter)."""

    source_frequency_hz: float = Field(default=400.0, gt=0)
    shared_recognition_bias_hz: float = 0.0
    sound_speed_mps: float = Field(default=1500.0, gt=0)


class BearingConfig(BaseModel):
    """Horizontal true bearing observation (observer -> target), inside the max slant range.

    Errors: normal, zero mean, sigma_deg, independent in time and between observers."""

    enabled: bool = True
    sigma_deg: float = Field(default=15.0, gt=0, le=90)
    interval_s: int = Field(default=15, ge=1, le=600)
    random_seed: int = 11


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


class ScenarioConfig(BaseModel):
    scenario_name: str = "AQUA-DRIFT default"
    observer_limit: int = Field(default=100, ge=1, le=100)
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


class EstimatorSettings(BaseModel):
    """Subset of the configuration the estimator is allowed to see (no truth)."""

    max_slant_range_yd: float
    smoothing_window_seconds: int
    presence_probability_pct: float
    sound_speed_mps: float
    estimator: EstimatorConfig

    @classmethod
    def from_config(cls, config: ScenarioConfig) -> EstimatorSettings:
        return cls(
            max_slant_range_yd=config.max_slant_range_yd,
            smoothing_window_seconds=config.smoothing_window_seconds,
            presence_probability_pct=config.presence_probability_pct,
            sound_speed_mps=config.source.sound_speed_mps,
            estimator=config.estimator,
        )


# --------------------------------------------------------------------------- runtime state


class TickMessage(BaseModel):
    tick: int = Field(ge=0)


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


class ObserverPlacement(BaseModel):
    """Optional explicit placement handed to the next observer container that starts."""

    position: Position
    observer_id: str | None = None
    source: str = "manual"  # manual (operator) | forward (automatic 前程 deployment)


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


class DopplerTruth(BaseModel):
    """Simulator truth attached to an observation for evaluation/display only."""

    observer_id: str
    tick: int
    slant_range_yd: float
    relative_speed_kt: float
    relative_radial_speed_kt: float
    true_bearing_deg: float = 0.0


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


class DeploymentRequest(BaseModel):
    tick: int
    positions: list[Position]
    reason: str


class DeploymentFeed(BaseModel):
    """What the deployer may see: estimates and observer positions only (no truth)."""

    tick: int
    config: ForwardDeploymentConfig
    max_slant_range_yd: float
    depth_step_ft: float
    estimates: list[TrackEstimate]
    observer_positions: list[Position]
    pending_positions: list[Position]
    standby_count: int
    last_deploy_tick: int | None


class DeploymentStatus(BaseModel):
    standby_count: int = 0
    pending_placements: int = 0
    last_deploy_tick: int | None = None
    history: list[DeploymentRecord] = Field(default_factory=list)


class EstimatorFeed(BaseModel):
    tick: int
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


class EstimatorOutput(BaseModel):
    tick: int
    estimates: list[TrackEstimate]
    cpa: list[CpaResult]
    current: CurrentEstimate | None = None


class ObserverRecord(BaseModel):
    state: ObserverState
    registered_tick: int
    last_tick: int


class SimState(BaseModel):
    """Lightweight state polled by the simulation containers."""

    tick: int
    generation: int = 0
    config: ScenarioConfig
    target: TargetState | None
    observers: list[ObserverRecord]


class Snapshot(BaseModel):
    tick: int
    generation: int = 0
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
