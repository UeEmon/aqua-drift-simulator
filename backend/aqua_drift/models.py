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
    ONLINE = "ONLINE"
    SMOOTHED = "SMOOTHED"


class Position(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    depth_ft: float = Field(ge=0)


class Velocity(BaseModel):
    east_kt: float = 0.0
    north_kt: float = 0.0
    vertical_fps: float = 0.0


class TargetMotionConfig(BaseModel):
    initial_position: Position = Position(latitude=35.0, longitude=140.0, depth_ft=500.0)
    desired_hdg_deg: float = Field(default=90.0, ge=0, lt=360)
    desired_through_water_speed_kt: float = Field(default=8.0, ge=0)
    desired_depth_ft: float = Field(default=500.0, ge=0)
    hdg_rate_deg_per_sec: float = Field(default=1.0, gt=0)
    speed_rate_kt_per_sec: float = Field(default=0.1, gt=0)
    depth_rate_ft_per_sec: float = Field(default=2.0, gt=0)


class CurrentFieldConfig(BaseModel):
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
    source_frequency_hz: float = Field(default=400.0, gt=0)
    shared_recognition_bias_hz: float = 0.0
    sound_speed_mps: float = Field(default=1500.0, gt=0)


class ScenarioConfig(BaseModel):
    scenario_name: str = "AQUA-DRIFT default"
    observer_limit: int = Field(default=100, ge=1, le=100)
    max_slant_range_yd: float = Field(default=12000.0, gt=0)
    max_observation_seconds: int = Field(default=10800, ge=1, le=10800)
    doppler_interval_seconds: int = Field(default=1, ge=1)
    smoothing_window_seconds: int = Field(default=300, ge=1, le=10800)
    presence_probability_pct: float = Field(default=95.0, gt=0, lt=100)
    target: TargetMotionConfig = TargetMotionConfig()
    current_field: CurrentFieldConfig = CurrentFieldConfig()
    source: SourceFrequencyConfig = SourceFrequencyConfig()


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
    ground_velocity: Velocity
    status: ObserverStatus = ObserverStatus.ACTIVE


class DopplerObservation(BaseModel):
    observer_id: str
    tick: int = Field(ge=0)
    observed_frequency_hz: float
    recognized_frequency_hz: float
    relative_radial_speed_kt: float
    relative_speed_kt: float
    slant_range_yd: float = Field(ge=0)
    is_new_closest: bool = False


class PresenceRegionComponent(BaseModel):
    observer_id: str
    center: Position
    radius_yd: float = Field(ge=0)
    description: str


class PresenceRegion(BaseModel):
    probability_pct: float
    components: list[PresenceRegionComponent] = Field(default_factory=list)
    disconnected: bool = False


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
    presence_region: PresenceRegion
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObserverRecord(BaseModel):
    state: ObserverState
    registered_tick: int
    last_tick: int


class Snapshot(BaseModel):
    tick: int
    config: ScenarioConfig
    target: TargetState | None
    observers: list[ObserverRecord]
    doppler: list[DopplerObservation]
    estimates: list[TrackEstimate]
    archived_observer_ids: list[str]
