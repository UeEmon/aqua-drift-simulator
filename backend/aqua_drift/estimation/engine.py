"""Tracking engine: observations in, ONLINE / SMOOTHED estimates out.

The engine only consumes `DopplerBatch` observations (time, exact observer position/depth,
detection flag, received frequency, recognized frequency) and `EstimatorSettings`. It never
sees the simulator truth.
"""
from __future__ import annotations

import math

import numpy as np

from aqua_drift.estimation.cpa import CpaAnalyzer
from aqua_drift.estimation.current_fit import CurrentFieldEstimator, CurrentFit
from aqua_drift.estimation.frame import (
    FT_TO_M,
    KNOT_TO_MPS,
    NM_TO_M,
    YD_TO_M,
    LocalFrame,
)
from aqua_drift.estimation.lloyd import LloydDepthEstimator, LloydFitSettings
from aqua_drift.estimation.particle_filter import DopplerParticleFilter, ObservationRow
from aqua_drift.estimation.region import presence_region
from aqua_drift.models import (
    CurrentEstimate,
    DopplerBatch,
    EstimateMode,
    EstimatorOutput,
    EstimatorSettings,
    LloydDepthEstimate,
    PresenceRegion,
    RelativeKinematics,
    TrackEstimate,
    TrackPoint,
    Uncertainty,
    Velocity,
)

MAX_TRACK_POINTS = 2000


def _circular_mean_std(angles_rad: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    s = float(weights @ np.sin(angles_rad))
    c = float(weights @ np.cos(angles_rad))
    resultant = min(math.hypot(s, c), 1.0)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    std = math.degrees(math.sqrt(max(-2.0 * math.log(max(resultant, 1e-12)), 0.0)))
    return mean, std


def _thin(points: list[TrackPoint]) -> list[TrackPoint]:
    if len(points) <= MAX_TRACK_POINTS:
        return points
    half = len(points) // 2
    return points[:half:2] + points[half:]


class TrackingEngine:
    def __init__(self, settings: EstimatorSettings) -> None:
        self.settings = settings
        self.frame: LocalFrame | None = None
        self.current_estimator = CurrentFieldEstimator()
        self.current_fit = CurrentFit.zero()
        self.cpa = CpaAnalyzer()
        self.pf = self._new_filter()
        self.tick = -1
        self.last_rows: list[ObservationRow] = []
        self.last_detect_tick: int | None = None
        self.nondetect_tick: dict[str, int] = {}
        self.online_track: list[TrackPoint] = []
        self.smoothed_track: dict[int, TrackPoint] = {}
        self.divergence = 0
        self.update_info: dict[str, float] = {}
        self.cpa_results = []
        self.reinitializations = 0
        self.lloyd = LloydDepthEstimator()
        self.lloyd_result: LloydDepthEstimate | None = None

    # ------------------------------------------------------------------ settings
    def _new_filter(self) -> DopplerParticleFilter:
        e = self.settings.estimator
        pf = DopplerParticleFilter(
            particle_count=e.particle_count,
            max_range_m=self.settings.max_slant_range_yd * YD_TO_M,
            sound_speed=self.settings.sound_speed_mps,
            model_sigma_hz=e.model_frequency_sigma_hz,
            bias_sigma_hz=e.assumed_bias_sigma_hz,
            max_speed_mps=e.max_target_speed_kt * KNOT_TO_MPS,
            max_depth_m=e.max_target_depth_ft * FT_TO_M,
            accel_h=e.horizontal_accel_sigma_mps2,
            accel_v=e.vertical_accel_sigma_mps2,
            gate_softness_m=e.range_gate_softness_yd * YD_TO_M,
            seed=e.random_seed,
            maneuver_fraction=e.maneuver_fraction,
            maneuver_accel=e.maneuver_accel_sigma_mps2,
            maneuver_vertical=e.maneuver_vertical_sigma_mps,
            move_min_window_s=e.move_min_window_s,
            move_mismatch_chi2=e.move_mismatch_chi2,
            use_bearing=e.use_bearing,
            bearing_sigma_rad=math.radians(e.bearing_sigma_deg),
        )
        pf.configure_history(self.settings.smoothing_window_seconds, e.track_store_slots)
        return pf

    def apply_settings(self, settings: EstimatorSettings) -> None:
        if settings == self.settings:
            return
        old = self.settings
        self.settings = settings
        e = settings.estimator
        if e.particle_count != old.estimator.particle_count or e.random_seed != old.estimator.random_seed:
            self.pf = self._new_filter()  # re-initializes on next detection
            return
        pf = self.pf
        pf.max_range = settings.max_slant_range_yd * YD_TO_M
        pf.c = settings.sound_speed_mps
        pf.sigma_f = e.model_frequency_sigma_hz
        pf.bias_sigma = e.assumed_bias_sigma_hz
        pf.max_speed = e.max_target_speed_kt * KNOT_TO_MPS
        pf.max_depth = e.max_target_depth_ft * FT_TO_M
        pf.accel_h = e.horizontal_accel_sigma_mps2
        pf.accel_v = e.vertical_accel_sigma_mps2
        pf.gate_soft = e.range_gate_softness_yd * YD_TO_M
        pf.maneuver_fraction = e.maneuver_fraction
        pf.maneuver_accel = e.maneuver_accel_sigma_mps2
        pf.maneuver_vertical = e.maneuver_vertical_sigma_mps
        pf.move_min_window_s = e.move_min_window_s
        pf.move_mismatch_chi2 = e.move_mismatch_chi2
        pf.use_bearing = e.use_bearing
        pf.bearing_sigma = math.radians(e.bearing_sigma_deg)
        pf.configure_history(settings.smoothing_window_seconds, e.track_store_slots)

    # ------------------------------------------------------------------ processing
    def detectable_time_s(self) -> float:
        """Detectable time = detectable distance / target speed (used as the period over
        which the current field is treated as linear)."""
        r = self.settings.max_slant_range_yd * YD_TO_M
        if self.pf.initialized:
            speed = float(self.pf.w @ np.linalg.norm(self.pf.x[:, 3:5], axis=1))
        else:
            speed = self.settings.estimator.max_target_speed_kt * KNOT_TO_MPS
        speed = max(speed, 1.0 * KNOT_TO_MPS)
        return float(np.clip(r / speed, 60.0, 10800.0))

    def _observers_in_area(self) -> set[str] | None:
        if not self.pf.initialized:
            return None
        centre = self.pf.mean_state()[0:3]
        limit = 2.0 * self.pf.max_range
        ids = set()
        for row in self.last_rows:
            if np.linalg.norm(row.position - centre) <= limit:
                ids.add(row.observer_id)
        return ids or None

    def process(self, batch: DopplerBatch) -> None:
        if batch.tick <= self.tick:
            return
        if self.frame is None:
            if not batch.observations:
                return
            self.frame = LocalFrame(batch.observations[0].observer_position)
        for obs in batch.observations:
            self.current_estimator.add_fix(
                obs.observer_id, batch.tick, self.frame.to_local(obs.observer_position)
            )
        if self.tick < 0 or batch.tick % 5 == 0 or not self.pf.initialized:
            self.current_fit = self.current_estimator.fit(
                batch.tick,
                self.detectable_time_s(),
                self._observers_in_area(),
                self.settings.estimator.current_gradient_ridge,
            )
        rows: list[ObservationRow] = []
        for obs in batch.observations:
            point = self.frame.to_local(obs.observer_position)
            velocity = self.current_estimator.observer_velocity(obs.observer_id)
            if velocity is None:
                velocity = self.current_fit.velocity(point[None])[0]
            rows.append(
                ObservationRow(
                    observer_id=obs.observer_id,
                    position=point,
                    velocity=velocity,
                    detected=obs.detected,
                    frequency=obs.observed_frequency_hz,
                    recognized=obs.recognized_frequency_hz,
                    bearing=(
                        math.radians(obs.bearing_deg) if obs.bearing_deg is not None else None
                    ),
                )
            )
            self.cpa.add(
                obs.observer_id,
                batch.tick,
                obs.detected,
                obs.observed_frequency_hz,
                obs.recognized_frequency_hz,
            )
            if (
                self.settings.lloyd_enabled
                and obs.received_level_db is not None
                and obs.observed_frequency_hz is not None
            ):
                self.lloyd.add(obs.observer_id, batch.tick, point, obs.received_level_db, obs.observed_frequency_hz)
        previously_watching = {
            row.observer_id
            for row in rows
            if self.nondetect_tick.get(row.observer_id) == batch.tick - 1
        }
        any_detection = any(row.detected for row in rows)
        if any_detection:
            self.last_detect_tick = batch.tick

        self.pf.record_epoch(batch.tick, rows, self.settings.estimator.move_window_s)
        if not self.pf.initialized:
            if any_detection:
                self.pf.configure_history(
                    self.settings.smoothing_window_seconds,
                    self.settings.estimator.track_store_slots,
                )
                if self.pf.initialize(batch.tick, rows, self.current_fit, previously_watching):
                    self.pf.update(rows, self.current_fit)
        else:
            self.pf.predict(batch.tick - self.pf.tick, self.current_fit)
            self.pf.tick = batch.tick
            detections = sum(1 for row in rows if row.detected and row.frequency is not None)
            if detections:
                loglik = self.pf.log_likelihood(rows)
                best = float(loglik.max()) / detections
                self.divergence = self.divergence + 1 if best < -200.0 else 0
            if self.divergence >= 3:
                self.reinitializations += 1
                self.divergence = 0
                self.pf = self._new_filter()
                self.online_track.append(self._track_point())  # keep last pre-reset point
                if self.pf.initialize(batch.tick, rows, self.current_fit, previously_watching):
                    self.pf.update(rows, self.current_fit)
            else:
                self.update_info = self.pf.update(rows, self.current_fit)
            est = self.settings.estimator
            if self.pf.initialized and batch.tick % est.move_interval_s == 0:
                self.update_info.update(
                    self.pf.move(self.current_fit, est.move_epochs, est.move_starts)
                )

        self._lloyd_step(batch.tick)
        for row in rows:
            if not row.detected:
                self.nondetect_tick[row.observer_id] = batch.tick
        self.last_rows = rows
        self.tick = batch.tick
        if self.pf.initialized and batch.tick % self.pf.hist_stride == 0:
            self.online_track.append(self._track_point())
            self.online_track = _thin(self.online_track)

    # ------------------------------------------------------------------ Lloyd's mirror depth
    LLOYD_MAX_TRACK_SIGMA_M = 150.0

    def _lloyd_step(self, tick: int) -> None:
        """Optional (heavy) depth fit to the received-level interference pattern."""
        e = self.settings.estimator
        if not self.settings.lloyd_enabled:
            if self.lloyd.samples or self.lloyd_result is None or self.lloyd_result.enabled:
                self.lloyd.clear()
                self.pf.set_depth_fix(None)
                self.lloyd_result = LloydDepthEstimate(enabled=False, tick=max(tick, 0), status="OFF")
            return
        self.lloyd.prune(tick, e.lloyd_window_s)
        if self.lloyd_result is None or not self.lloyd_result.enabled:
            self.lloyd_result = LloydDepthEstimate(enabled=True, tick=tick, status="WAITING")
        if tick % e.lloyd_fit_interval_s != 0 or not self.pf.initialized:
            return
        pf = self.pf
        mean = pf.w @ pf.x
        diff = pf.x[:, 0:2] - mean[0:2]
        horizontal_sigma = math.sqrt(max(float(np.sum(pf.w[:, None] * diff * diff)), 0.0))
        if horizontal_sigma > self.LLOYD_MAX_TRACK_SIGMA_M:
            self.lloyd_result = LloydDepthEstimate(enabled=True, tick=tick, status="WAITING")
            return  # the horizontal range from the Doppler track is not good enough yet
        ground = pf.w @ pf.ground_v
        position = mean[0:3]

        def track(ticks: np.ndarray) -> np.ndarray:
            # constant velocity within the window (the window is cut after a maneuver)
            return position[None, :] - ground[None, :] * (tick - ticks)[:, None]

        window = float(min(e.lloyd_window_s, getattr(pf, "move_window_eff", e.lloyd_window_s) or e.lloyd_window_s))
        settings = LloydFitSettings(
            sound_speed=self.settings.sound_speed_mps,
            max_depth_m=e.max_target_depth_ft * FT_TO_M,
            depth_step_m=e.lloyd_depth_step_ft * FT_TO_M,
            min_samples=e.lloyd_min_samples,
            window_s=window,
            model_error_frac=e.lloyd_model_error_pct / 100.0,
            noise_correlation_s=e.lloyd_noise_correlation_s,
        )
        result = self.lloyd.fit(tick, track, settings)
        if result.depth_ft is not None and result.sigma_ft is not None:
            depth_m = result.depth_ft * FT_TO_M
            sigma_m = result.sigma_ft * FT_TO_M
            # successive fits share most of their data: spread one window's information over
            # the fits made within it so that it is not counted many times
            repeats = max(window / e.lloyd_fit_interval_s, 1.0)
            pf.apply_depth_measurement(depth_m, sigma_m * math.sqrt(repeats), self.current_fit)
            pf.set_depth_fix(depth_m, sigma_m)
            result.applied = True
        self.lloyd_result = result

    # ------------------------------------------------------------------ summaries
    def _track_point(self, tick: int | None = None) -> TrackPoint:
        pf = self.pf
        w = pf.w
        mean = w @ pf.x
        diff = pf.x[:, 0:3] - mean[0:3]
        cov = (diff * w[:, None]).T @ diff
        geo = self.frame.to_geo(*mean[0:3])
        ground = w @ pf.ground_v
        return TrackPoint(
            tick=pf.tick if tick is None else tick,
            latitude=geo.latitude,
            longitude=geo.longitude,
            depth_ft=geo.depth_ft,
            horizontal_sigma_yd=math.sqrt(max(cov[0, 0] + cov[1, 1], 0.0)) / YD_TO_M,
            depth_sigma_ft=math.sqrt(max(cov[2, 2], 0.0)) / FT_TO_M,
            ground_speed_kt=math.hypot(ground[0], ground[1]) / KNOT_TO_MPS,
            cog_deg=math.degrees(math.atan2(ground[0], ground[1])) % 360.0,
        )

    def _update_smoothed(self) -> None:
        for tick, mean, cov in self.pf.smoothed_history():
            geo = self.frame.to_geo(*mean[0:3])
            self.smoothed_track[tick] = TrackPoint(
                tick=tick,
                latitude=geo.latitude,
                longitude=geo.longitude,
                depth_ft=geo.depth_ft,
                horizontal_sigma_yd=math.sqrt(max(cov[0, 0] + cov[1, 1], 0.0)) / YD_TO_M,
                depth_sigma_ft=math.sqrt(max(cov[2, 2], 0.0)) / FT_TO_M,
                ground_speed_kt=math.hypot(mean[3], mean[4]) / KNOT_TO_MPS,
                cog_deg=math.degrees(math.atan2(mean[3], mean[4])) % 360.0,
            )
        if len(self.smoothed_track) > MAX_TRACK_POINTS:
            keys = sorted(self.smoothed_track)
            half = len(keys) // 2
            for key in keys[:half][1::2]:
                del self.smoothed_track[key]

    def _current_estimate(self) -> CurrentEstimate | None:
        if self.frame is None:
            return None
        fit = self.current_fit
        mps_to_fps = 1.0 / FT_TO_M
        base = Velocity(
            east_kt=float(fit.base[0] / KNOT_TO_MPS),
            north_kt=float(fit.base[1] / KNOT_TO_MPS),
            vertical_fps=float(fit.base[2] * mps_to_fps),
        )
        # (m/s)/m -> kt/NM for the horizontal rows, ft/s per NM for the vertical row
        scale = np.array([[NM_TO_M / KNOT_TO_MPS], [NM_TO_M / KNOT_TO_MPS], [NM_TO_M * mps_to_fps]])
        gradient = (fit.gradient * scale).tolist()
        return CurrentEstimate(
            base_velocity=base,
            gradient_per_nm=gradient,
            reference_position=self.frame.to_geo(*fit.reference),
            observer_count=fit.observer_count,
            sample_count=fit.sample_count,
            window_seconds=fit.window_seconds,
            residual_kt=fit.residual_mps / KNOT_TO_MPS,
        )

    def output(self) -> EstimatorOutput:
        e = self.settings.estimator
        cpa = self.cpa.results(
            self.tick,
            self.settings.sound_speed_mps,
            e.assumed_bias_sigma_hz,
            e.model_frequency_sigma_hz,
            e.cpa_fit_half_window_s,
            e.cpa_min_post_samples,
        )
        base_meta = {
            "direction_input_available": bool(e.use_bearing),
            "observation_inputs": (
                "Doppler frequency, detection flag, observer time/position/depth"
                + (", horizontal bearing" if e.use_bearing else "")
                + (", received level (Lloyd's mirror depth)" if self.settings.lloyd_enabled else "")
            ),
            "detectable_time_s": self.detectable_time_s(),
            "particle_count": self.pf.n,
            "recompute_window_s": self.settings.smoothing_window_seconds,
            "history_stride_s": self.pf.hist_stride,
            "reinitializations": self.reinitializations,
            "maneuver_detected_tick": self.pf.maneuver_detected_tick,
        }
        if not self.pf.initialized or self.frame is None:
            estimates = [
                TrackEstimate(
                    mode=mode,
                    tick=max(self.tick, 0),
                    observability_status="NO_DETECTION",
                    presence_region=PresenceRegion(
                        probability_pct=self.settings.presence_probability_pct
                    ),
                    metadata=base_meta,
                )
                for mode in (EstimateMode.ONLINE, EstimateMode.SMOOTHED)
            ]
            return EstimatorOutput(
                tick=max(self.tick, 0), estimates=estimates, cpa=cpa,
                current=self._current_estimate(), lloyd=self.lloyd_result,
            )

        pf = self.pf
        w = pf.w
        x = pf.x
        mean = w @ x
        diff = x[:, 0:3] - mean[0:3]
        cov = (diff * w[:, None]).T @ diff
        evals, evecs = np.linalg.eigh(cov[:2, :2])
        major_vec = evecs[:, 1]
        ground = pf.ground_v
        g_speed = np.hypot(ground[:, 0], ground[:, 1])
        w_speed = np.hypot(x[:, 3], x[:, 4])
        cog, cog_std = _circular_mean_std(np.arctan2(ground[:, 0], ground[:, 1]), w)
        hdg, hdg_std = _circular_mean_std(np.arctan2(x[:, 3], x[:, 4]), w)
        g_mean = float(w @ g_speed)
        u_mean = float(w @ w_speed)
        uncertainty = Uncertainty(
            horizontal_major_yd=math.sqrt(max(evals[1], 0.0)) / YD_TO_M,
            horizontal_minor_yd=math.sqrt(max(evals[0], 0.0)) / YD_TO_M,
            horizontal_major_axis_deg=math.degrees(math.atan2(major_vec[0], major_vec[1])) % 180.0,
            depth_sigma_ft=math.sqrt(max(cov[2, 2], 0.0)) / FT_TO_M,
            ground_speed_sigma_kt=math.sqrt(max(float(w @ (g_speed - g_mean) ** 2), 0.0))
            / KNOT_TO_MPS,
            through_water_speed_sigma_kt=math.sqrt(max(float(w @ (w_speed - u_mean) ** 2), 0.0))
            / KNOT_TO_MPS,
            cog_sigma_deg=cog_std,
            hdg_sigma_deg=hdg_std,
            bias_sigma_hz=math.sqrt(max(float(w @ (x[:, 6] - mean[6]) ** 2), 0.0)),
        )
        region = presence_region(
            self.frame, x[:, 0:3], w, self.settings.presence_probability_pct
        )
        relative: list[RelativeKinematics] = []
        for row in self.last_rows:
            rel = ground - row.velocity
            rel_speed = float(w @ np.linalg.norm(rel, axis=1)) / KNOT_TO_MPS
            slant = float(w @ np.linalg.norm(x[:, 0:3] - row.position, axis=1)) / YD_TO_M
            relative.append(
                RelativeKinematics(
                    observer_id=row.observer_id,
                    relative_speed_kt=rel_speed,
                    slant_range_yd=slant,
                    detected=row.detected,
                )
            )
        detecting = [item for item in relative if item.detected]
        coasting = self.last_detect_tick is None or self.tick - self.last_detect_tick > 0
        if region.disconnected:
            status = "AMBIGUOUS"
        elif uncertainty.horizontal_major_yd < 0.25 * self.settings.max_slant_range_yd:
            status = "TRACKING"
        else:
            status = "LOW_CONFIDENCE"
        if coasting:
            status = f"COASTING_{status}"
        position_basis = "posterior mean"
        position_mean = mean
        if region.disconnected:
            labels = pf.clusters()
            masses = {lab: float(w[labels == lab].sum()) for lab in np.unique(labels)}
            dominant = max(masses, key=masses.get)
            members = labels == dominant
            position_mean = (w[members] @ x[members]) / max(w[members].sum(), 1e-12)
            position_basis = f"dominant mode ({masses[dominant] * 100:.0f}% of mass)"
        base_meta["position_basis"] = position_basis
        geo = self.frame.to_geo(*position_mean[0:3])
        self._update_smoothed()
        common = {
            "tick": self.tick,
            "observability_status": status,
            "current_position": geo,
            "depth_ft": geo.depth_ft,
            "relative_speed_kt": (
                float(np.mean([item.relative_speed_kt for item in detecting])) if detecting else None
            ),
            "ground_speed_kt": g_mean / KNOT_TO_MPS,
            "through_water_speed_kt": u_mean / KNOT_TO_MPS,
            "hdg_deg": hdg,
            "cog_deg": cog,
            "vertical_rate_fps": float(mean[5] / FT_TO_M),
            "source_bias_hz": float(mean[6]),
            "uncertainty": uncertainty,
            "presence_region": region,
            "relative": relative,
        }
        live = self._track_point()
        online = TrackEstimate(
            mode=EstimateMode.ONLINE,
            track=[*self.online_track, live],
            metadata={**base_meta, "past_track": "not updated (filtered values frozen)",
                      **{k: float(v) for k, v in self.update_info.items()}},
            **common,
        )
        smoothed_points = [self.smoothed_track[key] for key in sorted(self.smoothed_track)]
        smoothed = TrackEstimate(
            mode=EstimateMode.SMOOTHED,
            track=[*smoothed_points, live],
            metadata={
                **base_meta,
                "past_track": "updated by fixed-lag smoothing within the recompute window",
            },
            **common,
        )
        return EstimatorOutput(
            tick=self.tick,
            estimates=[online, smoothed],
            cpa=cpa,
            current=self._current_estimate(),
            lloyd=self.lloyd_result,
        )
