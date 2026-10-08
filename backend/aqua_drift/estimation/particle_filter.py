"""Regularized particle filter for Doppler-only 3-D target motion analysis.

State per particle (local frame, metres / seconds):

    x = [e, n, d, u_e, u_n, u_d, b]

* (e, n, d)       position, d = depth (positive down)
* (u_e, u_n, u_d) velocity THROUGH THE WATER (constant course / speed base motion)
* b               common, time-invariant source-frequency recognition bias [Hz]

Ground velocity = through-water velocity + estimated current at the particle position.

Measurements per observer j and epoch (1 s, synchronized):
* detected  -> received frequency f_j = f0 (1 - rdot_j / c),  f0 = f_recognized - b
* detected / not detected -> the target is / is not inside the common max slant range
  (no missed detections inside the range, so non-detection is information too).

Why a particle filter rather than a Kalman filter: with Doppler only the posterior is strongly
non-Gaussian (ring / mirror ambiguities about each observer's drift line, hard range-gate
constraints), which an EKF/UKF cannot represent. See docs/estimation-methods.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from aqua_drift.estimation.current_fit import CurrentFit
from aqua_drift.estimation.frame import KNOT_TO_MPS, YD_TO_M
from aqua_drift.estimation.move import (
    Epoch,
    EpochStore,
    GaussianMixture,
    MoveParams,
    Window,
    find_modes,
    window_loglik,
)


@dataclass
class ObservationRow:
    observer_id: str
    position: np.ndarray  # local metres
    velocity: np.ndarray  # m/s (= water velocity at the observer)
    detected: bool
    frequency: float | None
    recognized: float
    bearing: float | None = None  # radians, horizontal true bearing observer -> target


class DopplerParticleFilter:
    STATE_DIM = 7

    def __init__(
        self,
        particle_count: int,
        max_range_m: float,
        sound_speed: float,
        model_sigma_hz: float,
        bias_sigma_hz: float,
        max_speed_mps: float,
        max_depth_m: float,
        accel_h: float,
        accel_v: float,
        gate_softness_m: float,
        seed: int = 7,
        maneuver_fraction: float = 0.1,
        maneuver_accel: float = 0.12,
        maneuver_vertical: float = 0.15,
        move_min_window_s: int = 60,
        move_mismatch_chi2: float = 4.0,
        use_bearing: bool = True,
        bearing_sigma_rad: float = math.radians(15.0),
    ) -> None:
        self.n = particle_count
        self.max_range = max_range_m
        self.c = sound_speed
        self.sigma_f = model_sigma_hz
        self.bias_sigma = bias_sigma_hz
        self.max_speed = max_speed_mps
        self.max_depth = max_depth_m
        self.accel_h = accel_h
        self.accel_v = accel_v
        self.gate_soft = gate_softness_m
        self.maneuver_fraction = maneuver_fraction
        self.maneuver_accel = maneuver_accel
        self.maneuver_vertical = maneuver_vertical
        self.move_min_window_s = move_min_window_s
        self.move_mismatch_chi2 = move_mismatch_chi2
        self.use_bearing = use_bearing
        self.bearing_sigma = bearing_sigma_rad
        self.move_window_eff: float | None = None
        self.maneuver_cut_tick: int | None = None
        self.maneuver_detected_tick: int | None = None  # last time a maneuver was detected
        self.depth_fix: tuple[float, float, int] | None = None  # (depth m, sigma m, tick)
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros((self.n, self.STATE_DIM))
        self.w = np.full(self.n, 1.0 / self.n)
        self.ground_v = np.zeros((self.n, 3))
        self.initialized = False
        self.tick = 0
        # fixed-lag history (positions + ground velocities) for past-track updating
        self.hist = np.zeros((0, self.n, 6), dtype=np.float32)
        self.hist_ticks: list[int] = []
        self.hist_stride = 1
        self.hist_slots = 0
        self.epochs = EpochStore(900)
        self._modes: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._modes_tick = 0
        self.move_info: dict[str, float] = {}

    # ------------------------------------------------------------------ history
    def configure_history(self, window_s: int, slots: int) -> None:
        stride = max(1, math.ceil(window_s / max(slots, 1)))
        count = max(2, math.ceil(window_s / stride) + 1)
        if stride != self.hist_stride or count != self.hist_slots:
            self.hist_stride = stride
            self.hist_slots = count
            self.hist = np.zeros((0, self.n, 6), dtype=np.float32)
            self.hist_ticks = []

    def _push_history(self) -> None:
        if self.tick % self.hist_stride != 0:
            return
        row = np.concatenate([self.x[:, 0:3], self.ground_v], axis=1).astype(np.float32)[None]
        self.hist = np.concatenate([self.hist, row], axis=0)
        self.hist_ticks.append(self.tick)
        if len(self.hist_ticks) > self.hist_slots:
            drop = len(self.hist_ticks) - self.hist_slots
            self.hist = self.hist[drop:]
            self.hist_ticks = self.hist_ticks[drop:]

    # ------------------------------------------------------------------ initialization
    def initialize(
        self,
        tick: int,
        rows: list[ObservationRow],
        current: CurrentFit,
        previously_seen: set[str],
    ) -> bool:
        """Sample the prior from the first detection epoch.

        For an observer that was already watching (it reported a non-detection one epoch
        earlier) the target must have just crossed its range sphere, so the prior is a thin
        shell; otherwise the full sphere. Velocity is then projected so the predicted Doppler
        matches the measured one for the seeding observer.
        """
        detecting = [row for row in rows if row.detected and row.frequency is not None]
        if not detecting:
            return False
        anchor = min(detecting, key=lambda row: 0 if row.observer_id in previously_seen else 1)
        shell = anchor.observer_id in previously_seen
        inner = max(0.0, self.max_range - (self.max_speed + 3.0) * 2.0) if shell else 0.0
        accepted: list[np.ndarray] = []
        need = self.n
        for _ in range(200):
            m = need * 6
            direction = self.rng.normal(size=(m, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            radius = np.cbrt(self.rng.uniform(inner**3, self.max_range**3, size=m))
            pos = anchor.position + direction * radius[:, None]
            ok = (pos[:, 2] >= 0.0) & (pos[:, 2] <= self.max_depth)
            for row in rows:
                dist = np.linalg.norm(pos - row.position, axis=1)
                ok &= (dist <= self.max_range) if row.detected else (dist > self.max_range)
            pos = pos[ok]
            if len(pos) == 0:
                continue
            speed = self.max_speed * np.sqrt(self.rng.uniform(0, 1, size=len(pos)))
            heading = self.rng.uniform(0, 2 * np.pi, size=len(pos))
            water = np.stack(
                [speed * np.sin(heading), speed * np.cos(heading),
                 self.rng.normal(0, 0.05, size=len(pos))], axis=1
            )
            bias = self.rng.normal(0, self.bias_sigma, size=len(pos)) if self.bias_sigma else (
                np.zeros(len(pos))
            )
            # Project the radial relative velocity onto the measured Doppler of the anchor.
            f0 = anchor.recognized - bias
            rdot_needed = self.c * (1.0 - anchor.frequency / f0)
            los = pos - anchor.position
            unit = los / np.maximum(np.linalg.norm(los, axis=1, keepdims=True), 1e-6)
            ground = water + current.velocity(pos)
            rel = ground - anchor.velocity
            correction = (rdot_needed - np.sum(rel * unit, axis=1))[:, None] * unit
            water = water + correction
            fast = np.linalg.norm(water[:, :2], axis=1) <= self.max_speed
            states = np.hstack([pos, water, bias[:, None]])[fast]
            accepted.append(states)
            need -= len(states)
            if need <= 0:
                break
        if not accepted:
            return False
        states = np.vstack(accepted)
        if len(states) < self.n:
            states = states[self.rng.integers(0, len(states), size=self.n)]
        self.x = states[: self.n].copy()
        self.w = np.full(self.n, 1.0 / self.n)
        self.tick = tick
        self.ground_v = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])
        self.initialized = True
        self.hist = np.zeros((0, self.n, 6), dtype=np.float32)
        self.hist_ticks = []
        return True

    # ------------------------------------------------------------------ predict / update
    def predict(self, dt: float, current: CurrentFit) -> None:
        if dt <= 0:
            return
        sh = self.accel_h * math.sqrt(dt)
        sv = self.accel_v * math.sqrt(dt)
        self.x[:, 3] += self.rng.normal(0, sh, self.n) if sh else 0.0
        self.x[:, 4] += self.rng.normal(0, sh, self.n) if sh else 0.0
        self.x[:, 5] += self.rng.normal(0, sv, self.n) if sv else 0.0
        if self.maneuver_fraction > 0:
            # maneuver mode (particle-form IMM): course / speed / depth-rate changes
            jump = self.rng.uniform(size=self.n) < self.maneuver_fraction
            k = int(jump.sum())
            if k:
                self.x[jump, 3:5] += self.rng.normal(0, self.maneuver_accel * dt, (k, 2))
                self.x[jump, 5] += self.rng.normal(0, self.maneuver_vertical, k)
                self.x[jump, 5] = np.clip(self.x[jump, 5], -1.0, 1.0)
        self._limit_speed()
        # midpoint integration through the (linear) current field
        ground0 = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])
        mid = self.x[:, 0:3] + 0.5 * dt * ground0
        ground = self.x[:, 3:6] + current.velocity(mid)
        self.x[:, 0:3] += ground * dt
        self._reflect_depth()
        self.ground_v = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])
        self.tick += round(dt)

    def _limit_speed(self) -> None:
        horizontal = np.linalg.norm(self.x[:, 3:5], axis=1)
        over = horizontal > self.max_speed
        if np.any(over):
            self.x[over, 3:5] *= (self.max_speed / horizontal[over])[:, None]

    def _reflect_depth(self) -> None:
        d = self.x[:, 2]
        shallow = d < 0
        self.x[shallow, 2] = -d[shallow]
        self.x[shallow, 5] = np.abs(self.x[shallow, 5])
        deep = self.x[:, 2] > self.max_depth
        self.x[deep, 2] = 2 * self.max_depth - self.x[deep, 2]
        self.x[deep, 5] = -np.abs(self.x[deep, 5])

    def log_likelihood(self, rows: list[ObservationRow]) -> np.ndarray:
        x = self.x
        pos = x[:, 0:3]
        ground = self.ground_v
        total = np.zeros(len(x))
        for row in rows:
            los = pos - row.position
            dist = np.linalg.norm(los, axis=1)
            margin = (self.max_range - dist) / self.gate_soft
            if row.detected:
                total += -np.logaddexp(0.0, -margin)  # log sigmoid(margin)
                if row.frequency is not None:
                    unit = los / np.maximum(dist, 1e-6)[:, None]
                    rdot = np.sum((ground - row.velocity) * unit, axis=1)
                    f0 = row.recognized - x[:, 6]
                    predicted = f0 * (1.0 - rdot / self.c)
                    total += -0.5 * ((predicted - row.frequency) / self.sigma_f) ** 2
                if self.use_bearing and row.bearing is not None:
                    predicted_b = np.arctan2(los[:, 0], los[:, 1])
                    diff = (predicted_b - row.bearing + np.pi) % (2 * np.pi) - np.pi
                    total += -0.5 * (diff / self.bearing_sigma) ** 2
            else:
                total += -np.logaddexp(0.0, margin)  # log sigmoid(-margin)
        return total

    def update(self, rows: list[ObservationRow], current: CurrentFit) -> dict[str, float]:
        """Progressive-correction update: split the likelihood into tempered stages so the
        (error-free, hence very sharp) Doppler likelihood never collapses the particle set."""
        loglik = self.log_likelihood(rows)
        remaining = 1.0
        stages = 0
        while remaining > 1e-9 and stages < 8:
            stages += 1
            beta = self._choose_beta(loglik, remaining)
            logw = np.log(np.maximum(self.w, 1e-300)) + beta * loglik
            logw -= logw.max()
            w = np.exp(logw)
            self.w = w / w.sum()
            remaining -= beta
            if remaining > 1e-9 or self.ess() < 0.5 * self.n:
                index = self._systematic_resample()
                self._apply_resample(index)
                self._roughen(current)
                if remaining > 1e-9:
                    loglik = self.log_likelihood(rows)
        self._push_history()
        return {"stages": stages, "ess": self.ess()}

    def _choose_beta(self, loglik: np.ndarray, remaining: float) -> float:
        """Largest beta <= remaining keeping ESS >= n/3 (bisection)."""
        target = self.n / 3.0
        base = np.log(np.maximum(self.w, 1e-300))

        def ess_for(beta: float) -> float:
            lw = base + beta * loglik
            lw -= lw.max()
            w = np.exp(lw)
            w /= w.sum()
            return 1.0 / np.sum(w * w)

        if ess_for(remaining) >= target:
            return remaining
        lo, hi = 0.0, remaining
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if ess_for(mid) >= target:
                lo = mid
            else:
                hi = mid
        return max(lo, remaining * 1e-3)

    def ess(self) -> float:
        return float(1.0 / np.sum(self.w * self.w))

    def _systematic_resample(self) -> np.ndarray:
        positions = (self.rng.uniform() + np.arange(self.n)) / self.n
        cumulative = np.cumsum(self.w)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions)

    def _apply_resample(self, index: np.ndarray) -> None:
        self.x = self.x[index]
        self.ground_v = self.ground_v[index]
        self.w = np.full(self.n, 1.0 / self.n)
        if len(self.hist_ticks):
            self.hist = self.hist[:, index]

    def clusters(self) -> np.ndarray:
        """Label particles by horizontally connected groups (keeps separate modes apart)."""
        pos = self.x[:, 0:2]
        cell = max(200.0, 0.08 * self.max_range)
        lo = pos.min(axis=0)
        idx = np.floor((pos - lo) / cell).astype(int)
        shape = idx.max(axis=0) + 1
        if shape[0] * shape[1] > 4_000_000:
            return np.zeros(self.n, dtype=int)
        grid = np.zeros(shape, dtype=bool)
        grid[idx[:, 0], idx[:, 1]] = True
        labels, _ = ndimage.label(grid, structure=np.ones((3, 3)))
        return labels[idx[:, 0], idx[:, 1]] - 1

    def _roughen(self, current: CurrentFit) -> None:
        """Regularized PF kernel jitter (Musso, Oudjane & Le Gland), applied per cluster so
        that the kernel bandwidth reflects each mode, not the spread between modes."""
        d = self.STATE_DIM
        h = (4.0 / (self.n * (d + 2))) ** (1.0 / (d + 4))
        floor = np.diag([1.0, 1.0, 0.3, 1e-4, 1e-4, 1e-5, 1e-6])
        labels = self.clusters()
        noise = np.zeros_like(self.x)
        for label in np.unique(labels):
            members = labels == label
            count = int(members.sum())
            if count < 2:
                cov = floor * 100.0
            else:
                cov = np.cov(self.x[members].T) + floor
                cov = 0.5 * (cov + cov.T)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-9)))
            noise[members] = self.rng.normal(size=(count, d)) @ chol.T
        self.x += noise * h
        self._limit_speed()
        self._reflect_depth()
        if self.bias_sigma == 0:
            self.x[:, 6] = 0.0
        self.ground_v = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])

    # ------------------------------------------------------------------ resample-move
    def record_epoch(self, tick: int, rows: list[ObservationRow], window_s: int) -> None:
        if not rows:
            return
        self.epochs.window_s = window_s
        self.epochs.add(
            Epoch(
                tick=tick,
                pos=np.array([r.position for r in rows]),
                vel=np.array([r.velocity for r in rows]),
                det=np.array([r.detected and r.frequency is not None for r in rows]),
                freq=np.array([r.frequency if r.frequency is not None else np.nan for r in rows]),
                rec=np.array([r.recognized for r in rows]),
                brg=np.array([
                    r.bearing if (r.bearing is not None and self.use_bearing) else np.nan
                    for r in rows
                ]),
                ids=tuple(r.observer_id for r in rows),
            )
        )

    def _move_params(self) -> MoveParams:
        return MoveParams(
            max_range=self.max_range,
            gate_soft=self.gate_soft,
            c=self.c,
            sigma_f=self.sigma_f,
            bias_sigma=self.bias_sigma,
            max_speed=self.max_speed,
            max_depth=self.max_depth,
            bearing_sigma=self.bearing_sigma,
            depth_fix=self._active_depth_fix(),
        )

    # ------------------------------------------------------------------ external depth fixes
    DEPTH_FIX_MAX_AGE_S = 60

    def _active_depth_fix(self) -> tuple[float, float] | None:
        fix = getattr(self, "depth_fix", None)
        if fix is None or self.tick - fix[2] > self.DEPTH_FIX_MAX_AGE_S:
            return None
        return fix[0], fix[1]

    def set_depth_fix(self, depth_m: float | None, sigma_m: float | None = None) -> None:
        """Latest depth measurement (Lloyd's mirror); used in the resample-move target."""
        self.depth_fix = None if depth_m is None or sigma_m is None else (depth_m, sigma_m, self.tick)

    def apply_depth_measurement(self, depth_m: float, sigma_m: float, current: CurrentFit) -> float:
        """Sequential update with a depth measurement (tempered like the Doppler update so the
        particle set never collapses). Returns the ESS afterwards."""
        loglik = -0.5 * ((self.x[:, 2] - depth_m) / sigma_m) ** 2
        remaining = 1.0
        for _ in range(8):
            beta = self._choose_beta(loglik, remaining)
            logw = np.log(np.maximum(self.w, 1e-300)) + beta * loglik
            logw -= logw.max()
            w = np.exp(logw)
            self.w = w / w.sum()
            remaining -= beta
            if remaining > 1e-9 or self.ess() < 0.5 * self.n:
                self._apply_resample(self._systematic_resample())
                self._roughen(current)
                loglik = -0.5 * ((self.x[:, 2] - depth_m) / sigma_m) ** 2
            if remaining <= 1e-9:
                break
        return self.ess()

    def move(self, current: CurrentFit, epochs: int = 60, starts: int = 8) -> dict[str, float]:
        """Optimization-assisted Metropolis-Hastings rejuvenation (see estimation/move.py).

        The window is adaptive: if even the best constant-velocity fit is inconsistent with
        the data (a maneuver inside the window), the window is shortened; it then grows back
        towards the configured length while the fit stays consistent."""
        full = float(self.epochs.window_s)
        window = full
        if self.maneuver_cut_tick is not None:
            window = min(full, self.tick - self.maneuver_cut_tick)
        self.move_window_eff = float(np.clip(window, self.move_min_window_s, full))
        sample = self.epochs.sample(epochs, int(self.tick - self.move_window_eff))
        if not sample or not any(e.det.any() for e, w in sample if w > 0):
            return {}
        now = self.tick
        prm = self._move_params()
        lo = self.x[:, 0:3].min(axis=0)
        hi = self.x[:, 0:3].max(axis=0)
        regular_ticks = sorted(e.tick for e, w in sample if w > 0)
        coarse = set(regular_ticks[::4])

        def relevant(epoch: Epoch) -> np.ndarray:
            """Gate-only (non-detecting) rows are kept on a coarser grid (and at every
            detection transition) and only for observers the particle cloud could have been
            within max range of at that time."""
            if not epoch.transition and epoch.tick not in coarse:
                return np.zeros(len(epoch.det), dtype=bool)
            reach = self.max_range + (self.max_speed + 2.0) * (now - epoch.tick) + 300.0
            nearest = np.clip(epoch.pos, lo, hi)
            return np.linalg.norm(epoch.pos - nearest, axis=1) <= reach

        win = Window.build(now, sample, relevant)
        if self.ess() < 0.999 * self.n:
            self._apply_resample(self._systematic_resample())
        ll = window_loglik(self.x, win, current, prm)
        finite = np.isfinite(ll)
        order = np.argsort(np.where(finite, ll, -np.inf))[::-1]
        chosen: list[int] = []
        for index in order:
            if len(chosen) >= max(starts - 1, 1):
                break
            if all(np.linalg.norm(self.x[index, 0:3] - self.x[j, 0:3]) > 300.0 for j in chosen):
                chosen.append(int(index))
        start_states = [self.x[chosen]]
        if self._modes:  # warm start: previous modes propagated to now
            previous = np.array([m[0] for m in self._modes])
            dt = now - self._modes_tick
            previous[:, 0:3] += (previous[:, 3:6] + current.velocity(previous[:, 0:3])) * dt
            start_states.append(previous)
        start_states.append(self.x[self.rng.integers(0, self.n, size=1)])
        modes = find_modes(np.vstack(start_states), win, current, prm)
        self._modes = [(m[0].copy(), m[1], m[2]) for m in modes]
        self._modes_tick = now
        doppler_rows = float(np.sum(win.w_dop > 0) * max(win.w_dop.max(), 1.0))
        chi2 = 2.0 * modes[0][2] / max(doppler_rows, 1.0) if modes else 0.0
        if chi2 > self.move_mismatch_chi2:
            # maneuver inside the window: data older than half the window are not used
            self.maneuver_cut_tick = int(now - max(self.move_min_window_s, 0.5 * self.move_window_eff))
            self.maneuver_detected_tick = int(now)
        accepted_ind = 0.0
        if modes:
            mixture = GaussianMixture(modes)
            proposal = mixture.sample(self.n, self.rng)
            ll_prop = window_loglik(proposal, win, current, prm)
            with np.errstate(invalid="ignore"):
                log_alpha = (ll_prop - ll) + (mixture.logpdf(self.x) - mixture.logpdf(proposal))
            log_alpha = np.where(np.isfinite(ll), log_alpha, np.where(np.isfinite(ll_prop), 0.0, -np.inf))
            accept = np.log(self.rng.uniform(size=self.n)) < np.nan_to_num(log_alpha, nan=-np.inf)
            self.x[accept] = proposal[accept]
            ll[accept] = ll_prop[accept]
            accepted_ind = float(accept.mean())
        # random-walk MH within each cluster
        labels = self.clusters()
        step = np.zeros_like(self.x)
        for label in np.unique(labels):
            members = labels == label
            count = int(members.sum())
            cov = (np.cov(self.x[members].T) if count > 2 else np.zeros((7, 7))) + np.diag(
                [4.0, 4.0, 1.0, 1e-4, 1e-4, 1e-5, 1e-6]
            )
            try:
                chol = np.linalg.cholesky(0.5 * (cov + cov.T))
            except np.linalg.LinAlgError:
                chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-9)))
            step[members] = (self.rng.normal(size=(count, 7)) @ chol.T) * (0.6 / np.sqrt(7))
        proposal = self.x + step
        if self.bias_sigma == 0:
            proposal[:, 6] = 0.0
        ll_prop = window_loglik(proposal, win, current, prm)
        with np.errstate(invalid="ignore"):
            delta = np.nan_to_num(ll_prop - ll, nan=-np.inf)
        accept = np.log(self.rng.uniform(size=self.n)) < delta
        self.x[accept] = proposal[accept]
        self.ground_v = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])
        self.move_info = {
            "move_modes": float(len(modes)),
            "move_accept_independent": accepted_ind,
            "move_accept_random_walk": float(accept.mean()),
            "move_window_s": float(self.move_window_eff),
            "move_fit_chi2": float(chi2),
        }
        return self.move_info

    # ------------------------------------------------------------------ summaries
    def mean_state(self) -> np.ndarray:
        return self.w @ self.x

    def smoothed_history(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Fixed-lag smoothed (tick, mean[6], cov_pos[3x3]) using ancestor-traced paths."""
        output = []
        for slot, tick in enumerate(self.hist_ticks):
            values = self.hist[slot].astype(np.float64)
            mean = self.w @ values
            diff = values[:, 0:3] - mean[0:3]
            cov = (diff * self.w[:, None]).T @ diff
            output.append((tick, mean, cov))
        return output

    def max_range_yd(self) -> float:
        return self.max_range / YD_TO_M

    @staticmethod
    def kt(value_mps: float) -> float:
        return value_mps / KNOT_TO_MPS
