"""Linear (affine) current-field estimation from observer drift.

Observers move only because of the water, at the same velocity as the surrounding water, and
their time/position/depth are known exactly. Successive fixes therefore give exact samples of
the local current. Within a limited period and area the field is assumed linear:

    v(p) = a + G (p - p_ref)

with one coefficient set shared by every observer inside the detection area. The period is the
detectable time T = R_max / V_target (detectable distance / target speed).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CurrentFit:
    base: np.ndarray  # a, m/s at p_ref
    gradient: np.ndarray  # G, (m/s)/m
    reference: np.ndarray  # p_ref, local metres
    observer_count: int
    sample_count: int
    window_seconds: float
    residual_mps: float

    def velocity(self, points: np.ndarray) -> np.ndarray:
        """Vectorized current at (N,3) local points."""
        return self.base + (points - self.reference) @ self.gradient.T

    @classmethod
    def zero(cls) -> CurrentFit:
        return cls(np.zeros(3), np.zeros((3, 3)), np.zeros(3), 0, 0, 0.0, 0.0)


class CurrentFieldEstimator:
    def __init__(self, history_seconds: int = 10800, sample_stride_s: int = 5) -> None:
        self.history_seconds = history_seconds
        self.sample_stride_s = sample_stride_s
        self._fixes: dict[str, deque[tuple[int, np.ndarray]]] = {}

    def add_fix(self, observer_id: str, tick: int, point: np.ndarray) -> None:
        fixes = self._fixes.setdefault(observer_id, deque())
        if fixes and fixes[-1][0] >= tick:
            return
        fixes.append((tick, point.copy()))
        while fixes and tick - fixes[0][0] > self.history_seconds:
            fixes.popleft()

    def observer_velocity(self, observer_id: str) -> np.ndarray | None:
        """Exact drift (= water) velocity from the two most recent fixes."""
        fixes = self._fixes.get(observer_id)
        if not fixes or len(fixes) < 2:
            return None
        (t0, p0), (t1, p1) = fixes[-2], fixes[-1]
        return (p1 - p0) / max(t1 - t0, 1)

    def last_fix(self, observer_id: str) -> tuple[int, np.ndarray] | None:
        fixes = self._fixes.get(observer_id)
        return fixes[-1] if fixes else None

    def samples(
        self, now: int, window_s: float, observer_ids: set[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, int]:
        points, velocities, used = [], [], 0
        for observer_id, fixes in self._fixes.items():
            if observer_ids is not None and observer_id not in observer_ids:
                continue
            series = [item for item in fixes if now - item[0] <= window_s]
            if len(series) < 2:
                continue
            used += 1
            step = max(1, self.sample_stride_s)
            for index in range(len(series) - 1, 0, -step):
                (t0, p0), (t1, p1) = series[index - 1], series[index]
                dt = max(t1 - t0, 1)
                points.append(0.5 * (p0 + p1))
                velocities.append((p1 - p0) / dt)
        if not points:
            return np.zeros((0, 3)), np.zeros((0, 3)), 0
        return np.asarray(points), np.asarray(velocities), used

    def fit(
        self,
        now: int,
        window_s: float,
        observer_ids: set[str] | None = None,
        ridge: float = 1e-6,
    ) -> CurrentFit:
        points, velocities, used = self.samples(now, window_s, observer_ids)
        if len(points) == 0 and observer_ids is not None:
            points, velocities, used = self.samples(now, window_s, None)
        if len(points) == 0:
            return CurrentFit.zero()
        reference = points.mean(axis=0)
        centred = points - reference
        design = np.hstack([np.ones((len(points), 1)), centred])  # (n, 4)
        # Ridge on the gradient terms only; scale so ridge is dimensionless w.r.t. spread.
        spread = max(float(np.mean(np.sum(centred**2, axis=1))), 1.0)
        penalty = np.diag([0.0, 1.0, 1.0, 1.0]) * ridge * spread * len(points)
        # Rank-deficient geometry (e.g. one observer, observers at one depth) is handled by
        # the ridge: unobservable gradient directions shrink to zero (uniform current).
        penalty += np.diag([0.0, 1.0, 1.0, 1.0]) * 1e-12 * len(points)
        normal = design.T @ design + penalty
        coeffs = np.linalg.solve(normal, design.T @ velocities)  # (4, 3)
        base = coeffs[0]
        gradient = coeffs[1:].T  # (3,3): v_i = base_i + sum_j G_ij dp_j
        residual = velocities - design @ coeffs
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        return CurrentFit(base, gradient, reference, used, len(points), float(window_s), rms)
