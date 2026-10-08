"""Target depth from the Lloyd's mirror interference of the received level (optional).

The direct path and the surface-reflected path (image source above the surface, phase
reversed) interfere; the received level of the tonal therefore rises and falls as the
geometry changes. With the receiver depth known exactly and the horizontal range taken from
the Doppler track, the pattern depends essentially on the target depth only:

    path difference  dR = r2 - r1 ~ 2 zs zr / R       (r1 direct, r2 reflected)
    level(t) = SL + 20 log10 | exp(i k r1)/r1 - mu exp(i k r2)/r2 |

For every observer the depth is found by a grid search over the depth (and a few surface
reflection magnitudes mu), with the unknown source level removed as a common offset. The
profile of the misfit gives the statistical uncertainty (corrected for correlated level
fluctuations); an assumed sound-speed model error (a fraction of the depth, since dR scales
with the depth) is added. Observers are combined by inverse variance.

This is the heavy part (grid x samples x observers), so it runs only while the operator has
the Lloyd's mirror calculation switched on.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aqua_drift.models import LloydDepthEstimate, LloydObserverFit

FT_TO_M = 0.3048
REFLECTIONS = (0.3, 0.6, 0.9)  # coherent surface reflection magnitudes tried by the fit


@dataclass
class LloydFitSettings:
    sound_speed: float
    max_depth_m: float
    depth_step_m: float
    min_samples: int
    window_s: float
    model_error_frac: float
    noise_correlation_s: float


@dataclass
class _ObserverFit:
    observer_id: str
    status: str
    samples: int
    fringes: float
    depth_m: float | None = None
    stat_sigma_m: float | None = None
    reflection: float | None = None


class LloydDepthEstimator:
    """Keeps a window of (tick, observer position, level, frequency) per observer and fits the
    target depth to each observer's interference pattern."""

    def __init__(self) -> None:
        self.samples: dict[str, deque] = {}

    def clear(self) -> None:
        self.samples.clear()

    def add(self, observer_id: str, tick: int, position: np.ndarray, level_db: float, frequency_hz: float) -> None:
        self.samples.setdefault(observer_id, deque()).append(
            (int(tick), float(position[0]), float(position[1]), float(position[2]), float(level_db), float(frequency_hz))
        )

    def prune(self, now: int, keep_s: float) -> None:
        for observer_id in list(self.samples):
            rows = self.samples[observer_id]
            while rows and rows[0][0] < now - keep_s:
                rows.popleft()
            if not rows:
                del self.samples[observer_id]

    # ------------------------------------------------------------------ fitting
    def fit_observer(
        self,
        observer_id: str,
        rows: np.ndarray,
        track: Callable[[np.ndarray], np.ndarray],
        settings: LloydFitSettings,
    ) -> _ObserverFit:
        n = len(rows)
        if n < settings.min_samples:
            return _ObserverFit(observer_id, "FEW_SAMPLES", n, 0.0)
        ticks = rows[:, 0]
        obs = rows[:, 1:4]
        level = rows[:, 4]
        k = 2.0 * math.pi * rows[:, 5] / settings.sound_speed  # (N,)
        target = track(ticks)
        horizontal = np.hypot(target[:, 0] - obs[:, 0], target[:, 1] - obs[:, 1])  # (N,)
        zr = np.maximum(obs[:, 2], 0.0)
        mid_depth = 0.5 * settings.max_depth_m
        r1_mid = np.sqrt(horizontal**2 + (mid_depth - zr) ** 2)
        r2_mid = np.sqrt(horizontal**2 + (mid_depth + zr) ** 2)
        wavelength = 2.0 * math.pi / float(np.mean(k))
        fringes = float(np.ptp(r2_mid - r1_mid) / wavelength)
        if fringes < 1.0:
            return _ObserverFit(observer_id, "NO_FRINGES", n, fringes)

        def costs(depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            zs = depths[:, None]
            r1 = np.sqrt(horizontal[None, :] ** 2 + (zs - zr[None, :]) ** 2)  # (Z,N)
            r2 = np.sqrt(horizontal[None, :] ** 2 + (zs + zr[None, :]) ** 2)
            direct = np.exp(1j * k[None, :] * r1) / r1
            reflected = np.exp(1j * k[None, :] * r2) / r2
            best_cost = np.full(len(depths), np.inf)
            best_mu = np.zeros(len(depths))
            for mu in REFLECTIONS:
                model = 20.0 * np.log10(np.maximum(np.abs(direct - mu * reflected), 1e-12))
                resid = level[None, :] - model
                cost = np.sum(resid * resid, axis=1) - np.sum(resid, axis=1) ** 2 / n  # source level removed
                better = cost < best_cost
                best_cost[better] = cost[better]
                best_mu[better] = mu
            return best_cost, best_mu

        # coarse grid over the whole depth range, then a fine grid around the best coarse depth
        coarse_step = max(settings.depth_step_m, 3.0)
        coarse = np.arange(coarse_step, settings.max_depth_m + 1e-9, coarse_step)
        coarse_cost, coarse_mu = costs(coarse)
        j = int(np.argmin(coarse_cost))
        fine = np.arange(max(coarse[j] - 2 * coarse_step, settings.depth_step_m),
                         min(coarse[j] + 2 * coarse_step, settings.max_depth_m) + 1e-9, settings.depth_step_m)
        fine_cost, fine_mu = costs(fine)
        depths = np.concatenate([coarse, fine])
        best_cost = np.concatenate([coarse_cost, fine_cost])
        best_mu = np.concatenate([coarse_mu, fine_mu])
        order = np.argsort(depths)
        depths, best_cost, best_mu = depths[order], best_cost[order], best_mu[order]
        mid = int(np.argmin(np.abs(depths - mid_depth)))
        # no-interference model (direct path only, spreading loss) for the significance test
        flat = level - 20.0 * np.log10(1.0 / np.maximum(np.sqrt(horizontal**2 + (depths[mid] - zr) ** 2), 1.0))
        flat_cost = float(np.sum(flat * flat) - np.sum(flat) ** 2 / n)
        i = int(np.argmin(best_cost))
        s_min = float(best_cost[i])
        sigma2 = max(s_min / max(n - 3, 1), 1e-6)
        a = math.exp(-1.0 / settings.noise_correlation_s) if settings.noise_correlation_s > 0 else 0.0
        n_eff = n * (1.0 - a) / (1.0 + a)
        chi2 = (best_cost - s_min) / sigma2 * (n_eff / n)
        if (flat_cost - s_min) / sigma2 * (n_eff / n) < 25.0:
            return _ObserverFit(observer_id, "NO_PATTERN", n, fringes)
        inside = chi2 <= 1.0
        lo = i
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        hi = i
        while hi < len(depths) - 1 and inside[hi + 1]:
            hi += 1
        stat_sigma = max(0.5 * (depths[hi] - depths[lo]), 0.5 * settings.depth_step_m)
        # another depth that fits nearly as well (a different fringe order) -> ambiguous
        outside = np.abs(depths - depths[i]) > max(4.0 * stat_sigma, 4.0 * settings.depth_step_m)
        ambiguous = bool(np.any(outside & (chi2 < 9.0)))
        return _ObserverFit(
            observer_id, "AMBIGUOUS" if ambiguous else "OK", n, fringes,
            float(depths[i]), float(stat_sigma), float(best_mu[i]),
        )

    def fit(
        self,
        now: int,
        track: Callable[[np.ndarray], np.ndarray],
        settings: LloydFitSettings,
    ) -> LloydDepthEstimate:
        started = time.perf_counter()
        fits: list[_ObserverFit] = []
        for observer_id, rows in sorted(self.samples.items()):
            window = np.array([r for r in rows if r[0] >= now - settings.window_s], dtype=float)
            if len(window) == 0:
                continue
            fits.append(self.fit_observer(observer_id, window, track, settings))
        good = [f for f in fits if f.status == "OK"]
        depth = sigma = None
        if good:
            weights = np.array([1.0 / f.stat_sigma_m**2 for f in good])
            values = np.array([f.depth_m for f in good])
            depth = float(np.sum(weights * values) / np.sum(weights))
            stat = math.sqrt(1.0 / float(np.sum(weights)))
            # observers disagreeing more than their statistics allow -> widen
            if len(good) > 1:
                spread = float(np.sqrt(np.sum(weights * (values - depth) ** 2) / np.sum(weights)))
                stat = max(stat, spread / math.sqrt(len(good)))
            sigma = math.sqrt(stat**2 + (settings.model_error_frac * depth) ** 2)

        def total_sigma(f: _ObserverFit) -> float | None:
            if f.stat_sigma_m is None or f.depth_m is None:
                return None
            return math.sqrt(f.stat_sigma_m**2 + (settings.model_error_frac * f.depth_m) ** 2) / FT_TO_M

        return LloydDepthEstimate(
            enabled=True,
            tick=now,
            status="OK" if good else ("WAITING" if not fits else "NO_RESULT"),
            depth_ft=None if depth is None else depth / FT_TO_M,
            sigma_ft=None if sigma is None else sigma / FT_TO_M,
            used_observers=len(good),
            observers=[
                LloydObserverFit(
                    observer_id=f.observer_id, status=f.status, samples=f.samples,
                    fringes=round(f.fringes, 2),
                    depth_ft=None if f.depth_m is None else f.depth_m / FT_TO_M,
                    sigma_ft=total_sigma(f), reflection=f.reflection,
                )
                for f in fits
            ],
            fit_ms=(time.perf_counter() - started) * 1000.0,
        )
