"""Closest-point-of-approach (CPA) analysis from a single observer's Doppler history.

With constant relative velocity the slant range is r(t) = sqrt(R^2 + V^2 (t - tc)^2) and

    f(t) = f0 * (1 - rdot / c),   rdot = V^2 (t - tc) / r(t)

* CPA time      tc : the received frequency equals the source frequency (rdot = 0). The
                     observer only knows the *recognized* frequency f0' = f0 + b, so a common,
                     time-invariant bias b shifts the estimated tc by about b / |df/dt|.
* Relative speed V : from the frequency change before and after CPA
                     (f(tc - tau) - f(tc + tau) -> 2 f0 V / c for large tau).
* CPA range     R  : from the slope at CPA, |df/dt| = f0 V^2 / (c R), i.e. R = f0 V^2/(c |df/dt|).
                     Errors come from the source-frequency error (b) and the speed error:
                     dR/R = db/f0 + 2 dV/V (plus the CPA-time shift through the curve).

The primary result follows that recognized-frequency method; its error is evaluated by
recomputing with f0' +/- sigma_b (frequency error) and from the speed resolution (speed error).
As a cross-check a full-curve least-squares fit with f0 free is also reported; because the
curve is antisymmetric about the true CPA, it also estimates the common bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from aqua_drift.estimation.frame import KNOT_TO_MPS, YD_TO_M
from aqua_drift.models import CpaResult


@dataclass
class _Pass:
    index: int
    ticks: list[int] = field(default_factory=list)
    freqs: list[float] = field(default_factory=list)
    recognized: float = 0.0
    crossing_tick: float | None = None
    ended: bool = False
    result: CpaResult | None = None
    finalized: bool = False


def doppler_curve(t: np.ndarray, f0: float, c: float, v: float, r: float, tc: float) -> np.ndarray:
    dt = t - tc
    rng = np.sqrt(r * r + (v * dt) ** 2)
    return f0 * (1.0 - (v * v * dt) / (c * rng))


def classical_cpa(
    ticks: np.ndarray, freqs: np.ndarray, f_rec: float, c: float, half_window: float
) -> tuple[float, float, float, float] | None:
    """Recognized-frequency method. Returns (tc, V m/s, R m, slope Hz/s) or None.

    tc    : zero crossing of f - f_rec
    slope : local linear fit of f around tc
    V, R  : from the symmetric frequency change D = f(tc - tau) - f(tc + tau) and the slope s:
            D = 2 s tau / sqrt(1 + s c tau^2 / (f0 R))  ->  R, then V = sqrt(s c R / f0)
    """
    tc = _crossing(ticks, freqs, f_rec)
    if tc is None:
        return None
    near = np.abs(ticks - tc) <= 10.0
    if near.sum() < 3:
        return None
    slope = float(np.polyfit(ticks[near], freqs[near], 1)[0])
    s = -slope
    if s <= 0:
        return None
    tau = min(tc - ticks.min(), ticks.max() - tc, half_window)
    if tau < 5.0:
        return None
    before = float(np.interp(tc - tau, ticks, freqs))
    after = float(np.interp(tc + tau, ticks, freqs))
    d = before - after
    ratio = (2.0 * s * tau / max(d, 1e-12)) ** 2 - 1.0
    if ratio <= 1e-9:
        # tau too short to see curvature: fall back to the asymptotic speed
        v = c * d / (2.0 * f_rec)
        r = f_rec * v * v / (c * s)
    else:
        r = s * c * tau * tau / (f_rec * ratio)
        v = math.sqrt(s * c * r / f_rec)
    return tc, v, r, slope


def free_fit(
    ticks: np.ndarray, freqs: np.ndarray, c: float, guess: tuple[float, float, float, float],
    model_sigma_hz: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Full-curve fit with the source frequency free: params [f0, V, R, tc]."""
    if len(ticks) < 8:
        return None

    def residual(params: np.ndarray) -> np.ndarray:
        f0, v, r, tc = params
        return (doppler_curve(ticks, f0, c, v, r, tc) - freqs) / model_sigma_hz

    f0, v, r, tc = guess
    try:
        sol = least_squares(
            residual,
            x0=np.array([f0, max(v, 0.05), max(r, 1.0), tc]),
            bounds=(
                [f0 * 0.9, 0.01, 0.5, ticks.min() - 3600],
                [f0 * 1.1, 60.0, 1e6, ticks.max() + 3600],
            ),
            x_scale=np.array([0.1, 1.0, max(r, 50.0), 30.0]),
        )
    except ValueError:
        return None
    cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    return sol.x, cov


class CpaAnalyzer:
    def __init__(self) -> None:
        self._passes: dict[str, list[_Pass]] = {}
        self._last_detected: dict[str, bool] = {}

    def add(self, observer_id: str, tick: int, detected: bool, f: float | None, f_rec: float) -> None:
        passes = self._passes.setdefault(observer_id, [])
        was_detected = self._last_detected.get(observer_id, False)
        self._last_detected[observer_id] = detected
        if not detected or f is None:
            if was_detected and passes:
                passes[-1].ended = True
            return
        if not was_detected or not passes or passes[-1].ended:
            passes.append(_Pass(index=len(passes), recognized=f_rec))
        current = passes[-1]
        if current.freqs and current.crossing_tick is None:
            prev = current.freqs[-1] - f_rec
            now = f - f_rec
            if prev > 0.0 >= now:  # closing -> opening
                frac = prev / (prev - now) if prev != now else 0.0
                current.crossing_tick = current.ticks[-1] + frac * (tick - current.ticks[-1])
        current.ticks.append(tick)
        current.freqs.append(f)
        current.recognized = f_rec

    def results(
        self,
        now: int,
        c: float,
        bias_sigma_hz: float,
        model_sigma_hz: float,
        half_window_s: int,
        min_post_samples: int,
    ) -> list[CpaResult]:
        output: list[CpaResult] = []
        for observer_id, passes in self._passes.items():
            for item in passes:
                if item.finalized and item.result is not None:
                    output.append(item.result)
                    continue
                if item.crossing_tick is None:
                    continue
                ticks = np.asarray(item.ticks, dtype=float)
                post = int(np.sum(ticks > item.crossing_tick))
                if post < min_post_samples and not item.ended:
                    continue
                final = item.ended or (ticks.max() - item.crossing_tick) >= half_window_s
                # Refit at most every 10 s while provisional.
                if item.result is not None and not final and now % 10 != 0:
                    output.append(item.result)
                    continue
                result = self._analyze(
                    observer_id, item, c, bias_sigma_hz, model_sigma_hz, half_window_s, final
                )
                if result is not None:
                    item.result = result
                    item.finalized = final
                    output.append(result)
        return output

    @staticmethod
    def _analyze(
        observer_id: str,
        item: _Pass,
        c: float,
        bias_sigma_hz: float,
        model_sigma_hz: float,
        half_window_s: int,
        final: bool,
    ) -> CpaResult | None:
        ticks = np.asarray(item.ticks, dtype=float)
        freqs = np.asarray(item.freqs, dtype=float)
        keep = np.abs(ticks - item.crossing_tick) <= half_window_s
        ticks, freqs = ticks[keep], freqs[keep]
        f_rec = item.recognized
        nominal = classical_cpa(ticks, freqs, f_rec, c, half_window_s)
        if nominal is None:
            return None
        tc, v, r, slope = nominal

        # Speed error: a frequency resolution of sigma_f on each end of the pre/post change
        # gives sigma_V ~ c * sqrt(2) sigma_f / (2 f0); R ~ V^2 so dR = 2 R dV / V.
        sigma_v = c * math.sqrt(2.0) * model_sigma_hz / (2.0 * f_rec)
        speed_r = 2.0 * r * sigma_v / max(v, 1e-6)

        # Source-frequency error: recompute with f_rec +/- sigma_b.
        bias_r = bias_t = 0.0
        if bias_sigma_hz > 0:
            alt = [
                classical_cpa(ticks, freqs, f_rec + sign * bias_sigma_hz, c, half_window_s)
                for sign in (-1.0, 1.0)
            ]
            if all(item_ is not None for item_ in alt):
                bias_t = abs(alt[1][0] - alt[0][0]) / 2.0
                bias_r = abs(alt[1][2] - alt[0][2]) / 2.0
            else:
                bias_t = bias_sigma_hz / max(abs(slope), 1e-9)
                bias_r = r * bias_sigma_hz / f_rec
        total_r = math.hypot(bias_r, speed_r)

        fitted = free_fit(ticks, freqs, c, (f_rec, v, r, tc), model_sigma_hz)
        fit_values: dict[str, float | None] = {
            "fit_source_frequency_hz": None,
            "fit_cpa_tick": None,
            "fit_cpa_slant_range_yd": None,
            "fit_relative_speed_kt": None,
        }
        if fitted is not None:
            (f0_fit, v_fit, r_fit, tc_fit), _ = fitted
            fit_values = {
                "fit_source_frequency_hz": float(f0_fit),
                "fit_cpa_tick": float(tc_fit),
                "fit_cpa_slant_range_yd": float(r_fit / YD_TO_M),
                "fit_relative_speed_kt": float(v_fit / KNOT_TO_MPS),
            }
        return CpaResult(
            observer_id=observer_id,
            pass_index=item.index,
            final=final,
            cpa_tick=float(tc),
            cpa_tick_sigma_s=bias_t,
            cpa_slant_range_yd=r / YD_TO_M,
            cpa_slant_range_sigma_yd=total_r / YD_TO_M,
            relative_speed_kt=v / KNOT_TO_MPS,
            relative_speed_sigma_kt=sigma_v / KNOT_TO_MPS,
            slope_hz_per_s=slope,
            method_note=(
                "tc: zero crossing vs recognized frequency; V: pre/post-CPA frequency change; "
                "R: CPA slope; sigma: source-frequency error + speed error"
            ),
            range_from_slope_yd=r / YD_TO_M,
            bias_shift_tick_s=bias_t,
            bias_range_sigma_yd=bias_r / YD_TO_M,
            speed_range_sigma_yd=speed_r / YD_TO_M,
            **fit_values,
        )


def _crossing(ticks: np.ndarray, freqs: np.ndarray, f0: float) -> float | None:
    diff = freqs - f0
    for index in range(1, len(diff)):
        if diff[index - 1] > 0.0 >= diff[index]:
            prev, now = diff[index - 1], diff[index]
            frac = prev / (prev - now) if prev != now else 0.0
            return float(ticks[index - 1] + frac * (ticks[index] - ticks[index - 1]))
    return None
