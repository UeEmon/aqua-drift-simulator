"""Maneuver timing from the Doppler (time differences of the change between observers).

When the target starts or ends a change of course, speed or depth rate, its velocity starts
to change, so the received frequency f(t) of every observer gets a kink (a step in df/dt).
The change is carried by the sound, so observer j hears it at

    t_j = t_m + r_j / c        r_j = |target at t_m - observer j at t_j|

with the same (unknown) maneuver time t_m for every observer. The differences t_j - t_k are
range differences (r_j - r_k) / c: a hyperbola per observer pair, independent of the Doppler
level, the source frequency and its bias.

Onset per observer: over the last `window_s` seconds, f(t) is fitted with a quadratic (the
smooth geometric Doppler curve) plus a hinge k * max(0, t - tau). The quadratic part is
projected out once per window length, so every candidate tau costs one dot product. A kink
is accepted when the hinge lowers the squared error by more than chi2 * sigma^2 (sigma: the
measurement error or the fit residual, whichever is larger), the slope step is at least
min_slope_hz_s, and the kink lies at least SETTLE_S inside the window (enough samples after it for a stable tau). The 1-sigma
onset error is the half-width of the tau interval within sigma^2 of the best fit, plus a
model floor (the turn is not an exact kink).

Grouping: onsets of different observers within max_range / c (+ margins) belong to one event;
the event closes when every observer that could hear it had time to detect it. An event with
two or more observers becomes a likelihood term on the target position at t_m:

    log L = -1/2 sum_j ((t_j - r_j / c) - t_m)^2 / sigma_j^2,  t_m = weighted mean (per particle)
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

GRID_S = 0.25  # onset-time grid inside the window
EDGE_S = 6.0  # no kink in the first EDGE_S seconds of the window (quadratic needs support)
SETTLE_S = 12.0  # a kink is reported once it is this far inside the window
SIGMA_FLOOR_HZ = 1e-3
EDGE_GUARD_S = 2.0  # a best fit this close to the ends of the tau range is a kink outside it
MAX_ONSET_SIGMA_S = 0.75  # vaguer onsets are not used


@dataclass
class Onset:
    observer_id: str
    tick: float  # reception time of the change
    sigma_s: float
    position: np.ndarray  # observer position (local m) at the onset
    slope_change_hz_s: float


@dataclass
class TimingEvent:
    first: float
    onsets: list[Onset] = field(default_factory=list)
    closed: bool = False
    used: bool = True  # False: no particle could explain it (discarded)

    def summary(self) -> dict:
        first = min(o.tick for o in self.onsets)
        return {
            "tick": round(first, 2),
            "observers": [
                {"observer_id": o.observer_id, "offset_s": round(o.tick - first, 2),
                 "sigma_s": round(o.sigma_s, 2), "slope_change_hz_s": round(o.slope_change_hz_s, 4)}
                for o in sorted(self.onsets, key=lambda o: o.tick)
            ],
        }


class _Basis:
    """Hinge columns (slope step and curvature change after tau) with the quadratic projected
    out, for a contiguous window of n seconds."""

    def __init__(self, n: int) -> None:
        u = np.arange(n, dtype=float)
        poly = np.stack([np.ones(n), u / n, (u / n) ** 2], axis=1)
        self.q, _ = np.linalg.qr(poly)
        self.taus = np.arange(EDGE_S, n - 1 - SETTLE_S + 1e-9, GRID_S)
        after = np.maximum(0.0, u[:, None] - self.taus[None, :])  # (n,G)
        columns = []
        for h in (after, after * after / n):
            columns.append(h - self.q @ (self.q.T @ h))
        self.h1, self.h2 = columns
        g11 = np.einsum("ij,ij->j", self.h1, self.h1)
        g12 = np.einsum("ij,ij->j", self.h1, self.h2)
        g22 = np.einsum("ij,ij->j", self.h2, self.h2)
        det = np.maximum(g11 * g22 - g12 * g12, 1e-12)
        self.inv = np.stack([g22 / det, -g12 / det, g11 / det], axis=1)  # (G,3) of the 2x2


class ManeuverTiming:
    def __init__(self, window_s: int, chi2: float, sigma_floor_s: float, min_slope_hz_s: float,
                 max_range_m: float, sound_speed: float) -> None:
        self.configure(window_s, chi2, sigma_floor_s, min_slope_hz_s, max_range_m, sound_speed)
        self.series: dict[str, deque[tuple[int, float, float, np.ndarray]]] = {}
        self.last_onset: dict[str, float] = {}
        self.open: TimingEvent | None = None
        self.events: deque[TimingEvent] = deque(maxlen=20)  # closed events, newest last
        self._ready: list[TimingEvent] = []  # closed, not yet returned by step()
        self._bases: dict[int, _Basis] = {}

    def configure(self, window_s: int, chi2: float, sigma_floor_s: float, min_slope_hz_s: float,
                  max_range_m: float, sound_speed: float) -> None:
        self.window_s = int(window_s)
        self.chi2 = chi2
        self.sigma_floor = sigma_floor_s
        self.min_slope = min_slope_hz_s
        self.max_delay = max_range_m / sound_speed

    def add(self, tick: int, observer_id: str, frequency: float | None, sigma: float,
            position: np.ndarray) -> Onset | None:
        """One second of an observer's (fused) frequency; returns a new onset if one is found."""
        series = self.series.setdefault(observer_id, deque())
        if frequency is None or (series and tick - series[-1][0] != 1):
            series.clear()  # not detected / gap: the window must be contiguous
            if frequency is None:
                return None
        series.append((tick, frequency, sigma, position))
        while len(series) > self.window_s:
            series.popleft()
        if len(series) < self.window_s:
            return None
        onset = self._fit(observer_id, series)
        if onset is not None:
            self.last_onset[observer_id] = onset.tick
            self._group(onset)
        return onset

    def _fit(self, observer_id: str, series: deque) -> Onset | None:
        n = len(series)
        basis = self._bases.get(n) or self._bases.setdefault(n, _Basis(n))
        if not len(basis.taus):
            return None
        start = series[0][0]
        f = np.array([item[1] for item in series])
        r0 = f - basis.q @ (basis.q.T @ f)
        d1, d2 = basis.h1.T @ r0, basis.h2.T @ r0
        a11, a12, a22 = basis.inv.T
        slopes = a11 * d1 + a12 * d2  # slope step of the best fit at each tau
        score = d1 * slopes + d2 * (a12 * d1 + a22 * d2)  # squared-error reduction
        best = int(np.argmax(score))
        guard = int(EDGE_GUARD_S / GRID_S)
        if best < guard or best > len(score) - 1 - guard:
            return None  # the kink lies outside the part of the window that can place it
        slope = float(slopes[best])
        # local noise: measurement error, or what is left after the fit (over a short window
        # the smooth Doppler curve is a quadratic, so the model error of the filter does not
        # count here), with a floor
        residual = max(float(r0 @ r0) - float(score[best]), 0.0) / max(n - 5, 1)
        sigma = max(series[-1][2], np.sqrt(residual), SIGMA_FLOOR_HZ)
        if score[best] < self.chi2 * sigma * sigma or abs(slope) < self.min_slope:
            return None
        tick = start + float(basis.taus[best])
        a, b, c = score[best - 1], score[best], score[best + 1]
        denom = a - 2 * b + c
        if denom < 0:  # parabolic refinement between grid points
            tick += GRID_S * 0.5 * (a - c) / denom
        inside = basis.taus[score >= score[best] - sigma * sigma]
        half_width = 0.5 * (inside.max() - inside.min()) if len(inside) else GRID_S
        sigma_s = float(np.hypot(max(half_width, 0.5 * GRID_S), self.sigma_floor))
        if sigma_s > max(MAX_ONSET_SIGMA_S, 1.5 * self.sigma_floor):
            return None
        previous = self.last_onset.get(observer_id)
        if previous is not None and abs(tick - previous) < max(3.0, 4.0 * sigma_s):
            return None  # the kink already reported, still inside the window
        index = min(max(round(tick - start), 0), n - 1)
        return Onset(
            observer_id=observer_id, tick=tick, sigma_s=sigma_s, position=series[index][3].copy(),
            slope_change_hz_s=slope,
        )

    def _group(self, onset: Onset) -> None:
        event = self.open
        if event is not None and all(
            o.observer_id != onset.observer_id
            and abs(onset.tick - o.tick) <= self.max_delay + 3.0 * math.hypot(o.sigma_s, onset.sigma_s)
            for o in event.onsets
        ):
            event.onsets.append(onset)
            event.first = min(event.first, onset.tick)
            return
        if event is not None:
            self._close(event)
        self.open = TimingEvent(first=onset.tick, onsets=[onset])

    def _close(self, event: TimingEvent) -> TimingEvent | None:
        event.closed = True
        self.open = None
        if len(event.onsets) >= 2:
            self.events.append(event)
            self._ready.append(event)
            return event
        return None

    def step(self, tick: int) -> list[TimingEvent]:
        """Close the open event once every observer that could hear it had time to report it;
        returns the events closed since the last call that two or more observers heard."""
        event = self.open
        if event is not None and tick >= event.first + self.max_delay + SETTLE_S + 3.0:
            self._close(event)
        ready, self._ready = self._ready, []
        return ready

    def recent(self, since_tick: float) -> list[TimingEvent]:
        return [e for e in self.events if e.first >= since_tick]


def _weights(event: TimingEvent) -> np.ndarray:
    return 1.0 / np.array([o.sigma_s for o in event.onsets]) ** 2


def emission_times(
    position_at: Callable[[np.ndarray], np.ndarray],
    event: TimingEvent,
    sound_speed: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per state: the emission time implied by each observer's onset (count, m) and their
    weighted mean, the maneuver time t_m (count,). `position_at(t)` gives the target position
    (count, 3) of every state at the per-state times t (count,)."""
    ticks = np.array([o.tick for o in event.onsets])
    weights = _weights(event)
    observers = np.array([o.position for o in event.onsets])  # (m,3)
    t_m = np.full(count, ticks.min() - 1.0)
    for _ in range(3):
        p = position_at(t_m)
        r = np.linalg.norm(p[:, None, :] - observers[None, :, :], axis=2)  # (count,m)
        emitted = ticks[None, :] - r / sound_speed
        t_m = (emitted @ weights) / weights.sum()
    return emitted, t_m


def timing_loglik(
    position_at: Callable[[np.ndarray], np.ndarray],
    event: TimingEvent,
    sound_speed: float,
    count: int,
) -> np.ndarray:
    """Log-likelihood of `count` states for one event (the common t_m profiled out)."""
    emitted, t_m = emission_times(position_at, event, sound_speed, count)
    return -0.5 * ((emitted - t_m[:, None]) ** 2) @ _weights(event)
