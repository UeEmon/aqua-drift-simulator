"""Resample-move step (Gilks & Berzuini) for the Doppler particle filter.

Error-free Doppler makes the posterior collapse onto thin, curved manifolds (e.g. the ring of
tracks consistent with a single observer's Doppler curve). A plain sequential filter then loses
the true state within seconds. The move step restores the particle set using the whole recent
observation window:

1. window log-likelihood pi(x) of every particle under the constant-course/speed (through the
   water) model, back-propagated through the linear current field;
2. multi-start Levenberg-Marquardt from the best particles -> local modes with Laplace
   covariances (this also finds modes the particle set has lost);
3. Metropolis-Hastings independence moves from the mixture of those Gaussians, then a
   random-walk MH move. MH acceptance keeps pi(x) as the invariant density, so the moves
   rejuvenate the particles without biasing the posterior.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from aqua_drift.estimation.current_fit import CurrentFit


@dataclass
class Epoch:
    tick: int
    pos: np.ndarray  # (R,3)
    vel: np.ndarray  # (R,3)
    det: np.ndarray  # (R,) bool
    freq: np.ndarray  # (R,) nan where not detected
    rec: np.ndarray  # (R,)
    ids: tuple[str, ...] = ()
    transition: bool = False  # some observer changed detected/not-detected here (or next)


@dataclass
class MoveParams:
    max_range: float
    gate_soft: float
    c: float
    sigma_f: float
    bias_sigma: float
    max_speed: float
    max_depth: float


class EpochStore:
    def __init__(self, window_s: int = 900) -> None:
        self.window_s = window_s
        self.epochs: deque[Epoch] = deque()
        self._last_det: dict[str, bool] = {}

    def add(self, epoch: Epoch) -> None:
        flipped = False
        for observer_id, det in zip(epoch.ids, epoch.det, strict=False):
            previous = self._last_det.get(observer_id)
            if previous is not None and previous != bool(det):
                flipped = True
            self._last_det[observer_id] = bool(det)
        if flipped:
            epoch.transition = True
            if self.epochs:
                self.epochs[-1].transition = True
        self.epochs.append(epoch)
        while self.epochs and epoch.tick - self.epochs[0].tick > self.window_s:
            self.epochs.popleft()

    def sample(self, count: int, since_tick: int | None = None) -> list[tuple[Epoch, float]]:
        """Evenly spread subset plus every detection-transition epoch.

        The returned weight multiplies only the Doppler term (each kept epoch stands for the
        epochs skipped around it); range-gate terms are counted once, because entering or
        leaving the detection range is a single event, not a repeated measurement."""
        epochs = [e for e in self.epochs if since_tick is None or e.tick >= since_tick]
        n = len(epochs)
        if n == 0:
            return []
        if n <= count:
            return [(e, 1.0) for e in epochs]
        regular = set(np.unique(np.linspace(0, n - 1, count).round().astype(int)).tolist())
        transitions = {i for i, e in enumerate(epochs) if e.transition}
        weight = n / len(regular)
        return [(epochs[i], weight if i in regular else 0.0) for i in sorted(regular | transitions)]


@dataclass
class Window:
    """Sampled epochs flattened to one row per (epoch, observer)."""

    dt: np.ndarray  # (K,) seconds before now
    pos: np.ndarray  # (K,3)
    vel: np.ndarray  # (K,3)
    det: np.ndarray  # (K,) bool
    freq: np.ndarray  # (K,)
    rec: np.ndarray  # (K,)
    w_dop: np.ndarray  # (K,) Doppler weight (0 where not detected / not regular)
    has_doppler: bool

    @classmethod
    def build(
        cls,
        now: int,
        sample: list[tuple[Epoch, float]],
        relevant: callable | None = None,
    ) -> Window:
        """`relevant(epoch) -> bool mask` drops non-detecting rows that cannot affect any
        particle (observer far beyond max range of the whole particle cloud)."""
        dt, pos, vel, det, freq, rec, wd = [], [], [], [], [], [], []
        for epoch, weight in sample:
            keep = epoch.det | (relevant(epoch) if relevant is not None else True)
            keep = np.broadcast_to(keep, epoch.det.shape)
            k = int(keep.sum())
            if k == 0:
                continue
            dt.append(np.full(k, float(now - epoch.tick)))
            pos.append(epoch.pos[keep])
            vel.append(epoch.vel[keep])
            det.append(epoch.det[keep])
            freq.append(np.where(epoch.det[keep], epoch.freq[keep], 0.0))
            rec.append(epoch.rec[keep])
            wd.append(np.where(epoch.det[keep], weight, 0.0))
        if not dt:
            empty = np.zeros(0)
            return cls(empty, np.zeros((0, 3)), np.zeros((0, 3)), empty.astype(bool), empty,
                       empty, empty, False)
        w_dop = np.concatenate(wd)
        return cls(
            dt=np.concatenate(dt),
            pos=np.vstack(pos),
            vel=np.vstack(vel),
            det=np.concatenate(det),
            freq=np.concatenate(freq),
            rec=np.concatenate(rec),
            w_dop=w_dop,
            has_doppler=bool(np.any(w_dop > 0)),
        )


def _geometry(
    states: np.ndarray, win: Window, current: CurrentFit
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(los (M,K,3), dist (M,K), rdot (M,K)) with positions back-propagated to each row time
    under constant through-water velocity in the linear current (2nd order in the gradient)."""
    p = states[:, 0:3]
    u = states[:, 3:6]
    g = u + current.velocity(p)  # (M,3)
    gg = g @ current.gradient.T  # (M,3)
    dt = win.dt[None, :, None]
    q = p[:, None, :] - g[:, None, :] * dt + 0.5 * dt * dt * gg[:, None, :]
    gq = g[:, None, :] - gg[:, None, :] * dt
    los = q - win.pos[None, :, :]
    dist = np.linalg.norm(los, axis=2)
    rdot = np.sum((gq - win.vel[None, :, :]) * los, axis=2) / np.maximum(dist, 1e-6)
    return los, dist, rdot


def window_loglik(
    states: np.ndarray,
    win: Window,
    current: CurrentFit,
    prm: MoveParams,
) -> np.ndarray:
    total = np.empty(len(states))
    chunk = int(max(64, 1_500_000 // max(len(win.dt), 1)))
    for begin in range(0, len(states), chunk):
        part = states[begin : begin + chunk]
        _, dist, rdot = _geometry(part, win, current)
        margin = (prm.max_range - dist) / prm.gate_soft
        gate = np.where(win.det[None, :], -np.logaddexp(0.0, -margin), -np.logaddexp(0.0, margin))
        pred = (win.rec[None, :] - part[:, 6:7]) * (1.0 - rdot / prm.c)
        dop = win.w_dop[None, :] * ((pred - win.freq[None, :]) / prm.sigma_f) ** 2
        total[begin : begin + chunk] = gate.sum(axis=1) - 0.5 * dop.sum(axis=1)
    if prm.bias_sigma > 0:
        total += -0.5 * (states[:, 6] / prm.bias_sigma) ** 2
    speed = np.linalg.norm(states[:, 3:5], axis=1)
    bad = (speed > prm.max_speed) | (states[:, 2] < 0) | (states[:, 2] > prm.max_depth)
    total[bad] = -np.inf
    return total


def _residuals(x: np.ndarray, win: Window, current: CurrentFit, prm: MoveParams) -> np.ndarray:
    _, dist, rdot = _geometry(x[None, :], win, current)
    dist, rdot = dist[0], rdot[0]
    margin = (prm.max_range - dist) / prm.gate_soft
    soft = np.where(win.det, np.logaddexp(0.0, -margin), np.logaddexp(0.0, margin))
    pred = (win.rec - x[6]) * (1.0 - rdot / prm.c)
    parts = [
        np.sqrt(2.0 * soft),
        np.sqrt(win.w_dop) * (pred - win.freq) / prm.sigma_f,
        np.array([x[6] / prm.bias_sigma if prm.bias_sigma > 0 else 0.0]),
        np.array([max(0.0, float(np.hypot(x[3], x[4])) - prm.max_speed) * 100.0]),
    ]
    return np.concatenate(parts)


def find_modes(
    starts: np.ndarray,
    win: Window,
    current: CurrentFit,
    prm: MoveParams,
    max_nfev: int = 40,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Multi-start trust-region least squares. Returns [(mean, cov, cost)], duplicates merged."""
    lower = np.array([-np.inf, -np.inf, 0.0, -prm.max_speed, -prm.max_speed, -2.0, -np.inf])
    upper = np.array([np.inf, np.inf, prm.max_depth, prm.max_speed, prm.max_speed, 2.0, np.inf])
    if prm.bias_sigma > 0:
        lower[6], upper[6] = -6 * prm.bias_sigma, 6 * prm.bias_sigma
    else:
        lower[6], upper[6] = -1e-9, 1e-9
    modes: list[tuple[np.ndarray, np.ndarray, float]] = []
    for start in starts:
        x0 = np.clip(start, lower + 1e-6, upper - 1e-6)
        try:
            sol = least_squares(
                _residuals, x0, args=(win, current, prm), bounds=(lower, upper),
                x_scale=np.array([300.0, 300.0, 30.0, 0.5, 0.5, 0.05, 0.05]),
                max_nfev=max_nfev,
            )
        except ValueError:
            continue
        jtj = sol.jac.T @ sol.jac
        cov = np.linalg.pinv(jtj + np.diag([1e-8, 1e-8, 1e-6, 1e-4, 1e-4, 1e-2, 1e-2]))
        cov = 0.5 * (cov + cov.T)
        duplicate = False
        for index, (mean, _, cost) in enumerate(modes):
            if np.linalg.norm(mean[0:3] - sol.x[0:3]) < 50.0 and np.linalg.norm(
                mean[3:5] - sol.x[3:5]
            ) < 0.2:
                duplicate = True
                if sol.cost < cost:
                    modes[index] = (sol.x, cov, float(sol.cost))
                break
        if not duplicate:
            modes.append((sol.x, cov, float(sol.cost)))
    best = min((m[2] for m in modes), default=0.0)
    kept = sorted((m for m in modes if m[2] - best < 50.0), key=lambda m: m[2])
    return kept[:12]


class GaussianMixture:
    def __init__(self, modes: list[tuple[np.ndarray, np.ndarray, float]], inflate: float = 1.5):
        self.means = np.array([m[0] for m in modes])
        covs = []
        for _, cov, _ in modes:
            c = cov * inflate + np.diag([25.0, 25.0, 9.0, 0.01, 0.01, 1e-4, 1e-4])
            w, v = np.linalg.eigh(c)
            covs.append((v * np.maximum(w, 1e-9)) @ v.T)
        self.covs = np.array(covs)
        self.chols = np.array([np.linalg.cholesky(c) for c in self.covs])
        self.invs = np.array([np.linalg.inv(c) for c in self.covs])
        self.logdets = np.array([np.linalg.slogdet(c)[1] for c in self.covs])
        costs = np.array([m[2] for m in modes])
        logw = -(costs - costs.min())  # Laplace-style mode weights, tempered
        logw = np.maximum(logw, -10.0)
        self.log_weights = logw - np.logaddexp.reduce(logw)

    def sample(self, count: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(len(self.means), size=count, p=np.exp(self.log_weights))
        z = rng.normal(size=(count, self.means.shape[1]))
        return self.means[comp] + np.einsum("nij,nj->ni", self.chols[comp], z)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        terms = []
        for k in range(len(self.means)):
            diff = x - self.means[k]
            maha = np.einsum("ni,ij,nj->n", diff, self.invs[k], diff)
            terms.append(self.log_weights[k] - 0.5 * (maha + self.logdets[k] + d * np.log(2 * np.pi)))
        return np.logaddexp.reduce(np.array(terms), axis=0)
