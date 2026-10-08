"""Depth from Doppler: how well can the target depth be obtained, and what improves it?

Study tool (not used at run time). Geometry is set up in the WATER frame: observers drift with
the water and the target is carried by the same water, so relative to the observers the target
moves on a straight line at its through-water velocity and the observers are fixed. (Exact for
a uniform current; the runtime linear current field is close to uniform over one pass.)

Measurement model = the runtime estimator's:
    f_j(t) = (f_rec - b) * (1 - rdot_j(t) / c)      every 1 s while detected, sigma_f each
    detection start / end of observer j: slant range = R_max (known, common, no missed detections)
    unknowns theta = [p0 (e, n, d), v (e, n, d), b]  (constant-velocity pass, common bias b)
    weak priors: depth uniform 0..1500 ft (sigma 433 ft), bias sigma 0.5 Hz

Physics of the depth information (single straight pass, speed v, 3-D miss distance R0):
    rdot(t) = v^2 (t - tc) / sqrt(R0^2 + v^2 (t - tc)^2)
    -> one observer's Doppler curve gives f0, v, tc and the SLANT miss distance R0 only.
    R0^2 = d^2 + (z_t - z_j)^2 with d the horizontal miss distance, so
    dR0/dz_t = (z_t - z_j) / R0 : depth is seen only through the vertical share of the slant
    range. Three ways to make that share large:
      A. overflight  : an observer the target passes (nearly) over (d <~ depth difference)
      B. vertical baseline : sensors at clearly different depths; for two sensors at the same
         horizontal position  R0s^2 - R0d^2 = 2 z_t (z_d - z_s) + z_s^2 - z_d^2, i.e. LINEAR in
         the target depth and independent of the horizontal miss distance
      C. many observers at different horizontal miss distances (the current 4-point surround
         plus forward drops), which constrains depth only weakly when d >> depth

    python -m aqua_drift.analysis.depth_doppler            # prints all study tables
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

YD = 0.9144
FT = 0.3048
KT = 0.5144444444444445


@dataclass
class Pass:
    """One straight pass through a fixed (water-frame) observer field."""

    observers: list[tuple[float, float, float]]  # (east m, north m, depth m) per sensor
    target_depth_m: float = 500 * FT
    speed_kt: float = 8.0
    heading_deg: float = 90.0
    track_offset_m: float = 0.0  # track passes this far north of the origin (heading 090)
    vertical_fps: float = 0.0  # depth rate, positive = going deeper
    half_length_m: float = 6000.0  # pass from -L to +L along track (relative to the origin)
    f0: float = 400.0
    c: float = 1500.0
    sigma_f: float = 0.03
    max_range_m: float = 6000 * YD
    gate_sigma_m: float = 3.0
    use_gate: bool = True
    depth_prior_sigma_ft: float = 433.0
    bias_prior_sigma_hz: float = 0.5
    vertical_rate_prior_fps: float | None = None  # "depth keeping" prior on the depth rate
    report: str = "mid"  # where the depth error is reported: "mid" pass or "end" of the pass
    ticks: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        v = self.speed_kt * KT
        self.ticks = np.arange(0.0, 2 * self.half_length_m / v + 1.0, 1.0)

    def truth(self) -> np.ndarray:
        """theta at t = 0: [e, n, d, ve, vn, vd, b]. The track passes `track_offset_m` to the
        left of the origin (north of it for heading 090) at mid-pass."""
        h = math.radians(self.heading_deg)
        v = self.speed_kt * KT
        ux, uy = math.sin(h), math.cos(h)
        mid_e, mid_n = -uy * self.track_offset_m, ux * self.track_offset_m
        e0 = mid_e - ux * self.half_length_m
        n0 = mid_n - uy * self.half_length_m
        return np.array([e0, n0, self.target_depth_m, ux * v, uy * v, self.vertical_fps * FT, 0.0])


def _positions(theta: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    return theta[None, 0:3] + theta[None, 3:6] * ticks[:, None]


def measurement_vector(theta: np.ndarray, p: Pass, det: np.ndarray, gates: list[tuple[float, int]]) -> np.ndarray:
    """Whitened predictions (each entry divided by its sigma)."""
    obs = np.asarray(p.observers)
    pos = _positions(theta, p.ticks)
    rel = pos[:, None, :] - obs[None, :, :]
    rng = np.linalg.norm(rel, axis=2)
    rdot = np.sum(theta[None, None, 3:6] * rel, axis=2) / np.maximum(rng, 1e-6)
    f = (p.f0 - theta[6]) * (1.0 - rdot / p.c)
    out = [f[det] / p.sigma_f]
    if p.use_gate and gates:
        g = []
        for t_mid, j in gates:
            q = theta[0:3] + theta[3:6] * t_mid
            g.append(np.linalg.norm(q - obs[j]) / p.gate_sigma_m)
        out.append(np.array(g))
    out.append(np.array([theta[2] / (p.depth_prior_sigma_ft * FT), theta[6] / p.bias_prior_sigma_hz]))
    if p.vertical_rate_prior_fps:
        out.append(np.array([theta[5] / (p.vertical_rate_prior_fps * FT)]))
    return np.concatenate(out)


def detections(p: Pass, theta: np.ndarray) -> tuple[np.ndarray, list[tuple[float, int]]]:
    obs = np.asarray(p.observers)
    pos = _positions(theta, p.ticks)
    rng = np.linalg.norm(pos[:, None, :] - obs[None, :, :], axis=2)
    det = rng <= p.max_range_m
    gates = [
        (p.ticks[i] - 0.5, j)
        for i in range(1, len(p.ticks))
        for j in range(det.shape[1])
        if det[i, j] != det[i - 1, j]
    ]
    return det, gates


def crlb(p: Pass) -> dict:
    theta = p.truth()
    det, gates = detections(p, theta)
    base = measurement_vector(theta, p, det, gates)
    steps = np.array([0.5, 0.5, 0.2, 0.002, 0.002, 0.001, 0.0005])
    jac = np.stack([
        (measurement_vector(theta + np.eye(7)[i] * h, p, det, gates) - base) / h for i, h in enumerate(steps)
    ], axis=1)
    fisher = jac.T @ jac
    cov = np.linalg.pinv(fisher)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    # report at mid-pass (t = T/2): position covariance propagated with the velocity
    t_mid = p.ticks[-1] / 2 if p.report == "mid" else p.ticks[-1]
    jac_mid = np.hstack([np.eye(3), np.eye(3) * t_mid, np.zeros((3, 1))])
    cov_mid = jac_mid @ cov @ jac_mid.T
    return {
        "depth_ft": math.sqrt(max(cov_mid[2, 2], 0.0)) / FT,
        "horizontal_yd": math.sqrt(max(cov_mid[0, 0] + cov_mid[1, 1], 0.0)) / YD,
        "vertical_rate_fps": sd[5] / FT,
        "bias_hz": sd[6],
        "detections": int(det.sum()),
        "gates": len(gates),
    }


# ------------------------------------------------------------------------------- layouts
def surround(radius_yd: float = 3000.0, depths_ft=(200.0, 350.0, 500.0, 200.0)) -> list[tuple[float, float, float]]:
    """Runtime default: 4 observers on a circle around the target's start; depths cycle
    200 / 350 / 500 Ft (deployment.depth_ft + (index % 3) x depth_step_ft)."""
    out = []
    for k, depth in enumerate(depths_ft):
        a = math.radians(45 + 90 * k)
        out.append((radius_yd * YD * math.sin(a), radius_yd * YD * math.cos(a), depth * FT))
    return out


def with_overflight(base: list, miss_m: float, depth_ft: float = 200.0, along_m: float = 0.0) -> list:
    """Add one observer on (or `miss_m` beside) the target track (track along east, north = 0)."""
    return [*base, (along_m, miss_m, depth_ft * FT)]


def vertical_pairs(base: list, deep_ft: float) -> list:
    """Each observer carries a second hydrophone at `deep_ft` on the same cable."""
    return [*base, *[(e, n, deep_ft * FT) for e, n, _ in base]]


def tables(sigma_f: float = 0.03) -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    base = surround()

    rows = []
    for miss_ft in (0, 150, 300, 600, 1500, 3000, 6000):
        p = Pass(observers=with_overflight(base, miss_ft * FT), sigma_f=sigma_f)
        rows.append((f"{miss_ft} Ft", crlb(p)))
    rows.insert(0, ("（追加なし・現行の4点）", crlb(Pass(observers=base, sigma_f=sigma_f))))
    out["A. 航跡上の観測者（4点＋1、水平ずれ）"] = rows

    rows = []
    for label, depths in (
        ("全員 200 Ft", (200, 200, 200, 200)),
        ("200/350/500 Ft（現行）", (200, 350, 500, 200)),
        ("60/1000 Ft 交互", (60, 1000, 60, 1000)),
        ("60/1500 Ft 交互", (60, 1500, 60, 1500)),
    ):
        rows.append((label, crlb(Pass(observers=surround(depths_ft=depths), sigma_f=sigma_f))))
    for deep in (600, 1000, 1500):
        rows.append((f"各観測者に 200＋{deep} Ft の2受波器", crlb(Pass(observers=vertical_pairs(surround(depths_ft=(200,) * 4), deep), sigma_f=sigma_f))))
    out["B. 観測者の深度（鉛直基線）"] = rows

    rows = []
    for depth_ft in (100, 300, 500, 1000, 1400):
        current = crlb(Pass(observers=base, target_depth_m=depth_ft * FT, sigma_f=sigma_f))
        over = crlb(Pass(observers=with_overflight(base, 0.0), target_depth_m=depth_ft * FT, sigma_f=sigma_f))
        pair = crlb(Pass(observers=vertical_pairs(surround(depths_ft=(200,) * 4), 1000), target_depth_m=depth_ft * FT, sigma_f=sigma_f))
        rows.append((f"目標 {depth_ft} Ft", {"current": current["depth_ft"], "overflight": over["depth_ft"], "pairs": pair["depth_ft"]}))
    out["C. 目標深度ごとの深度精度（Ft）"] = rows

    rows = []
    for sf in (0.01, 0.03, 0.1):
        current = crlb(Pass(observers=base, sigma_f=sf))
        over = crlb(Pass(observers=with_overflight(base, 0.0), sigma_f=sf))
        pair = crlb(Pass(observers=vertical_pairs(surround(depths_ft=(200,) * 4), 1000), sigma_f=sf))
        rows.append((f"σf {sf} Hz", {"current": current["depth_ft"], "overflight": over["depth_ft"], "pairs": pair["depth_ft"]}))
    out["D. 周波数誤差ごとの深度精度（Ft）"] = rows

    rows = []
    for speed in (4, 8, 15):
        current = crlb(Pass(observers=base, speed_kt=speed, sigma_f=sigma_f))
        over = crlb(Pass(observers=with_overflight(base, 0.0), speed_kt=speed, sigma_f=sigma_f))
        pair = crlb(Pass(observers=vertical_pairs(surround(depths_ft=(200,) * 4), 1000), speed_kt=speed, sigma_f=sigma_f))
        rows.append((f"{speed} kt", {"current": current["depth_ft"], "overflight": over["depth_ft"], "pairs": pair["depth_ft"]}))
    out["E. 目標速力ごとの深度精度（Ft）"] = rows

    rows = []
    for label, prior in (("深度変化率の事前なし", None), ("±0.5 Ft/s", 0.5), ("±0.1 Ft/s（深度保持）", 0.1), ("±0.02 Ft/s（厳しい深度保持）", 0.02)):
        r = {}
        for name, obs in (("current", base), ("overflight", with_overflight(base, 0.0)),
                          ("pairs", vertical_pairs(surround(depths_ft=(200,) * 4), 1000))):
            r[name + "_mid"] = crlb(Pass(observers=obs, vertical_rate_prior_fps=prior, sigma_f=sigma_f))["depth_ft"]
            r[name + "_end"] = crlb(Pass(observers=obs, vertical_rate_prior_fps=prior, sigma_f=sigma_f, report="end"))["depth_ft"]
        rows.append((label, r))
    out["G. 深度変化率の事前（深度保持の仮定）とパス中央・終了時の深度精度（Ft）"] = rows
    return out


def lloyd_mirror_beat(source_depth_m: float, receiver_depth_m: float, horizontal_m: float,
                      range_rate_mps: float, f0: float = 400.0, c: float = 1500.0) -> dict:
    """Surface-reflected path (image source at -z_s): path difference ~ 2 z_s z_r / R and the
    beat (fading) frequency between the direct and reflected arrivals = their Doppler difference."""
    direct = math.hypot(horizontal_m, source_depth_m - receiver_depth_m)
    reflected = math.hypot(horizontal_m, source_depth_m + receiver_depth_m)
    path_diff = reflected - direct
    # d(path diff)/dt for horizontal range rate (target depth constant)
    rate = range_rate_mps * horizontal_m * (1 / reflected - 1 / direct)
    beat = abs(f0 * rate / c)
    return {"path_difference_m": path_diff, "beat_hz": beat, "beat_period_s": (1 / beat) if beat > 0 else math.inf}


# ------------------------------------------------------------------ particle-filter validation
PF_CASES = ("current", "overflight", "overflight_300yd", "depths_60_1000", "pairs_200_1000")


def pf_case_positions(case: str):
    """Observer positions for the runtime particle filter (same physics as the containers)."""
    from aqua_drift.deployment import _offset, default_position
    from aqua_drift.models import ScenarioConfig

    config = ScenarioConfig()
    base = [default_position(config, i) for i in range(4)]
    start = config.target.initial_position
    if case == "current":
        return config, base
    if case == "overflight":  # one more observer on the track, 2500 YD ahead (reached ~9 min)
        return config, [*base, _offset(start, 2500 * YD, 0.0, 200.0)]
    if case == "overflight_300yd":  # placed from an estimate that is 300 YD off the track
        return config, [*base, _offset(start, 2500 * YD, 300 * YD, 200.0)]
    if case == "depths_60_1000":
        return config, [p.model_copy(update={"depth_ft": d}) for p, d in zip(base, (60, 1000, 60, 1000))]
    if case == "pairs_200_1000":
        shallow = [p.model_copy(update={"depth_ft": 200.0}) for p in base]
        deep = [p.model_copy(update={"depth_ft": 1000.0}) for p in base]
        return config, [*shallow, *deep]
    raise ValueError(case)


def _shrinkage_roughen(self, current) -> None:
    """Variance-preserving kernel (Liu & West shrinkage), per cluster: x <- m + a (x - m) + h e,
    a = sqrt(1 - h^2). The plain regularized kernel adds h^2 of the cluster covariance at every
    resampling, which keeps WEAKLY observed directions (depth) inflated to the prior width."""
    d = self.STATE_DIM
    h = (4.0 / (self.n * (d + 2))) ** (1.0 / (d + 4))
    a = math.sqrt(max(1.0 - h * h, 0.0))
    floor = np.diag([1.0, 1.0, 0.3, 1e-4, 1e-4, 1e-5, 1e-6])
    labels = self.clusters()
    for label in np.unique(labels):
        members = labels == label
        count = int(members.sum())
        if count < 2:
            continue
        mean = self.x[members].mean(axis=0)
        cov = np.cov(self.x[members].T) + floor
        cov = 0.5 * (cov + cov.T)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-9)))
        noise = self.rng.normal(size=(count, d)) @ chol.T
        self.x[members] = mean + a * (self.x[members] - mean) + h * noise
    self._limit_speed()
    self._reflect_depth()
    if self.bias_sigma == 0:
        self.x[:, 6] = 0.0
    self.ground_v = self.x[:, 3:6] + current.velocity(self.x[:, 0:3])


def pf_validation(case: str, seconds: int = 1200, particles: int = 3000,
                  checkpoints=(300, 600, 900, 1200), vertical_maneuver_mps: float | None = None,
                  kernel: str = "runtime", move_epochs: int | None = None) -> list[dict]:
    from aqua_drift.estimation.particle_filter import DopplerParticleFilter
    from aqua_drift.scenario import ScenarioRun

    if kernel in ("shrink", "shrink+vprior"):
        DopplerParticleFilter._roughen = _shrinkage_roughen  # study process only
    if kernel == "shrink+vprior":
        # depth-keeping prior on the depth rate inside the resample-move target (window
        # likelihood and its least-squares mode search): vd ~ N(0, 0.03 m/s = 0.1 Ft/s)
        from aqua_drift.estimation import move as move_module
        from aqua_drift.estimation import particle_filter as pf_module

        sigma_vd = 0.03
        base_ll, base_res = move_module.window_loglik, move_module._residuals

        def loglik(states, win, current, prm):
            return base_ll(states, win, current, prm) - 0.5 * (states[:, 5] / sigma_vd) ** 2

        def residuals(x, win, current, prm):
            return np.append(base_res(x, win, current, prm), x[5] / sigma_vd)

        move_module.window_loglik = loglik
        pf_module.window_loglik = loglik
        move_module._residuals = residuals
        if vertical_maneuver_mps is None:
            vertical_maneuver_mps = 0.01
    config, positions = pf_case_positions(case)
    config.estimator.particle_count = particles
    if vertical_maneuver_mps is not None:
        config.estimator.maneuver_vertical_sigma_mps = vertical_maneuver_mps
    if move_epochs is not None:
        config.estimator.move_epochs = move_epochs
    run = ScenarioRun(config, len(positions), observer_positions=positions)
    rows = []
    for _ in range(seconds):
        run.step()
        if run.tick in checkpoints:
            output = run.engine.output()
            estimate = next(e for e in output.estimates if e.mode.value == "ONLINE")
            smoothed = next((e for e in output.estimates if e.mode.value == "SMOOTHED"), None)
            err = run.error(output)
            rows.append({
                "smoothed_depth_error_ft": (smoothed.current_position.depth_ft - run.target.position.depth_ft)
                if smoothed and smoothed.current_position else None,
                "t": run.tick,
                "depth_error_ft": err.get("depth_error_ft"),
                "depth_sigma_ft": estimate.uncertainty.depth_sigma_ft if estimate.uncertainty else None,
                "horizontal_error_yd": err.get("horizontal_error_yd"),
                "status": estimate.observability_status,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-f", type=float, default=0.03)
    parser.add_argument("--pf", choices=PF_CASES, help="run the particle-filter validation for one case")
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--particles", type=int, default=3000)
    parser.add_argument("--vertical-maneuver", type=float, default=None,
                        help="override estimator.maneuver_vertical_sigma_mps for the validation")
    parser.add_argument("--kernel", choices=("runtime", "shrink", "shrink+vprior"), default="runtime")
    parser.add_argument("--move-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.pf:
        import json
        print(json.dumps({"case": args.pf, "kernel": args.kernel, "rows": pf_validation(args.pf, args.seconds, args.particles,
                                                                       vertical_maneuver_mps=args.vertical_maneuver,
                                                                       kernel=args.kernel, move_epochs=args.move_epochs)}))
        return
    for title, rows in tables(args.sigma_f).items():
        print(f"\n## {title}")
        for label, r in rows:
            if "depth_ft" in r:
                print(f"{label:32s} depth {r['depth_ft']:7.1f} Ft  horiz {r['horizontal_yd']:6.1f} YD  "
                      f"vrate {r['vertical_rate_fps']:.3f} Ft/s  bias {r['bias_hz']:.4f} Hz")
            else:
                print(f"{label:32s} " + "  ".join(f"{k} {v:7.1f}" for k, v in r.items()))
    print("\n## F. Lloyd's mirror (surface reflection) beat")
    for horizontal in (500, 1000, 2000, 4000):
        r = lloyd_mirror_beat(500 * FT, 200 * FT, horizontal, 8 * KT)
        print(f"R {horizontal} m: path diff {r['path_difference_m']:.1f} m, beat {r['beat_hz']:.4f} Hz, period {r['beat_period_s']:.0f} s")


if __name__ == "__main__":
    main()
