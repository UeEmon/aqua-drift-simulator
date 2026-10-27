"""Turn model of the move step (estimation/move.py, requirement 4.27): the window fit with a
coordinated turn follows a course change that the constant-velocity fit lags."""
import math

import numpy as np

from aqua_drift.estimation.current_fit import CurrentFit
from aqua_drift.estimation.move import (
    Epoch,
    EpochStore,
    MoveParams,
    Window,
    _turn_back,
    find_modes,
    window_loglik,
)

SPEED = 4.1  # m/s (8 kt)
RATE = math.radians(1.0)  # 1 deg/s, 090 -> 180
F0 = 400.0
C = 1500.0
RANGE = 457.0  # 500 YD
OBSERVERS = np.array([[-300.0, 0.0, 30.0], [300.0, 0.0, 30.0]])  # one straight line, as laid at 500 YD


def _heading(t: float) -> float:
    """Truth heading [rad] at t seconds after the turn began (straight before and after)."""
    return math.radians(90.0) + RATE * min(max(t, 0.0), math.pi / 2 / RATE)


def _truth(t_now: float, step: float = 0.1) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Positions and velocities at whole seconds from -300 s to t_now (turn at 0 s)."""
    times = np.arange(-300.0, t_now + 1e-9, 1.0)
    p = np.array([-1200.0, 200.0, 60.0])
    t = -300.0
    pos, vel = [], []
    for target in times:
        while t < target - 1e-9:
            h = _heading(t + 0.5 * step)
            p = p + SPEED * step * np.array([math.sin(h), math.cos(h), 0.0])
            t += step
        h = _heading(target)
        pos.append(p.copy())
        vel.append(SPEED * np.array([math.sin(h), math.cos(h), 0.0]))
    return pos, vel, times


def _window(t_now: float, window_s: int) -> tuple[Window, np.ndarray]:
    pos, vel, times = _truth(t_now)
    store = EpochStore(window_s)
    for p, v, t in zip(pos, vel, times, strict=True):
        los = p - OBSERVERS
        dist = np.linalg.norm(los, axis=1)
        det = dist <= RANGE
        rdot = (los @ v) / dist
        freq = np.where(det, F0 * (1.0 - rdot / C), np.nan)
        store.add(Epoch(int(t), OBSERVERS.copy(), np.zeros((2, 3)), det, freq, np.full(2, F0),
                        ids=("a", "b")))
    sample = store.sample(10_000)
    now = int(times[-1])
    state = np.r_[pos[-1], vel[-1], 0.0]
    return Window.build(now, sample), state


def _params() -> MoveParams:
    return MoveParams(max_range=RANGE, gate_soft=14.0, c=C, sigma_f=0.03, bias_sigma=0.5,
                      max_speed=12.9, max_depth=457.0)


def _course(state: np.ndarray) -> float:
    return math.degrees(math.atan2(state[3], state[4])) % 360.0


def test_turn_back_reduces_to_constant_velocity_and_integrates_the_turn() -> None:
    u = np.array([[3.0, 1.0, 0.1]])
    dt = np.array([0.0, 10.0, 50.0, 120.0, 300.0])
    straight, _ = _turn_back(u, dt, None)
    no_turn, _ = _turn_back(u, dt, np.array([0.0, 20.0, 70.0]))
    np.testing.assert_allclose(no_turn, straight)
    turn = np.array([RATE, 20.0, 70.0])  # turned from 90 s until 20 s ago
    disp, vel = _turn_back(u, dt, turn)

    def velocity(tau: np.ndarray) -> np.ndarray:
        a = RATE * np.clip(tau - 20.0, 0.0, 70.0)
        return np.stack([3.0 * np.cos(a) - np.sin(a), np.cos(a) + 3.0 * np.sin(a)], -1)

    for k, tau in enumerate(dt):
        grid = np.linspace(0.0, tau, 20001)
        v = velocity(grid)
        integral = (v[:-1] + v[1:]).sum(axis=0) * (tau / 20000) / 2 if tau else np.zeros(2)
        np.testing.assert_allclose(disp[0, k, 0:2], integral, atol=1e-4)
        np.testing.assert_allclose(vel[0, k, 0:2], velocity(np.array([tau]))[0], atol=1e-9)


def test_turn_fit_follows_the_course_the_constant_velocity_fit_lags() -> None:
    """60 s into a 1 deg/s turn (truth 150 deg), seen by two collinear observers: the
    constant-velocity fit stays near the old course, the turn fit finds the new one."""
    win, truth = _window(60.0, 180)
    prm = _params()
    current = CurrentFit.zero()
    old_course = truth.copy()
    old_course[3:5] = [SPEED, 0.0]  # what the filter believed before the turn
    cv = find_modes(old_course[None, :], win, current, prm, max_nfev=80)
    detected = win.dt[win.w_dop > 0]
    starts = np.array([np.r_[old_course, sign * RATE, detected.min(), duration]
                       for sign in (1.0, -1.0) for duration in (30.0, 90.0)])
    turn = find_modes(starts, win, current, prm, max_nfev=80, turn_span=(detected.min(), 180.0))
    truth_course = _course(truth)
    assert abs(truth_course - 150.0) < 1.0
    cv_error = abs(_course(cv[0][0]) - truth_course)
    turn_error = abs(_course(turn[0][0]) - truth_course)
    assert cv_error > 30.0
    # two collinear observers barely tell "still turning" from "stopped turning seconds ago"
    assert turn_error < 15.0
    assert turn_error < cv_error / 3
    assert math.degrees(turn[0][0][7]) > 0.7  # starboard, about 1 deg/s
    assert turn[0][2] < cv[0][2] - 10.0
    # the turn hypothesis explains the window far better at the truth than constant velocity
    assert window_loglik(truth[None, :], win, current, prm, turn[0][0][7:10])[0] > (
        window_loglik(truth[None, :], win, current, prm)[0] + 10.0
    )


def test_turn_fit_after_the_turn_holds_the_new_course() -> None:
    """60 s after the turn ended, the fit ends the turn before now instead of turning on."""
    win, truth = _window(150.0, 240)
    prm = _params()
    current = CurrentFit.zero()
    detected = win.dt[win.w_dop > 0]
    if not len(detected):
        return
    start = truth.copy()
    starts = np.array([np.r_[start, RATE, detected.min(), 90.0], np.r_[start, RATE, 60.0, 90.0]])
    turn = find_modes(starts, win, current, prm, max_nfev=80, turn_span=(detected.min(), 240.0))
    assert abs(_course(turn[0][0]) - 180.0) < 5.0
