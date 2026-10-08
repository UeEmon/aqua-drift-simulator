"""Several tonals with bandwidth / stability, propagation delay, maneuver timing."""
import math
import random

import numpy as np
import pytest

from aqua_drift.estimation.current_fit import CurrentFit
from aqua_drift.estimation.engine import fused_frequency
from aqua_drift.estimation.maneuver_timing import ManeuverTiming
from aqua_drift.estimation.particle_filter import DopplerParticleFilter, ObservationRow
from aqua_drift.models import ObserverState, Position, ScenarioConfig, Tonal
from aqua_drift.physics import (
    SourceSignal,
    advance_target,
    doppler_observation,
    initial_target,
    local_offset_m,
)
from aqua_drift.scenario import ScenarioRun


def _observer(config: ScenarioConfig, north_yd: float, east_yd: float = 0.0) -> ObserverState:
    origin = config.target.initial_position
    lat = origin.latitude + math.degrees(north_yd * 0.9144 / 6_371_008.8)
    lon = origin.longitude + math.degrees(
        east_yd * 0.9144 / (6_371_008.8 * math.cos(math.radians(origin.latitude)))
    )
    return ObserverState(observer_id="obs", tick=0, position=Position(latitude=lat, longitude=lon, depth_ft=200))


def _signal(config: ScenarioConfig, seconds: int) -> tuple[SourceSignal, object]:
    signal = SourceSignal(config.source.random_seed)
    target = initial_target(config)
    signal.add(config, target)
    for _ in range(seconds):
        target = advance_target(config, target, 1.0)
        signal.add(config, target)
    return signal, target


def test_tonals_share_the_doppler_factor_and_bias_ratio() -> None:
    config = ScenarioConfig()
    config.source.shared_recognition_bias_hz = 0.4
    config.source.additional_tonals = [Tonal(frequency_hz=1000.0), Tonal(frequency_hz=1500.0)]
    signal, target = _signal(config, 30)
    obs, _ = doppler_observation(config, target, _observer(config, 3000), signal=signal)
    assert [t.recognized_frequency_hz for t in obs.tonals] == pytest.approx([400.4, 1001.0, 1501.5])
    factors = [t.observed_frequency_hz / f for t, f in zip(obs.tonals, (400, 1000, 1500), strict=True)]
    assert factors == pytest.approx([factors[0]] * 3, rel=1e-12)
    assert obs.observed_frequency_hz == obs.tonals[0].observed_frequency_hz


def test_propagation_delay_is_range_over_sound_speed() -> None:
    config = ScenarioConfig()
    signal, target = _signal(config, 30)
    observer = _observer(config, 4000)
    _, truth = doppler_observation(config, target, observer, signal=signal)
    east, north, down = local_offset_m(observer.position, target.position)
    assert truth.propagation_delay_s == pytest.approx(math.sqrt(east**2 + north**2 + down**2) / 1500.0, rel=0.01)
    config.source.propagation_delay = False
    _, instantaneous = doppler_observation(config, target, observer, signal=signal)
    assert instantaneous.propagation_delay_s == 0.0


def test_bandwidth_error_and_common_stability() -> None:
    config = ScenarioConfig()
    config.source.bandwidth_hz = 1.2
    signal, target = _signal(config, 30)
    rng = random.Random(3)
    observer = _observer(config, 3000)
    errors = []
    clean = ScenarioConfig()
    exact, _ = doppler_observation(clean, target, observer, signal=signal)
    for _ in range(4000):
        obs, _ = doppler_observation(config, target, observer, rng, signal=signal)
        errors.append(obs.observed_frequency_hz - exact.observed_frequency_hz)
    assert np.std(errors) == pytest.approx(1.2 / math.sqrt(12), rel=0.05)

    config = ScenarioConfig()
    config.source.stability_hz = 0.05
    config.source.propagation_delay = False  # same emission time for both observers
    signal, target = _signal(config, 200)
    a, _ = doppler_observation(config, target, _observer(config, 3000), signal=signal)
    b, _ = doppler_observation(config, target, _observer(config, -3000), signal=signal)
    ea, _ = doppler_observation(clean, target, _observer(config, 3000), signal=signal)
    eb, _ = doppler_observation(clean, target, _observer(config, -3000), signal=signal)
    da = a.observed_frequency_hz - ea.observed_frequency_hz
    db = b.observed_frequency_hz - eb.observed_frequency_hz
    assert da != 0.0 and da == pytest.approx(db, rel=0.01)  # the same emitted fluctuation


def test_fused_frequency_weights_tonals_by_bandwidth() -> None:
    config = ScenarioConfig()
    config.source.additional_tonals = [Tonal(frequency_hz=800.0, bandwidth_hz=2.0)]
    signal, target = _signal(config, 10)
    obs, _ = doppler_observation(config, target, _observer(config, 3000), signal=signal)
    frequency, recognized, sigma_meas, sigma_drift = fused_frequency(obs, 0.03, 0.0)
    assert recognized == 400.0
    assert frequency == pytest.approx(obs.tonals[0].observed_frequency_hz, abs=0.01)  # error-free line wins
    assert sigma_meas < 0.01 and sigma_drift == 0.0


def test_common_fluctuation_is_marginalized_over_observers() -> None:
    pf = DopplerParticleFilter(10, 6000.0, 1500.0, 0.03, 0.0, 12.0, 500.0, 0.0, 0.0, 15.0)
    pf.x = np.zeros((1, 7))
    pf.ground_v = np.zeros((1, 3))
    rows = [
        ObservationRow(observer_id=str(i), position=np.array([100.0 + i, 0.0, 0.0]), velocity=np.zeros(3),
                       detected=True, frequency=400.0 + e, recognized=400.0, sigma_meas=0.05, sigma_drift=0.2)
        for i, e in enumerate((0.31, 0.27, 0.35))
    ]
    pf.max_range = 1e9
    got = float(pf.log_likelihood(rows)[0])
    e = np.array([-0.31, -0.27, -0.35])
    cov = np.eye(3) * (0.03**2 + 0.05**2) + 0.2**2
    expected = -0.5 * e @ np.linalg.solve(cov, e)
    assert got == pytest.approx(expected, rel=1e-6)


def test_kink_onset_time() -> None:
    timing = ManeuverTiming(40, 25.0, 0.3, 0.003, 5486.0, 1500.0)
    found = []
    for t in range(200):
        f = 400 + 0.02 * t - 1e-4 * t * t + 0.012 * max(0.0, t - 100.4)
        onset = timing.add(t, "a", f, 0.0, np.zeros(3))
        if onset:
            found.append(onset.tick)
    assert len(found) == 1 and found[0] == pytest.approx(100.4, abs=0.1)


def test_maneuver_reaches_observers_after_their_delays() -> None:
    """A turn: each observer's onset is the maneuver time plus its own propagation delay."""
    config = ScenarioConfig()
    config.estimator.particle_count = 1000
    config.deployment.pattern = "ring"
    config.deployment.spacing_yd = 2500
    run = ScenarioRun(config, 12)
    run.run(500)
    run.config.target.desired_hdg_deg = 180.0
    run.config.target.desired_through_water_speed_kt = 12.0
    delays = {}
    for _ in range(40):
        batch = run.step()
        if batch.tick == 501:
            delays = {t.observer_id: t.propagation_delay_s for t in batch.truth}
    events = list(run.engine.timing.events)
    assert events, "the start of the turn is heard by several observers"
    event = events[0]
    assert len(event.onsets) >= 3
    for onset in event.onsets:  # the turn starts at emission time 500
        assert onset.tick == pytest.approx(500.0 + delays[onset.observer_id], abs=0.3)
    meta = run.engine.output().estimates[0].metadata["maneuver_timing"]
    assert meta["events_applied"] >= 1
    assert run.error(run.engine.output())["horizontal_error_yd"] < 300


def test_timing_likelihood_prefers_the_true_position() -> None:
    config = ScenarioConfig()
    config.deployment.pattern = "ring"
    config.deployment.spacing_yd = 2500
    run = ScenarioRun(config, 12)
    run.run(500)
    run.config.target.desired_hdg_deg = 180.0
    run.config.target.desired_through_water_speed_kt = 12.0
    run.run(40)
    event = run.engine.timing.events[0]
    engine = run.engine
    truth = run.truth_track[-1]
    p = engine.frame.to_local(truth.position)
    v = truth.ground_velocity
    ground = np.array([v.east_kt * 0.5144444444444445, v.north_kt * 0.5144444444444445, 0.0])
    state = np.concatenate([p, ground, [0.0]])[None]
    zero = CurrentFit.zero()
    angles = np.radians(np.arange(0, 360, 45))
    shifted = state + np.stack([1500 * np.sin(angles), 1500 * np.cos(angles)] + [np.zeros(8)] * 5, axis=1)
    pf = engine.pf
    at_truth = pf.timing_loglik(state, [event], zero)[0]
    at_shift = pf.timing_loglik(shifted, [event], zero)
    assert at_truth > -2.0 * len(event.onsets)
    # 0.3 s of onset error is ~450 m of range difference: a weak but real constraint
    assert (at_shift < at_truth).all() and at_shift.mean() < at_truth - 1.5
