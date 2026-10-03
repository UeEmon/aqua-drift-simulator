import math

import numpy as np
import pytest

from aqua_drift.estimation.cpa import CpaAnalyzer
from aqua_drift.estimation.current_fit import CurrentFieldEstimator
from aqua_drift.estimation.frame import LocalFrame
from aqua_drift.estimation.region import presence_region
from aqua_drift.models import EstimateMode, Position, ScenarioConfig
from aqua_drift.scenario import ScenarioRun


def _config(**estimator) -> ScenarioConfig:
    config = ScenarioConfig()
    config.estimator.particle_count = estimator.pop("particle_count", 3000)
    for key, value in estimator.items():
        setattr(config.estimator, key, value)
    return config


def test_no_estimate_before_detection() -> None:
    config = _config()
    config.max_slant_range_yd = 500.0
    run = ScenarioRun(config, 2)
    output = run.run(5)
    assert output.estimates[0].observability_status == "NO_DETECTION"
    assert output.estimates[0].current_position is None


def test_doppler_only_track_converges_through_observer_field() -> None:
    config = _config(use_bearing=False)
    config.deployment.pattern = "grid"
    run = ScenarioRun(config, 4)
    output = run.run(1300)
    error = run.error(output)
    estimate = output.estimates[0]
    assert estimate.metadata["direction_input_available"] is False
    assert error["horizontal_error_yd"] < 600
    assert abs(error["cog_error_deg"]) < 10
    assert abs(error["ground_speed_error_kt"]) < 1.5
    assert abs(error["water_speed_error_kt"]) < 1.5
    assert estimate.presence_region.components
    total = sum(c.probability_mass_pct for c in estimate.presence_region.components)
    assert total == pytest.approx(config_pct(run), abs=8)


def config_pct(run: ScenarioRun) -> float:
    return run.config.presence_probability_pct


def test_online_track_frozen_and_smoothed_track_updated() -> None:
    config = _config()
    config.smoothing_window_seconds = 300
    run = ScenarioRun(config, 4)
    first = run.run(900)
    online_before = {p.tick: p for p in first.estimates[0].track[:-1]}
    smoothed_before = {p.tick: p for p in first.estimates[1].track[:-1]}
    second = run.run(120)
    online_after = {p.tick: p for p in second.estimates[0].track[:-1]}
    smoothed_after = {p.tick: p for p in second.estimates[1].track[:-1]}
    common = sorted(set(online_before) & set(online_after))
    assert common
    for tick in common:  # past track NOT updated
        assert online_after[tick].latitude == online_before[tick].latitude
    recent = [t for t in smoothed_before if t > first.tick - 300 and t in smoothed_after]
    old = [t for t in smoothed_before if t < second.tick - 300 and t in smoothed_after]
    assert any(smoothed_after[t].latitude != smoothed_before[t].latitude for t in recent)
    for tick in old:  # outside the recompute window -> frozen
        assert smoothed_after[tick].latitude == smoothed_before[tick].latitude
    assert second.estimates[0].mode == EstimateMode.ONLINE
    assert second.estimates[1].mode == EstimateMode.SMOOTHED


def test_cpa_range_speed_and_bias_shift() -> None:
    c, f0, v, r, tc = 1500.0, 400.0, 4.0, 1200.0, 600.0
    analyzer = CpaAnalyzer()
    bias = 0.2
    for t in range(1201):
        dt = t - tc
        rng = math.sqrt(r * r + (v * dt) ** 2)
        f = f0 * (1 - v * v * dt / (c * rng))
        analyzer.add("a", t, True, f, f0 + bias)
    analyzer.add("a", 1201, False, None, f0 + bias)
    result = analyzer.results(1201, c, 0.2, 0.03, 600, 30)[0]
    slope = f0 * v * v / (c * r)
    assert result.final
    # common bias shifts the CPA time by ~ b / |df/dt|
    assert result.cpa_tick == pytest.approx(tc - bias / slope, abs=3.0)
    assert result.relative_speed_kt * 0.514444 == pytest.approx(v, rel=0.05)
    assert result.cpa_slant_range_yd * 0.9144 == pytest.approx(r, rel=0.25)
    assert result.cpa_tick_sigma_s > 0
    assert result.cpa_slant_range_sigma_yd > 0
    # the free-frequency curve fit recovers the true source frequency (i.e. the bias)
    assert result.fit_source_frequency_hz == pytest.approx(f0, abs=0.01)
    assert result.fit_cpa_tick == pytest.approx(tc, abs=2.0)


def test_cpa_exact_without_bias() -> None:
    c, f0, v, r, tc = 1500.0, 400.0, 4.0, 1200.0, 600.0
    analyzer = CpaAnalyzer()
    for t in range(1201):
        dt = t - tc
        rng = math.sqrt(r * r + (v * dt) ** 2)
        analyzer.add("a", t, True, f0 * (1 - v * v * dt / (c * rng)), f0)
    analyzer.add("a", 1201, False, None, f0)
    result = analyzer.results(1201, c, 0.0, 0.03, 600, 30)[0]
    assert result.cpa_tick == pytest.approx(tc, abs=0.5)
    assert result.relative_speed_kt * 0.514444 == pytest.approx(v, rel=0.01)
    assert result.cpa_slant_range_yd * 0.9144 == pytest.approx(r, rel=0.02)


def test_current_field_fit_recovers_affine_field() -> None:
    base = np.array([0.5, 0.2, 0.0])
    gradient = np.array([[1e-4, 0, 0], [0, -5e-5, 0], [0, 0, 0]])
    estimator = CurrentFieldEstimator(sample_stride_s=1)
    starts = [np.array([0, 0, 50.0]), np.array([3000, 0, 100.0]), np.array([0, 3000, 70.0])]
    for index, start in enumerate(starts):
        p = start.copy()
        for t in range(120):
            estimator.add_fix(str(index), t, p)
            p = p + base + gradient @ p
    fit = estimator.fit(119, 120)
    for start in starts:
        predicted = fit.velocity(start[None])[0]
        assert np.allclose(predicted[:2], (base + gradient @ start)[:2], atol=0.03)


def test_presence_region_mass_and_split_modes() -> None:
    frame = LocalFrame(Position(latitude=35, longitude=140, depth_ft=0))
    rng = np.random.default_rng(0)
    a = rng.normal([0, 0, 100], [100, 100, 20], size=(3000, 3))
    b = rng.normal([5000, 0, 100], [100, 100, 20], size=(3000, 3))
    points = np.vstack([a, b])
    weights = np.full(len(points), 1 / len(points))
    region = presence_region(frame, points, weights, 90.0)
    assert region.disconnected
    assert len(region.components) == 2
    assert sum(c.probability_mass_pct for c in region.components) >= 90.0


def test_common_frequency_bias_is_estimated() -> None:
    config = _config()
    config.source.shared_recognition_bias_hz = 0.2
    config.deployment.pattern = "ring"
    config.deployment.spacing_yd = 2500
    run = ScenarioRun(config, 8)
    output = run.run(800)
    estimate = output.estimates[0]
    assert estimate.source_bias_hz == pytest.approx(0.2, abs=0.06)
    assert run.error(output)["horizontal_error_yd"] < 300


def test_track_recovers_after_maneuver() -> None:
    config = _config()
    config.deployment.pattern = "ring"
    config.deployment.spacing_yd = 2500
    run = ScenarioRun(config, 12)
    run.run(500)
    run.config.target.desired_hdg_deg = 180.0
    run.config.target.desired_through_water_speed_kt = 12.0
    output = run.run(300)
    error = run.error(output)
    assert error["horizontal_error_yd"] < 300
    assert abs(error["hdg_error_deg"]) < 8
    assert abs(error["water_speed_error_kt"]) < 1.5


def test_bearings_resolve_collinear_mirror_ambiguity() -> None:
    config = _config()
    config.deployment.pattern = "line"
    config.deployment.depth_step_ft = 0.0
    run = ScenarioRun(config, 4)
    output = run.run(1000)
    estimate = output.estimates[0]
    assert estimate.metadata["direction_input_available"] is True
    assert not estimate.presence_region.disconnected
    assert run.error(output)["horizontal_error_yd"] < 600


def test_default_surround_with_bearings_tracks_from_start() -> None:
    run = ScenarioRun(_config(), 4)
    output = run.run(300)
    assert run.error(output)["horizontal_error_yd"] < 250
