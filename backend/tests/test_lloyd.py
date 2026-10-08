import json
import math

import numpy as np

from aqua_drift.analysis.wire_fixture import build_stream
from aqua_drift.deployment import _offset
from aqua_drift.estimation.engine import TrackingEngine
from aqua_drift.estimation.lloyd import LloydDepthEstimator, LloydFitSettings
from aqua_drift.models import EstimatorSettings, ObserverState, Position, ScenarioConfig
from aqua_drift.physics import (
    LevelNoise,
    doppler_observation,
    initial_target,
    lloyd_mirror_level_db,
    local_offset_m,
)
from aqua_drift.scenario import ScenarioRun

FT = 0.3048
ORIGIN = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)


def _settings(window_s: float = 600) -> LloydFitSettings:
    return LloydFitSettings(
        sound_speed=1500.0, max_depth_m=1500 * FT, depth_step_m=2 * FT, min_samples=120,
        window_s=window_s, model_error_frac=0.03, noise_correlation_s=20.0,
    )


def _pass(cross_m: float, noise_db: float = 2.0, seconds: int = 600) -> tuple[LloydDepthEstimator, callable]:
    config = ScenarioConfig()
    config.lloyd.enabled = True
    observer = _offset(ORIGIN, 0.0, 0.0, 200.0)
    estimator = LloydDepthEstimator()
    noise = LevelNoise(5)
    speed = 4.1
    east, north, _ = local_offset_m(ORIGIN, observer)
    for t in range(seconds):
        target = _offset(ORIGIN, -1200 + speed * t, cross_m, 500.0)
        level, _ = lloyd_mirror_level_db(config, target, observer, 400.0)
        level += noise.sample("a", noise_db, 20.0)
        estimator.add("a", t, np.array([east, north, 200 * FT]), level, 400.0)

    def track(ticks: np.ndarray) -> np.ndarray:
        return np.stack([-1200 + speed * ticks, np.full_like(ticks, cross_m), np.zeros_like(ticks)], axis=1)

    return estimator, track


def test_level_has_interference_nulls_at_whole_wavelength_path_differences() -> None:
    config = ScenarioConfig()
    config.lloyd.enabled = True
    config.lloyd.wave_height_rms_m = 0.0  # flat sea: perfect reflection
    observer = _offset(ORIGIN, 0.0, 0.0, 200.0)
    zs, zr, wavelength = 500 * FT, 200 * FT, 1500.0 / 400.0
    levels = []
    ranges = np.arange(800.0, 6000.0, 2.0)
    for r in ranges:
        level, mu = lloyd_mirror_level_db(config, _offset(ORIGIN, r, 0.0, 500.0), observer, 400.0)
        levels.append(level)
        assert mu == 1.0
    levels = np.array(levels)
    # first null (path difference = one wavelength) ~ 2 zs zr / lambda from the far-field formula
    first_null = 2 * zs * zr / wavelength
    window = (ranges > 0.8 * first_null) & (ranges < 1.2 * first_null)
    at = ranges[window][int(np.argmin(levels[window]))]
    exact = [r for r in ranges if abs(math.hypot(r, zs + zr) - math.hypot(r, zs - zr) - wavelength) < 0.01]
    assert abs(at - first_null) < 0.05 * first_null
    assert exact and abs(at - exact[0]) < 30.0
    assert levels[window].max() - levels[window].min() > 20.0  # deep fade


def test_received_level_only_when_enabled_and_detected() -> None:
    config = ScenarioConfig()
    target = initial_target(config).model_copy(update={"position": _offset(ORIGIN, 1000.0, 0.0, 500.0)})
    near = ObserverState(observer_id="a", tick=1, position=_offset(ORIGIN, 0.0, 0.0, 200.0))
    far = ObserverState(observer_id="b", tick=1, position=_offset(ORIGIN, 20000.0, 0.0, 200.0))
    off, _ = doppler_observation(config, target, near)
    assert off.received_level_db is None
    config.lloyd.enabled = True
    on, _ = doppler_observation(config, target, near, level_noise=LevelNoise(1))
    out_of_range, _ = doppler_observation(config, target, far, level_noise=LevelNoise(1))
    assert on.received_level_db is not None
    assert out_of_range.received_level_db is None


def test_fit_recovers_depth_from_the_interference_pattern() -> None:
    estimator, track = _pass(cross_m=700.0)
    result = estimator.fit(599, track, _settings())
    assert result.status == "OK"
    assert abs(result.depth_ft - 500.0) < 15.0
    assert 5.0 < result.sigma_ft < 40.0  # includes the assumed 3 % sound-speed model error
    assert result.observers[0].fringes > 2


def test_fit_refuses_patterns_with_too_few_fringes() -> None:
    estimator, track = _pass(cross_m=2500.0)
    result = estimator.fit(599, track, _settings())
    assert result.depth_ft is None
    assert result.observers[0].status in {"NO_FRINGES", "AMBIGUOUS", "NO_PATTERN"}


def test_lloyd_switch_off_skips_the_fit_and_clears_state() -> None:
    config = ScenarioConfig()
    config.estimator.particle_count = 500
    config.lloyd.enabled = True
    run = ScenarioRun(config, 4)
    run.engine.process = lambda batch: None
    batches = [run.step() for _ in range(30)]
    engine = TrackingEngine(EstimatorSettings.from_config(config))
    for batch in batches[:20]:
        engine.process(batch)
    assert engine.lloyd.samples  # levels collected while on
    config.lloyd.enabled = False
    engine.apply_settings(EstimatorSettings.from_config(config))
    for batch in batches[20:]:
        engine.process(batch)
    assert not engine.lloyd.samples
    assert engine.output().lloyd.status == "OFF"
    assert engine.pf.depth_fix is None


def test_lloyd_depth_tightens_the_particle_filter_depth() -> None:
    config = ScenarioConfig()
    config.estimator.particle_count = 1500
    config.lloyd.enabled = True
    run = ScenarioRun(config, 4)
    run.run(420)
    output = run.engine.output()
    assert output.lloyd.status == "OK"
    assert abs(output.lloyd.depth_ft - run.target.position.depth_ft) < 40.0
    assert output.lloyd.applied
    online = next(e for e in output.estimates if e.mode.value == "ONLINE")
    # without the Lloyd's mirror the depth stays near the prior width (~400 ft, see
    # docs/depth-from-doppler.md); with it the particle filter depth is well determined
    assert online.uncertainty.depth_sigma_ft < 150.0
    assert abs(run.error(output)["depth_error_ft"]) < 100.0


def test_stream_carries_the_lloyd_result_only_when_it_changes() -> None:
    stream = build_stream(120, window_s=60)
    carried = [json.loads(raw).get("lly") for raw in stream["messages"] if "lly" in json.loads(raw)]
    assert carried, "the Lloyd result is sent at least once"
    assert len(carried) < len(stream["messages"]) / 2
