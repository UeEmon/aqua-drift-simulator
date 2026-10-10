"""Lost contact (失探知): COASTING -> LOST -> REACQUIRED and the coasting estimate."""
import numpy as np

from aqua_drift.estimation.engine import TrackingEngine
from aqua_drift.models import EstimatorSettings, ScenarioConfig
from aqua_drift.scenario import ScenarioRun


def _short_range_run(**estimator) -> ScenarioRun:
    """A 500 YD detection range: the target leaves the initial square after a few minutes."""
    config = ScenarioConfig()
    config.max_slant_range_yd = 500.0
    config.deployment.surround_radius_yd = 400.0
    config.estimator.particle_count = 800
    for key, value in estimator.items():
        setattr(config.estimator, key, value)
    return ScenarioRun(config, 4)


def test_lost_contact_goes_coasting_then_lost() -> None:
    run = _short_range_run(lost_timeout_s=120, coast_maneuver_fraction=0.3)
    statuses: list[str] = []
    fractions: list[float] = []
    for _ in range(500):
        batch = run.step()
        if run.engine.pf.initialized:
            fractions.append(run.engine.pf.maneuver_fraction)
        if run.tick % 5 == 0:
            statuses.append(run.engine.output().estimates[0].observability_status)
    assert not any(o.detected for o in batch.observations), "the target should have left the field"
    first_coast = next(i for i, s in enumerate(statuses) if s.startswith("COASTING_"))
    assert all(s.startswith(("NO_DETECTION", "TRACKING", "LOW_CONFIDENCE", "AMBIGUOUS")) for s in statuses[:first_coast])  # no LOST before COASTING
    assert statuses[-1] == "LOST"
    estimate = run.engine.output().estimates[0]
    assert estimate.metadata["lost_contacts"] == 1
    assert estimate.metadata["seconds_since_detection"] >= 120
    assert estimate.metadata["position_basis"].startswith("densest part")
    # while coasting, coast_maneuver_fraction replaces maneuver_fraction
    config = run.config.estimator
    assert fractions[0] == config.maneuver_fraction
    assert fractions[-1] == config.coast_maneuver_fraction


def test_short_gap_is_not_coasting() -> None:
    engine = TrackingEngine(EstimatorSettings.from_config(ScenarioConfig()))
    engine.last_detect_tick = 100
    assert not engine._coasting(105)
    assert engine._coasting(100 + engine.settings.estimator.lost_debounce_s)


def _engine_with_cloud(points: np.ndarray) -> TrackingEngine:
    engine = TrackingEngine(EstimatorSettings.from_config(ScenarioConfig()))
    pf = engine.pf
    pf.x = np.zeros((len(points), pf.STATE_DIM))
    pf.x[:, 0:2] = points
    pf.w = np.full(len(points), 1.0 / len(points))
    pf.n = len(points)
    pf.initialized = True
    return engine


def test_densest_point_of_a_ring_is_on_the_ring() -> None:
    """Non-detection pushes the particles out of a detection circle: the mean of the ring lies
    inside it (where the target is known not to be), the densest part does not."""
    rng = np.random.default_rng(1)
    r = 5000.0
    angle = np.concatenate([rng.uniform(0, 2 * np.pi, 600), rng.normal(0.5, 0.2, 400)])
    ring = np.stack([r * np.sin(angle), r * np.cos(angle)], axis=1)
    engine = _engine_with_cloud(ring)
    point = engine._densest(engine.pf.x, engine.pf.w)
    assert np.hypot(*(engine.pf.w @ ring)) < 0.5 * r
    assert np.hypot(point[0], point[1]) > 0.7 * r


def test_lost_is_released_only_after_detections_last_and_the_track_converges() -> None:
    engine = _engine_with_cloud(np.random.default_rng(2).normal(0, 50.0, (500, 2)))
    e = engine.settings.estimator
    engine.lost = True
    engine.last_detect_tick = 1000
    engine.detect_since = 1000
    engine._update_contact(1000 + e.recover_hold_s - 1)
    assert engine.lost  # detections have not lasted long enough
    engine._update_contact(1000 + e.recover_hold_s)
    assert not engine.lost

    wide = _engine_with_cloud(np.random.default_rng(3).normal(0, 3000.0, (500, 2)))
    wide.lost, wide.last_detect_tick, wide.detect_since = True, 1000, 900
    wide._update_contact(1000)
    assert wide.lost  # detecting, but the cloud is still far wider than 0.15 R
