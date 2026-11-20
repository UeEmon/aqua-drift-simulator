import pytest

from aqua_drift.deployment import _offset
from aqua_drift.drop_rate import drop_rate
from aqua_drift.models import DeploymentRequest, LayerState, LayerUpdate, Position, ScenarioConfig
from aqua_drift.state import SimulationState

DATUM = Position(latitude=35.0, longitude=140.0, depth_ft=0.0)


def test_rate_is_measured_over_the_window() -> None:
    # 6 drops in the last 30 min (and 4 before it): 12 per hour
    ticks = [100, 200, 300, 400] + [2100 + 300 * k for k in range(6)]
    rate = drop_rate(ticks, tick=3800, start_tick=0, window_s=1800)
    assert (rate.laid, rate.window_laid, rate.measured_s) == (10, 6, 1800.0)
    assert rate.rate_per_hour == pytest.approx(12.0)
    assert rate.limit is None and rate.limit_eta_s is None and rate.last_drop_tick == 3600


def test_a_run_shorter_than_the_window_is_measured_from_its_start() -> None:
    rate = drop_rate([700, 800, 900], tick=1200, start_tick=600, window_s=1800)
    assert rate.measured_s == 600.0 and rate.window_laid == 3
    assert rate.rate_per_hour == pytest.approx(18.0)
    # too short to measure: one early drop is no rate
    assert drop_rate([610], tick=630, start_tick=600, window_s=1800).rate_per_hour is None


def test_limit_estimate_at_the_measured_rate() -> None:
    ticks = [600 * k for k in range(1, 7)]  # 6 per hour over the last hour
    rate = drop_rate(ticks, tick=3600, start_tick=0, window_s=3600, limit=10, open_count=2)
    assert (rate.limit, rate.remaining, rate.open, rate.reached) == (10, 4, 2, False)
    assert rate.limit_eta_s == pytest.approx(2400.0) and rate.limit_tick == 6000
    # nothing laid in the window: no estimate
    idle = drop_rate([], tick=3600, start_tick=0, window_s=1800, limit=10)
    assert idle.rate_per_hour == 0.0 and idle.limit_eta_s is None and idle.remaining == 10
    # reached: when the last allowed drop was laid
    full = drop_rate([100, 200, 300], tick=900, start_tick=0, window_s=1800, limit=3)
    assert full.reached and full.remaining == 0 and full.limit_tick == 300 and full.limit_eta_s == 0.0


def _laid(sim: SimulationState, task_ids: list[int]) -> LayerUpdate:
    return LayerUpdate(
        state=LayerState(tick=sim.tick, position=DATUM, heading_deg=0.0, speed_kt=200.0),
        completed={i: _offset(DATUM, 1000.0 * i, 0.0, 200.0) for i in task_ids},
    )


@pytest.mark.asyncio
async def test_layer_stops_at_the_drop_limit() -> None:
    config = ScenarioConfig()
    config.layer.drop_limit = 3
    sim = SimulationState(config)
    points = [_offset(DATUM, 3000.0 * k, 2000.0, 200.0) for k in range(1, 5)]
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=points[:2], reason="plan"))
    # 3 allowed, 2 planned: the planner may add one more
    assert (await sim.deployment_feed()).free_slots == 1
    sim.tick = 300
    await sim.set_layer_update(_laid(sim, [1, 2]))
    status = (await sim.snapshot()).deployment
    assert status.drop_rate.laid == 2 and status.drop_rate.remaining == 1
    assert status.drop_rate.rate_per_hour == pytest.approx(24.0)  # 2 in 300 s
    assert status.drop_rate.limit_tick == 450
    await sim.queue_deployment(DeploymentRequest(tick=sim.tick, positions=points[2:], reason="plan"))
    # 2 open, 1 left: the layer is given only the first in its flight order and nothing more is planned
    assert [t.task_id for t in (await sim.layer_feed()).tasks] == [3]
    assert (await sim.deployment_feed()).free_slots == 0
    sim.tick = 400
    await sim.set_layer_update(_laid(sim, [3, 4]))  # one too many: not laid
    status = (await sim.snapshot()).deployment
    assert [t.status for t in status.tasks] == ["DONE", "DONE", "DONE", "APPROVED"]
    assert status.drop_rate.reached and status.drop_rate.limit_tick == 400
    assert (await sim.layer_feed()).tasks == []
    # a runtime reset starts a new count
    await sim.reset_runtime()
    rate = (await sim.snapshot()).deployment.drop_rate
    assert rate.laid == 0 and not rate.reached and rate.measured_s == 0.0


@pytest.mark.asyncio
async def test_no_limit_by_default() -> None:
    sim = SimulationState(ScenarioConfig())
    await sim.queue_deployment(DeploymentRequest(tick=0, positions=[_offset(DATUM, 3000.0, 0.0, 200.0)], reason="plan"))
    rate = (await sim.snapshot()).deployment.drop_rate
    assert rate.limit is None and rate.open == 1
    assert (await sim.deployment_feed()).free_slots > 50
