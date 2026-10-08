from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from aqua_drift.forward_deployment import plan_forward_deployment_scheduled
from aqua_drift.models import (
    ClockSettings,
    ClockStatus,
    DeploymentFeed,
    DeploymentRecord,
    DeploymentRequest,
    DopplerBatch,
    DropDecision,
    DropReorder,
    DropReschedule,
    DropTask,
    EstimationControl,
    EstimatorFeed,
    EstimatorOutput,
    LayerFeed,
    LayerUpdate,
    ObserverAssignment,
    ObserverPlacement,
    ObserverState,
    OrchestratorFeed,
    ScenarioConfig,
    SimState,
    Snapshot,
    TargetState,
    TickMessage,
)
from aqua_drift.optimal_deployment import availability_from_feed, sensor_from_feed
from aqua_drift.state import ObserverRejected, SimulationState
from aqua_drift.storage import EventStore
from aqua_drift.wire import PROTOCOL_VERSION, WireEncoder

initial_config = ScenarioConfig(
    max_slant_range_yd=float(os.getenv("MAX_SLANT_RANGE_YD", "6000"))
)
state = SimulationState(
    initial_config,
    autostart_estimation=os.getenv("ESTIMATION_AUTOSTART", "true").lower() in ("1", "true", "yes"),
)
store = EventStore(os.getenv("DATABASE_URL"))
ESTIMATE_EVENT_INTERVAL_S = 10


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.connect()
    await store.save_config(initial_config)
    yield
    await store.close()


app = FastAPI(title="AQUA-DRIFT API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ------------------------------------------------------------------ public API
@app.get("/api/config", response_model=ScenarioConfig)
async def get_config() -> ScenarioConfig:
    return (await state.snapshot()).config


@app.put("/api/config", response_model=ScenarioConfig)
async def put_config(config: ScenarioConfig) -> ScenarioConfig:
    await state.set_config(config)
    await store.save_config(config)
    await store.append_event("config_changed", state.tick, config.model_dump(mode="json"))
    return config


@app.get("/api/snapshot", response_model=Snapshot)
async def get_snapshot() -> Snapshot:
    return await state.snapshot()


@app.get("/api/clock", response_model=ClockSettings)
async def get_clock() -> ClockSettings:
    return ClockSettings(time_scale=(await state.clock_status()).time_scale)


@app.put("/api/clock", response_model=ClockSettings)
async def put_clock(settings: ClockSettings) -> ClockSettings:
    """Simulation speed: 1 = real time, 2 = twice as fast, 0.5 = half, 0 = pause. Above 1x the
    clock never starts a tick before every container finished the previous one, so the
    achieved rate can stay below the requested one on a slow machine."""
    await state.set_time_scale(settings.time_scale)
    await store.append_event("clock_changed", state.tick, settings.model_dump(mode="json"))
    return settings


@app.post("/api/observers/placements")
async def queue_observer_placement(placement: ObserverPlacement) -> dict[str, int]:
    """Queue an explicit position for the next observer container that starts
    (e.g. `docker compose up -d --scale observer=N+1`)."""
    pending = await state.queue_placement(placement)
    return {"pending_placements": pending}


@app.post("/api/estimation/start", response_model=EstimationControl)
async def start_estimation() -> EstimationControl:
    control = await state.start_estimation()
    await store.append_event("estimation_started", state.tick, control.model_dump(mode="json"))
    return control


@app.post("/api/estimation/stop", response_model=EstimationControl)
async def stop_estimation() -> EstimationControl:
    control = await state.stop_estimation()
    await store.append_event("estimation_stopped", state.tick, control.model_dump(mode="json"))
    return control


@app.get("/api/estimation", response_model=EstimationControl)
async def get_estimation() -> EstimationControl:
    return (await state.snapshot()).estimation


@app.post("/api/reset")
async def reset_runtime(replace_observers: bool = True) -> dict[str, str]:
    """Restart from the configured initial target state. With replace_observers the
    default-placed observers are re-placed around the new initial target position."""
    await state.reset_runtime(replace_observers)
    await store.append_event("runtime_reset", state.tick, {"history_retained": True})
    return {"status": "reset", "history": "retained"}


# ------------------------------------------------------------------ internal (containers)
@app.get("/internal/sim-state", response_model=SimState)
async def sim_state() -> SimState:
    return await state.sim_state()


@app.get("/internal/clock", response_model=ClockStatus)
async def clock_status() -> ClockStatus:
    return await state.clock_status()


@app.post("/internal/clock")
async def set_clock(message: TickMessage) -> dict[str, int]:
    await state.set_tick(message.tick)
    return {"tick": message.tick}


@app.post("/internal/target")
async def set_target(target: TargetState) -> dict[str, int]:
    await state.set_target(target)
    payload = target.model_dump(mode="json")
    await store.append_event("target_state", target.tick, payload, "target")
    await store.save_target_position(target.tick, payload)
    return {"tick": target.tick}


@app.get("/internal/observer/assignment", response_model=None)
async def observer_assignment(observer_id: str) -> ObserverAssignment | Response:
    """Start position and session for an observer container; 204 = stay in standby."""
    assignment = await state.assign_position(observer_id)
    if assignment is None:
        return Response(status_code=204)
    return assignment


@app.get("/internal/orchestrator-feed", response_model=OrchestratorFeed)
async def orchestrator_feed() -> OrchestratorFeed:
    return await state.orchestrator_feed()


@app.post("/internal/observers/evict-oldest")
async def evict_oldest(count: int = 1) -> dict[str, list[str]]:
    """Free observer slots (oldest first) when all 99 are in use; history is kept."""
    evicted = await state.evict_oldest(count)
    for observer_id in evicted:
        await store.archive_observer(observer_id)
        await store.append_event("observer_evicted", state.tick, {"observer_id": observer_id}, observer_id)
    return {"evicted": evicted}


@app.get("/internal/deployment-feed", response_model=DeploymentFeed)
async def deployment_feed() -> DeploymentFeed:
    """Estimates and observer positions for the forward deployer (no truth)."""
    return await state.deployment_feed()


@app.post("/api/deployment/now")
async def deploy_now() -> dict[str, object]:
    """Operator request: deploy observers ahead of the current estimate immediately."""
    feed = await state.deployment_feed()
    estimate = next((e for e in feed.estimates if e.mode.value == "ONLINE"), None)
    positions, reason, planned = plan_forward_deployment_scheduled(
        feed.tick, estimate, feed.observer_positions, feed.pending_positions, feed.config,
        feed.max_slant_range_yd, feed.last_deploy_tick, feed.depth_step_ft, force=True,
        free_slots=feed.free_slots, source_frequency_hz=feed.source_frequency_hz,
        sound_speed_mps=feed.sound_speed_mps, frequency_sigma_hz=feed.frequency_sigma_hz,
        layer=availability_from_feed(feed), sensor=sensor_from_feed(feed),
        max_depth_ft=feed.max_target_depth_ft,
    )
    if not positions:
        raise HTTPException(status_code=409, detail=f"no deployment: {reason}")
    record = await state.queue_deployment(
        DeploymentRequest(tick=feed.tick, positions=positions, reason=reason, planned_ticks=planned),
        source="operator",
    )
    await store.append_event("forward_deployment", feed.tick, record.model_dump(mode="json"))
    return {"deployed": len(positions), "standby": feed.standby_count, "tick": feed.tick}


@app.post("/api/drops/approve", response_model=list[DropTask])
async def approve_drops(decision: DropDecision) -> list[DropTask]:
    """Operator approves proposed drop points (empty list / null = all proposed); the layer
    (設標者) then flies there and lays the observers."""
    tasks = await state.decide_tasks(decision.task_ids, approve=True)
    await store.append_event("drop_decision", state.tick, {"approve": [t.task_id for t in tasks]})
    return tasks


@app.post("/api/drops/reject", response_model=list[DropTask])
async def reject_drops(decision: DropDecision) -> list[DropTask]:
    tasks = await state.decide_tasks(decision.task_ids, approve=False)
    await store.append_event("drop_decision", state.tick, {"reject": [t.task_id for t in tasks]})
    return tasks


@app.post("/api/drops/cancel", response_model=list[DropTask])
async def cancel_drops(decision: DropDecision) -> list[DropTask]:
    """Operator cancels open drops (proposed or approved; null = all open)."""
    tasks = await state.cancel_tasks(decision.task_ids)
    await store.append_event("drop_decision", state.tick, {"cancel": [t.task_id for t in tasks]})
    return tasks


@app.post("/api/drops/reschedule", response_model=DropTask)
async def reschedule_drop(request: DropReschedule) -> DropTask:
    """Operator changes the drop time of an open drop (planned_tick null = as soon as possible)."""
    task = await state.reschedule_task(request.task_id, request.planned_tick)
    if task is None:
        raise HTTPException(status_code=404, detail=f"no open drop task {request.task_id}")
    await store.append_event("drop_decision", state.tick, {"reschedule": request.model_dump()})
    return task


@app.post("/api/drops/reorder", response_model=list[DropTask])
async def reorder_drops(request: DropReorder) -> list[DropTask]:
    """Operator changes the drop order (設標順) of the open drops; the drop times are planned
    again along the new order. Returns the open drops in the new order."""
    tasks = await state.reorder_tasks(request.task_ids)
    if tasks is None:
        raise HTTPException(status_code=404, detail=f"not all of {request.task_ids} are open drop tasks")
    await store.append_event("drop_decision", state.tick, {
        "reorder": [t.task_id for t in tasks], "planned_ticks": [t.planned_tick for t in tasks],
    })
    return tasks


@app.get("/internal/layer-feed", response_model=LayerFeed)
async def layer_feed() -> LayerFeed:
    return await state.layer_feed()


@app.post("/internal/layer")
async def layer_update(update: LayerUpdate) -> dict[str, list[int]]:
    done = await state.set_layer_update(update)
    if done:
        await store.append_event("drop_done", update.state.tick, {"tasks": done})
    return {"done": done}


@app.post("/internal/deploy", response_model=DeploymentRecord)
async def deploy(request: DeploymentRequest) -> DeploymentRecord:
    record = await state.queue_deployment(request)
    await store.append_event("forward_deployment", request.tick, record.model_dump(mode="json"))
    return record


@app.post("/internal/observer")
async def set_observer(observer: ObserverState) -> dict[str, str | None]:
    try:
        evicted = await state.set_observer(observer)
    except ObserverRejected as error:
        await store.archive_observer(observer.observer_id)
        await store.append_event(
            "observer_expired", observer.tick, {"observer_id": observer.observer_id},
            observer.observer_id,
        )
        raise HTTPException(status_code=410, detail=str(error)) from error
    record = await state.observer_record(observer.observer_id)
    payload = observer.model_dump(mode="json")
    await store.upsert_observer(
        observer.observer_id, record.registered_tick, record.last_tick, payload
    )
    await store.append_event("observer_state", observer.tick, payload, observer.observer_id)
    await store.save_observer_position(observer.observer_id, observer.tick, payload)
    if evicted:
        await store.archive_observer(evicted)
        await store.append_event("observer_evicted", observer.tick, {"observer_id": evicted}, evicted)
    return {"observer_id": observer.observer_id, "evicted_observer_id": evicted}


@app.post("/internal/doppler")
async def set_doppler(batch: DopplerBatch) -> dict[str, int]:
    await state.add_batch(batch)
    await store.append_event("doppler_batch", batch.tick, batch.model_dump(mode="json"))
    return {"tick": batch.tick, "observations": len(batch.observations)}


@app.get("/internal/estimator-feed", response_model=EstimatorFeed)
async def estimator_feed(after_tick: int = -1) -> EstimatorFeed:
    """Observations only: no target truth, no true range, no true source frequency."""
    return await state.estimator_feed(after_tick)


@app.post("/internal/estimate")
async def set_estimate(output: EstimatorOutput) -> dict[str, int]:
    await state.set_estimator_output(output)
    if output.tick % ESTIMATE_EVENT_INTERVAL_S == 0:
        await store.append_event("estimate", output.tick, output.model_dump(mode="json"))
    return {"tick": output.tick}


@app.websocket("/ws")
async def websocket_snapshot(websocket: WebSocket) -> None:
    """GIS stream, protocol v2: first message full, then compact deltas (see aqua_drift.wire).
    `/api/snapshot` still returns the full snapshot for tools and tests."""
    await websocket.accept()
    encoder = WireEncoder()
    loop = asyncio.get_running_loop()
    last_tick = -1
    last_scale: float | None = None
    try:
        while True:
            snapshot = await state.snapshot()
            if snapshot.tick != last_tick:
                message = encoder.encode(snapshot, loop.time())
                await websocket.send_text(json.dumps(message, separators=(",", ":")))
                last_tick = snapshot.tick
            elif snapshot.time_scale != last_scale:
                # speed changed while the tick stands still (paused): clock-only message
                await websocket.send_text(json.dumps({"v": PROTOCOL_VERSION, "clk": snapshot.time_scale}))
            last_scale = snapshot.time_scale
            await asyncio.sleep(0.25)  # send each new tick promptly, never twice
    except WebSocketDisconnect:
        return
