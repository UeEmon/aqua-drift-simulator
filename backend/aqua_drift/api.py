from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from aqua_drift.models import (
    DopplerBatch,
    EstimationControl,
    EstimatorFeed,
    EstimatorOutput,
    ObserverPlacement,
    ObserverState,
    Position,
    ScenarioConfig,
    SimState,
    Snapshot,
    TargetState,
    TickMessage,
)
from aqua_drift.state import ObserverRejected, SimulationState
from aqua_drift.storage import EventStore

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


@app.get("/internal/observer/assignment", response_model=Position)
async def observer_assignment(observer_id: str) -> Position:
    return await state.assign_position(observer_id)


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
    await websocket.accept()
    try:
        while True:
            snapshot = await state.snapshot()
            await websocket.send_json(snapshot.model_dump(mode="json"))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
