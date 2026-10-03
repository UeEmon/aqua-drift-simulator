from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from aqua_drift.models import (
    DopplerObservation,
    ObserverState,
    ScenarioConfig,
    Snapshot,
    TargetState,
    TickMessage,
    TrackEstimate,
)
from aqua_drift.state import ObserverRejected, SimulationState
from aqua_drift.storage import EventStore

initial_config = ScenarioConfig(
    max_slant_range_yd=float(os.getenv("MAX_SLANT_RANGE_YD", "12000"))
)
state = SimulationState(initial_config)
store = EventStore(os.getenv("DATABASE_URL"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.connect()
    await store.save_config(initial_config)
    yield
    await store.close()


app = FastAPI(title="AQUA-DRIFT API", version="0.1.0", lifespan=lifespan)
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


@app.get("/api/config", response_model=ScenarioConfig)
async def get_config() -> ScenarioConfig:
    return (await state.snapshot()).config


@app.put("/api/config", response_model=ScenarioConfig)
async def put_config(config: ScenarioConfig) -> ScenarioConfig:
    await state.set_config(config)
    await store.save_config(config)
    return config


@app.get("/api/snapshot", response_model=Snapshot)
async def get_snapshot() -> Snapshot:
    return await state.snapshot()


@app.post("/api/reset")
async def reset_runtime() -> dict[str, str]:
    await state.reset_runtime()
    await store.append_event("runtime_reset", 0, {"history_retained": True})
    return {"status": "reset", "history": "retained"}


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


@app.post("/internal/observer")
async def set_observer(observer: ObserverState) -> dict[str, str | None]:
    try:
        evicted = await state.set_observer(observer)
    except ObserverRejected as error:
        raise HTTPException(status_code=410, detail=str(error)) from error
    snapshot = await state.snapshot()
    record = next(item for item in snapshot.observers if item.state.observer_id == observer.observer_id)
    await store.upsert_observer(
        observer.observer_id,
        record.registered_tick,
        record.last_tick,
        observer.model_dump(mode="json"),
    )
    await store.append_event(
        "observer_state", observer.tick, observer.model_dump(mode="json"), observer.observer_id
    )
    await store.save_observer_position(
        observer.observer_id, observer.tick, observer.model_dump(mode="json")
    )
    if evicted:
        await store.archive_observer(evicted)
        await store.append_event("observer_evicted", observer.tick, {"observer_id": evicted}, evicted)
    return {"observer_id": observer.observer_id, "evicted_observer_id": evicted}


@app.post("/internal/doppler")
async def set_doppler(observation: DopplerObservation) -> dict[str, int]:
    await state.add_doppler(observation)
    await store.append_event(
        "doppler_observation",
        observation.tick,
        observation.model_dump(mode="json"),
        observation.observer_id,
    )
    return {"tick": observation.tick}


@app.post("/internal/estimate")
async def set_estimate(estimate: TrackEstimate) -> dict[str, int]:
    await state.set_estimate(estimate)
    await store.append_event(
        f"estimate_{estimate.mode.value.lower()}",
        estimate.tick,
        estimate.model_dump(mode="json"),
    )
    return {"tick": estimate.tick}


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
