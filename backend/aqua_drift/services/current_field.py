from __future__ import annotations

import httpx
from fastapi import FastAPI

from aqua_drift.models import Position, ScenarioConfig, Velocity
from aqua_drift.physics import current_at
from aqua_drift.services.common import snapshot

app = FastAPI(title="AQUA-DRIFT Current Field", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/vector", response_model=Velocity)
async def vector(latitude: float, longitude: float, depth_ft: float) -> Velocity:
    async with httpx.AsyncClient(trust_env=False) as client:
        data = await snapshot(client)
    config = ScenarioConfig.model_validate(data["config"])
    return current_at(
        config.current_field,
        Position(latitude=latitude, longitude=longitude, depth_ft=depth_ft),
    )
