"""Current-field container: serves the (truth) affine current v(p) = base + G (p - p_ref).

The field is held constant in time (no time-update rule was specified); it is re-read from
the API configuration at most once per second so parameter edits take effect promptly.
"""
from __future__ import annotations

import time

import httpx
from fastapi import FastAPI

from aqua_drift.models import Position, ScenarioConfig, Velocity
from aqua_drift.physics import current_at
from aqua_drift.services.common import API_URL

app = FastAPI(title="AQUA-DRIFT Current Field", version="0.2.0")
_cache: dict[str, object] = {"config": None, "at": 0.0}


async def _config() -> ScenarioConfig:
    now = time.monotonic()
    if _cache["config"] is None or now - float(_cache["at"]) > 1.0:
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(f"{API_URL}/api/config", timeout=5)
            response.raise_for_status()
        _cache["config"] = ScenarioConfig.model_validate(response.json())
        _cache["at"] = now
    return _cache["config"]  # type: ignore[return-value]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/vector", response_model=Velocity)
async def vector(latitude: float, longitude: float, depth_ft: float) -> Velocity:
    config = await _config()
    return current_at(
        config.current_field,
        Position(latitude=latitude, longitude=longitude, depth_ft=depth_ft),
    )
