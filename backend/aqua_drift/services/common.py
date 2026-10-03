from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from aqua_drift.models import Position, ScenarioConfig, Velocity
from aqua_drift.physics import current_at

API_URL = os.getenv("API_URL", "http://api:8000").rstrip("/")
CURRENT_FIELD_URL = os.getenv("CURRENT_FIELD_URL", "http://current-field:8001").rstrip("/")


async def wait_for_api(client: httpx.AsyncClient) -> None:
    while True:
        try:
            response = await client.get(f"{API_URL}/health", timeout=2)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)


async def snapshot(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get(f"{API_URL}/api/snapshot", timeout=5)
    response.raise_for_status()
    return response.json()


async def post(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> httpx.Response:
    return await client.post(f"{API_URL}{path}", json=payload, timeout=5)


async def current_vector(
    client: httpx.AsyncClient, config: ScenarioConfig, position: Position
) -> Velocity:
    try:
        response = await client.get(
            f"{CURRENT_FIELD_URL}/vector",
            params={
                "latitude": position.latitude,
                "longitude": position.longitude,
                "depth_ft": position.depth_ft,
            },
            timeout=2,
        )
        response.raise_for_status()
        return Velocity.model_validate(response.json())
    except httpx.HTTPError:
        return current_at(config.current_field, position)
