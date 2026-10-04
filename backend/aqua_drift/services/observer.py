"""Observer container (one per observer). It moves only with the water (passive drift) and
knows its time, position and depth exactly. Initial position: OBSERVER_LAT/LON/DEPTH_FT env,
otherwise assigned by the API (queued placement, else the default pattern for the first
deployment.initial_count observers). Further containers wait as standby observers until the
forward deployment places them ahead of the estimated target."""
from __future__ import annotations

import asyncio
import os
import socket

import httpx

from aqua_drift.models import ObserverState, Position, ScenarioConfig
from aqua_drift.physics import advance_observer
from aqua_drift.services.common import API_URL, current_vector, post, snapshot, wait_for_api


async def initial_position(client: httpx.AsyncClient, observer_id: str) -> Position:
    """Fixed position from env, otherwise ask the API. HTTP 204 means standby: keep waiting
    until the forward deployment (or the operator) queues a placement."""
    lat, lon = os.getenv("OBSERVER_LAT"), os.getenv("OBSERVER_LON")
    if lat and lon:
        return Position(
            latitude=float(lat),
            longitude=float(lon),
            depth_ft=float(os.getenv("OBSERVER_DEPTH_FT", "200")),
        )
    while True:
        response = await client.get(
            f"{API_URL}/internal/observer/assignment", params={"observer_id": observer_id}, timeout=5
        )
        response.raise_for_status()
        if response.status_code == 200:
            return Position.model_validate(response.json())
        await asyncio.sleep(1.0)


async def run() -> None:
    observer_id = os.getenv("OBSERVER_ID") or socket.gethostname()
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        state: ObserverState | None = None
        last_tick = -1
        generation = None
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            config = ScenarioConfig.model_validate(data["config"])
            if generation is not None and data.get("generation") != generation:
                state = None  # runtime reset: return to the assigned start position
            generation = data.get("generation")
            if state is None:
                state = ObserverState(
                    observer_id=observer_id,
                    tick=tick,
                    position=await initial_position(client, observer_id),
                )
                last_tick = tick - 1
            if tick > last_tick:
                elapsed = max(1, tick - last_tick)
                current = await current_vector(client, config, state.position)
                state = advance_observer(config, state, elapsed, current)
                state.tick = tick
                response = await post(client, "/internal/observer", state.model_dump(mode="json"))
                if response.status_code == 410:
                    return  # evicted or 3 h observation limit reached; history retained
                response.raise_for_status()
                last_tick = tick
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(run())
