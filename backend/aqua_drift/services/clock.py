"""Clock container: advances the simulation one tick (1 s of simulation time) at a time.

The operator sets the speed (time_scale, PUT /api/clock): a tick starts every 1 / time_scale
wall seconds, and 0 pauses. Above 1x a tick also waits until every container has finished the
previous one (target, observers, Doppler engine, estimator), so no epoch is skipped and each
container still steps in 1 s increments; that wait never makes the clock slower than real
time. Speed changes apply at the next tick."""
from __future__ import annotations

import asyncio
import time

import httpx

from aqua_drift.models import ClockStatus
from aqua_drift.services.common import API_URL, post, wait_for_api

PAUSED_POLL_S = 0.1
SYNC_POLL_S = 0.005


async def clock_status(client: httpx.AsyncClient) -> ClockStatus:
    response = await client.get(f"{API_URL}/internal/clock", timeout=5)
    response.raise_for_status()
    return ClockStatus.model_validate(response.json())


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        tick = (await clock_status(client)).tick
        last = time.monotonic()
        while True:
            status = await clock_status(client)
            now = time.monotonic()
            if status.time_scale <= 0:
                last = now  # resume one interval after the pause ends
                await asyncio.sleep(PAUSED_POLL_S)
                continue
            interval = 1.0 / status.time_scale
            due = last + interval
            if now < due:
                await asyncio.sleep(min(due - now, PAUSED_POLL_S))  # re-read the speed meanwhile
                continue
            if not status.synced and now < last + max(interval, 1.0):
                await asyncio.sleep(SYNC_POLL_S)
                continue
            tick = max(tick, status.tick) + 1
            response = await post(client, "/internal/clock", {"tick": tick})
            response.raise_for_status()
            # keep the schedule exact, but never burst to catch up after a slow tick
            last = due if now - due < interval else now


if __name__ == "__main__":
    asyncio.run(run())
