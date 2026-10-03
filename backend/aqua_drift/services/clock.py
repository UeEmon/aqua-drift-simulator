from __future__ import annotations

import asyncio
import time

import httpx

from aqua_drift.services.common import post, snapshot, wait_for_api


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        tick = int((await snapshot(client))["tick"])
        next_tick = time.monotonic()
        while True:
            tick += 1
            response = await post(client, "/internal/clock", {"tick": tick})
            response.raise_for_status()
            next_tick += 1.0
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))


if __name__ == "__main__":
    asyncio.run(run())
