from __future__ import annotations

import asyncio

import httpx

from aqua_drift.models import ScenarioConfig, TargetState
from aqua_drift.physics import advance_target, initial_target
from aqua_drift.services.common import current_vector, post, snapshot, wait_for_api


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        state: TargetState | None = None
        last_tick = -1
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            config = ScenarioConfig.model_validate(data["config"])
            if state is None:
                state = initial_target(config)
                state.tick = tick
            if tick > last_tick:
                elapsed = max(1, tick - max(last_tick, 0))
                current = await current_vector(client, config, state.position)
                state = advance_target(config, state, elapsed, current)
                state.tick = tick
                response = await post(client, "/internal/target", state.model_dump(mode="json"))
                response.raise_for_status()
                last_tick = tick
            await asyncio.sleep(0.15)


if __name__ == "__main__":
    asyncio.run(run())
