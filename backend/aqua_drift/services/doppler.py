from __future__ import annotations

import asyncio

import httpx

from aqua_drift.models import ObserverRecord, ScenarioConfig, TargetState
from aqua_drift.physics import doppler_observation
from aqua_drift.services.common import post, snapshot, wait_for_api


async def run() -> None:
    minimum_range: dict[str, float] = {}
    last_tick = -1
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            if tick <= last_tick or not data.get("target"):
                await asyncio.sleep(0.15)
                continue
            config = ScenarioConfig.model_validate(data["config"])
            target = TargetState.model_validate(data["target"])
            records = [ObserverRecord.model_validate(item) for item in data["observers"]]
            if target.tick < tick or not records or any(record.state.tick < tick for record in records):
                await asyncio.sleep(0.15)
                continue
            for record in records:
                observer = record.state
                observation = doppler_observation(config, target, observer)
                if observation.slant_range_yd > config.max_slant_range_yd:
                    continue
                old_minimum = minimum_range.get(observer.observer_id, float("inf"))
                observation.is_new_closest = observation.slant_range_yd < old_minimum
                if observation.is_new_closest:
                    minimum_range[observer.observer_id] = observation.slant_range_yd
                response = await post(
                    client, "/internal/doppler", observation.model_dump(mode="json")
                )
                response.raise_for_status()
            last_tick = tick
            await asyncio.sleep(0.15)


if __name__ == "__main__":
    asyncio.run(run())
