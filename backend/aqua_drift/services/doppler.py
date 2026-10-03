"""Acoustic source / Doppler engine container (音源演算部).

Every synchronized 1 s epoch it produces, for every active observer, the error-free received
frequency when the target is inside the common maximum slant range, and an explicit
non-detection otherwise (no missed detections inside the range). Every bearing.interval_s it
also adds a horizontal true bearing with normal error (sigma_deg), independent in time and
between observers. Truth (slant range,
relative speed) is attached separately and is stripped before reaching the estimator.
"""
from __future__ import annotations

import asyncio
import random

import httpx

from aqua_drift.models import DopplerBatch, ObserverRecord, ScenarioConfig, TargetState
from aqua_drift.physics import doppler_observation
from aqua_drift.services.common import post, snapshot, wait_for_api


async def run() -> None:
    last_tick = -1
    rng: random.Random | None = None
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            if tick <= last_tick or not data.get("target"):
                await asyncio.sleep(0.1)
                continue
            config = ScenarioConfig.model_validate(data["config"])
            if rng is None:
                rng = random.Random(config.bearing.random_seed)
            target = TargetState.model_validate(data["target"])
            records = [ObserverRecord.model_validate(item) for item in data["observers"]]
            # wait until target and every observer have published this epoch (time sync)
            if target.tick < tick or not records or any(r.state.tick < tick for r in records):
                await asyncio.sleep(0.1)
                continue
            if tick % config.doppler_interval_seconds == 0:
                observations, truth = [], []
                for record in records:
                    obs, tr = doppler_observation(config, target, record.state, rng)
                    obs.tick = tick
                    tr.tick = tick
                    observations.append(obs)
                    truth.append(tr)
                batch = DopplerBatch(tick=tick, observations=observations, truth=truth)
                response = await post(client, "/internal/doppler", batch.model_dump(mode="json"))
                response.raise_for_status()
            last_tick = tick
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(run())
