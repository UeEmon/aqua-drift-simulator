"""Acoustic source / Doppler engine container (音源演算部).

Every synchronized 1 s epoch it produces, for every active observer, the error-free received
frequency when the target is inside the common maximum slant range, and an explicit
non-detection otherwise (no missed detections inside the range). Every bearing.interval_s it
also adds a horizontal true bearing with normal error (sigma_deg), independent in time and
between observers. When the optional Lloyd's mirror calculation is enabled (config.lloyd),
it also adds the received level of the tonal (direct + surface-reflected path with level
fluctuation). Truth (slant range, relative speed) is attached separately and is stripped
before reaching the estimator.
"""
from __future__ import annotations

import asyncio
import random

import httpx

from aqua_drift.models import DopplerBatch, ObserverRecord, ScenarioConfig, TargetState
from aqua_drift.physics import LevelNoise, doppler_observation
from aqua_drift.services.common import poll_interval, post, snapshot, wait_for_api


async def run() -> None:
    last_tick = -1
    rng: random.Random | None = None
    level_noise: LevelNoise | None = None
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            if tick <= last_tick or not data.get("target"):
                await asyncio.sleep(poll_interval(data))
                continue
            config = ScenarioConfig.model_validate(data["config"])
            if rng is None:
                rng = random.Random(config.bearing.random_seed)
            if level_noise is None:
                level_noise = LevelNoise(config.lloyd.random_seed)
            target = TargetState.model_validate(data["target"])
            records = [ObserverRecord.model_validate(item) for item in data["observers"]]
            # wait until target and every observer have published this epoch (time sync)
            if target.tick < tick or not records or any(r.state.tick < tick for r in records):
                await asyncio.sleep(poll_interval(data))
                continue
            if tick % config.doppler_interval_seconds == 0:
                observations, truth = [], []
                for record in records:
                    obs, tr = doppler_observation(config, target, record.state, rng, level_noise)
                    obs.tick = tick
                    tr.tick = tick
                    observations.append(obs)
                    truth.append(tr)
                batch = DopplerBatch(tick=tick, observations=observations, truth=truth)
                response = await post(client, "/internal/doppler", batch.model_dump(mode="json"))
                response.raise_for_status()
            last_tick = tick
            await asyncio.sleep(poll_interval(data))


if __name__ == "__main__":
    asyncio.run(run())
