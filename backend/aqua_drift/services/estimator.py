"""Estimator container: pulls observation-only feed, runs the tracking engine, publishes
ONLINE (past track not updated) and SMOOTHED (past track updated) estimates."""
from __future__ import annotations

import asyncio
import logging
import time

import httpx

from aqua_drift.estimation.engine import TrackingEngine
from aqua_drift.models import EstimatorFeed
from aqua_drift.services.common import API_URL, post, wait_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s estimator %(message)s")
log = logging.getLogger(__name__)


async def run() -> None:
    engine: TrackingEngine | None = None
    generation = -1
    last_tick = -1
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            response = await client.get(
                f"{API_URL}/internal/estimator-feed", params={"after_tick": last_tick}, timeout=10
            )
            response.raise_for_status()
            feed = EstimatorFeed.model_validate(response.json())
            if engine is None or feed.generation != generation:
                engine = TrackingEngine(feed.settings)
                generation = feed.generation
                last_tick = -1
            engine.apply_settings(feed.settings)
            if not feed.batches:
                await asyncio.sleep(0.2)
                continue
            started = time.perf_counter()
            for batch in feed.batches:
                engine.process(batch)
                last_tick = batch.tick
            output = engine.output()
            posted = await post(client, "/internal/estimate", output.model_dump(mode="json"))
            posted.raise_for_status()
            if last_tick % 60 == 0:
                log.info(
                    "tick=%s status=%s batches=%s %.0f ms",
                    last_tick,
                    output.estimates[0].observability_status,
                    len(feed.batches),
                    (time.perf_counter() - started) * 1000,
                )
            await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(run())
