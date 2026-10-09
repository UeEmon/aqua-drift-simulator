"""Layer container (設標者): flies over the sea surface and lays the additional observers.

Every synchronized 1 s epoch it reads /internal/layer-feed (approved drop tasks, the
estimated target position as orbit centre, the estimated current), moves the layer
(aqua_drift.layer: 200 +- 50 kt, bank <= 15 deg), lets the planned drop points drift with the
estimated current (they are planned in the water frame) and reports its state. When it
reaches a drop point the task is complete and the API queues the observer placement (the
orchestrator then starts that observer's container).
"""
from __future__ import annotations

import asyncio
import logging
import random

import httpx

from aqua_drift.layer import advance, initial_state
from aqua_drift.models import LayerFeed
from aqua_drift.services.common import API_URL, post, wait_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s layer %(message)s")
log = logging.getLogger(__name__)


async def run() -> None:
    rng: random.Random | None = None
    state = None
    generation = None
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            response = await client.get(f"{API_URL}/internal/layer-feed", timeout=10)
            response.raise_for_status()
            feed = LayerFeed.model_validate(response.json())
            if not feed.config.enabled:
                state = None
                await asyncio.sleep(1.0)
                continue
            if rng is None or generation != feed.generation:
                rng = random.Random(feed.config.random_seed)  # runtime reset: new run
            if generation != feed.generation or state is None:
                generation = feed.generation
                state = feed.state if feed.state is not None and feed.state.tick <= feed.tick else None
                if state is None:
                    state = initial_state(feed.config, feed.datum, feed.tick, rng)
                    log.info("layer starts circling the datum at %.0f kt", state.speed_kt)
            if feed.tick <= state.tick:
                await asyncio.sleep(0.2)
                continue
            state, update = advance(feed, state, rng)
            for task_id in update.completed:
                log.info("tick=%s laid the observer of drop task %s", state.tick, task_id)
            result = await post(client, "/internal/layer", update.model_dump(mode="json"), generation)
            result.raise_for_status()
            await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(run())
