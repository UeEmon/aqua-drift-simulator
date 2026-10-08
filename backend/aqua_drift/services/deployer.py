"""Forward-deployment container: places standby observers ahead (前程) of the ESTIMATED
target. It reads /internal/deployment-feed (estimates and observer positions; no truth)."""
from __future__ import annotations

import asyncio
import logging

import httpx

from aqua_drift.forward_deployment import plan_forward_deployment_scheduled
from aqua_drift.models import DeploymentFeed, DeploymentRequest
from aqua_drift.optimal_deployment import availability_from_feed, sensor_from_feed
from aqua_drift.services.common import API_URL, post, wait_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s deployer %(message)s")
log = logging.getLogger(__name__)
CHECK_INTERVAL_S = 5.0


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        last_reason = ""
        while True:
            response = await client.get(f"{API_URL}/internal/deployment-feed", timeout=10)
            response.raise_for_status()
            feed = DeploymentFeed.model_validate(response.json())
            estimate = next((e for e in feed.estimates if e.mode.value == "ONLINE"), None)
            positions, reason, planned = plan_forward_deployment_scheduled(
                feed.tick,
                estimate,
                feed.observer_positions,
                feed.pending_positions,
                feed.config,
                feed.max_slant_range_yd,
                feed.last_deploy_tick,
                feed.depth_step_ft,
                free_slots=feed.free_slots,
                source_frequency_hz=feed.source_frequency_hz,
                sound_speed_mps=feed.sound_speed_mps,
                frequency_sigma_hz=feed.frequency_sigma_hz,
                layer=availability_from_feed(feed),
                sensor=sensor_from_feed(feed),
                max_depth_ft=feed.max_target_depth_ft,
            )
            if positions:
                request = DeploymentRequest(tick=feed.tick, positions=positions, reason=reason, planned_ticks=planned)
                result = await post(client, "/internal/deploy", request.model_dump(mode="json"), feed.generation)
                result.raise_for_status()
                log.info("tick=%s deployed %d observers (%s); the orchestrator starts their containers",
                         feed.tick, len(positions), reason)
            elif reason != last_reason:
                log.info("tick=%s no deployment: %s", feed.tick, reason)
            last_reason = reason
            await asyncio.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    asyncio.run(run())
