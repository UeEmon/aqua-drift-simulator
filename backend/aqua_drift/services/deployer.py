"""Forward-deployment container: places standby observers ahead (前程) of the ESTIMATED
target. It reads /internal/deployment-feed (estimates and observer positions; no truth)."""
from __future__ import annotations

import asyncio
import logging

import httpx

from aqua_drift.forward_deployment import plan_forward_deployment_scheduled
from aqua_drift.models import CancelSuggestions, DeploymentFeed, DeploymentRequest
from aqua_drift.optimal_deployment import availability_from_feed, sensor_from_feed
from aqua_drift.replanning import basis_for, cancel_suggestions, plan_replacement
from aqua_drift.services.common import API_URL, post, wait_for_api

logging.basicConfig(level=logging.INFO, format="%(asctime)s deployer %(message)s")
log = logging.getLogger(__name__)
CHECK_INTERVAL_S = 5.0


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        last_reason = last_replan = ""
        while True:
            response = await client.get(f"{API_URL}/internal/deployment-feed", timeout=10)
            response.raise_for_status()
            feed = DeploymentFeed.model_validate(response.json())
            estimate = next((e for e in feed.estimates if e.mode.value == "ONLINE"), None)
            if feed.replan is not None:
                # first: do the open drops still match the estimate? (replaced all or nothing)
                decision, why = plan_replacement(
                    feed.tick, estimate, feed.observer_positions, feed.replan.other_pending, feed.replan,
                    feed.config, feed.max_slant_range_yd, feed.free_slots, sensor_from_feed(feed),
                    feed.max_target_depth_ft,
                )
                if decision is not None:
                    request = DeploymentRequest(
                        tick=feed.tick, positions=decision.positions, reason=decision.reason,
                        planned_ticks=decision.planned_ticks, bases=decision.bases,
                        replaces=decision.replaces, revision=decision.revision,
                    )
                    result = await post(client, "/internal/deploy", request.model_dump(mode="json"), feed.generation)
                    if result.status_code == 409:  # a drop was laid or flown to meanwhile
                        log.info("tick=%s replan refused (%s); next cycle", feed.tick, result.text)
                    else:
                        result.raise_for_status()
                        log.info("tick=%s %s", feed.tick, decision.reason)
                    await asyncio.sleep(CHECK_INTERVAL_S)
                    continue
                if why != last_replan:
                    log.info("tick=%s no replan: %s", feed.tick, why)
                last_replan = why
                # the drops it keeps: propose cancelling those that no longer help detection (the
                # layer keeps flying them until the operator cancels)
                suggestions = cancel_suggestions(feed.tick, estimate, feed.replan, feed.config,
                                                 feed.max_slant_range_yd, feed.max_target_depth_ft)
                if suggestions:
                    body = CancelSuggestions(tick=feed.tick, suggestions=suggestions)
                    result = await post(client, "/internal/cancel-suggestions", body.model_dump(mode="json"),
                                        feed.generation)
                    result.raise_for_status()
                    for task_id in result.json()["changed"]:  # log new / withdrawn proposals once
                        why_cancel = suggestions[task_id]
                        log.info("tick=%s drop %s: %s", feed.tick, task_id,
                                 f"cancellation proposed: {why_cancel}" if why_cancel else "cancellation proposal withdrawn")
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
                request = DeploymentRequest(
                    tick=feed.tick, positions=positions, reason=reason, planned_ticks=planned,
                    bases=[basis_for(estimate, p) for p in positions],
                )
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
