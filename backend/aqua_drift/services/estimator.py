from __future__ import annotations

import asyncio
from statistics import fmean

import httpx

from aqua_drift.models import (
    DopplerObservation,
    EstimateMode,
    ObserverRecord,
    PresenceRegion,
    PresenceRegionComponent,
    ScenarioConfig,
    TrackEstimate,
)
from aqua_drift.services.common import post, snapshot, wait_for_api


def build_estimate(data: dict, mode: EstimateMode) -> TrackEstimate:
    config = ScenarioConfig.model_validate(data["config"])
    tick = int(data["tick"])
    minimum_tick = (
        max(0, tick - config.smoothing_window_seconds)
        if mode == EstimateMode.SMOOTHED
        else tick
    )
    observers = {
        record.state.observer_id: record.state
        for record in (ObserverRecord.model_validate(item) for item in data["observers"])
    }
    latest: dict[str, DopplerObservation] = {}
    closest: dict[str, DopplerObservation] = {}
    for raw in data["doppler"]:
        item = DopplerObservation.model_validate(raw)
        if item.tick < minimum_tick:
            continue
        latest[item.observer_id] = item
        previous_closest = closest.get(item.observer_id)
        if mode == EstimateMode.SMOOTHED and (
            previous_closest is None or item.slant_range_yd < previous_closest.slant_range_yd
        ):
            closest[item.observer_id] = item
    selected = closest if mode == EstimateMode.SMOOTHED else latest
    components = []
    speeds = []
    for observer_id, observation in selected.items():
        observer = observers.get(observer_id)
        if not observer:
            continue
        speeds.append(observation.relative_speed_kt)
        components.append(
            PresenceRegionComponent(
                observer_id=observer_id,
                center=observer.position,
                radius_yd=observation.slant_range_yd,
                description="Range-only candidate shell; direction data is unavailable.",
            )
        )
    status = "NO_OBSERVATION" if not components else "UNOBSERVABLE_WITHOUT_DIRECTION"
    return TrackEstimate(
        mode=mode,
        tick=tick,
        observability_status=status,
        relative_speed_kt=fmean(speeds) if speeds else None,
        presence_region=PresenceRegion(
            probability_pct=config.presence_probability_pct,
            components=components,
            disconnected=len(components) > 1,
        ),
        metadata={
            "direction_input_available": False,
            "smoothing_window_seconds": (
                config.smoothing_window_seconds if mode == EstimateMode.SMOOTHED else 0
            ),
            "note": "Absolute horizontal position is intentionally not asserted without direction data.",
        },
    )


async def run() -> None:
    last_tick = -1
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            if tick <= last_tick:
                await asyncio.sleep(0.2)
                continue
            if not any(int(item["tick"]) == tick for item in data["doppler"]):
                await asyncio.sleep(0.2)
                continue
            for mode in (EstimateMode.ONLINE, EstimateMode.SMOOTHED):
                estimate = build_estimate(data, mode)
                response = await post(
                    client, "/internal/estimate", estimate.model_dump(mode="json")
                )
                response.raise_for_status()
            last_tick = tick
            await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(run())
