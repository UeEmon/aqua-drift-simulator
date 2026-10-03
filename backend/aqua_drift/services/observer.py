from __future__ import annotations

import asyncio
import hashlib
import math
import os
import socket

import httpx

from aqua_drift.models import ObserverState, Position, ScenarioConfig, Velocity
from aqua_drift.physics import advance_observer
from aqua_drift.services.common import current_vector, post, snapshot, wait_for_api


def observer_identity() -> tuple[str, float]:
    observer_id = os.getenv("OBSERVER_ID", socket.gethostname())
    digest = hashlib.sha256(observer_id.encode()).digest()
    angle = int.from_bytes(digest[:2], "big") / 65535.0 * 2.0 * math.pi
    return observer_id, angle


def initial_state(observer_id: str, angle: float, config: ScenarioConfig, tick: int) -> ObserverState:
    origin = config.target.initial_position
    radius_nm = float(os.getenv("OBSERVER_RADIUS_NM", "2.0"))
    north_nm = radius_nm * math.cos(angle)
    east_nm = radius_nm * math.sin(angle)
    latitude = origin.latitude + north_nm / 60.0
    longitude = origin.longitude + east_nm / max(60.0 * math.cos(math.radians(origin.latitude)), 1e-8)
    depth = float(os.getenv("OBSERVER_DEPTH_FT", str(100.0 + (angle / (2 * math.pi)) * 300.0)))
    return ObserverState(
        observer_id=observer_id,
        tick=tick,
        position=Position(latitude=latitude, longitude=longitude, depth_ft=depth),
        ground_velocity=Velocity(),
    )


async def run() -> None:
    observer_id, angle = observer_identity()
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        state: ObserverState | None = None
        last_tick = -1
        while True:
            data = await snapshot(client)
            tick = int(data["tick"])
            config = ScenarioConfig.model_validate(data["config"])
            if state is None:
                state = initial_state(observer_id, angle, config, tick)
            if tick > last_tick:
                elapsed = max(1, tick - max(last_tick, 0))
                current = await current_vector(client, config, state.position)
                state = advance_observer(config, state, elapsed, current)
                state.tick = tick
                response = await post(client, "/internal/observer", state.model_dump(mode="json"))
                if response.status_code == 410:
                    return
                response.raise_for_status()
                last_tick = tick
            await asyncio.sleep(0.15)


if __name__ == "__main__":
    asyncio.run(run())
