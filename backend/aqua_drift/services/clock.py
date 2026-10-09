"""Clock container: advances the simulation one tick (1 s of simulation time) at a time.

The schedule is computed from the system time, not by adding up elapsed intervals: at speed
time_scale (PUT /api/clock) tick n is due at the system time anchor_wall + (n - anchor_tick) /
time_scale, where the anchor is the system time and tick at the start or the last speed change
(0 pauses). At real-time speed the simulation time (epoch + tick, see SimulationState.epoch_s)
therefore stays on the system time instead of drifting behind it by every slow tick. A tick
also waits until every container has finished the previous one (target, observers, Doppler
engine, estimator), so no epoch is skipped and each container still steps in 1 s increments;
that wait never holds a tick more than max(interval, 1 s). A tick that is late is caught up as
soon as the containers are done; more than MAX_LAG_S behind (host suspended, overloaded) the
schedule is re-anchored instead of bursting through the backlog."""
from __future__ import annotations

import asyncio
import time

import httpx

from aqua_drift.models import ClockStatus
from aqua_drift.services.common import API_URL, post, wait_for_api

PAUSED_POLL_S = 0.1
SYNC_POLL_S = 0.005
MAX_LAG_S = 30.0


async def clock_status(client: httpx.AsyncClient) -> ClockStatus:
    response = await client.get(f"{API_URL}/internal/clock", timeout=5)
    response.raise_for_status()
    return ClockStatus.model_validate(response.json())


class Schedule:
    """Due system time of each tick at the current speed, anchored to the system clock."""

    def __init__(self) -> None:
        self.scale: float | None = None
        self.anchor_wall = 0.0
        self.anchor_tick = 0

    def anchor(self, tick: int, scale: float, wall: float, epoch_s: float | None = None) -> None:
        self.scale = scale
        if scale == 1.0 and epoch_s is not None and abs(epoch_s + tick - wall) < 1.0:
            self.anchor_wall, self.anchor_tick = epoch_s, 0  # still on the system time: keep it
        else:
            self.anchor_wall, self.anchor_tick = wall, tick

    def due(self, tick: int) -> float:
        assert self.scale
        return self.anchor_wall + (tick - self.anchor_tick) / self.scale


async def run() -> None:
    async with httpx.AsyncClient(trust_env=False) as client:
        await wait_for_api(client)
        status = await clock_status(client)
        tick = status.tick
        schedule = Schedule()
        last = time.monotonic()  # when the last tick was posted
        while True:
            status = await clock_status(client)
            wall, now = time.time(), time.monotonic()
            tick = max(tick, status.tick)
            if status.time_scale <= 0:
                schedule.scale = None  # resume one interval after the pause ends
                last = now
                await asyncio.sleep(PAUSED_POLL_S)
                continue
            if status.time_scale != schedule.scale:
                schedule.anchor(tick, status.time_scale, wall, status.epoch_s)
            interval = 1.0 / status.time_scale
            due = schedule.due(tick + 1)
            if wall < due:
                await asyncio.sleep(min(due - wall, PAUSED_POLL_S))  # re-read the speed meanwhile
                continue
            if wall - due > MAX_LAG_S * max(1.0, interval):
                schedule.anchor(tick, status.time_scale, wall - interval)  # drop the backlog
            if not status.synced and now < last + max(interval, 1.0):
                await asyncio.sleep(SYNC_POLL_S)
                continue
            tick += 1
            response = await post(client, "/internal/clock", {"tick": tick, "wall_s": schedule.due(tick)})
            response.raise_for_status()
            last = now


if __name__ == "__main__":
    asyncio.run(run())
