"""Observer container orchestration (variable container start-up).

Observer containers are started only when an observer is needed and removed when it is done,
instead of keeping idle standby containers in memory. Each container owns one observer slot
"obs-01" .. "obs-99"; a slot whose observer ended (3 h limit or evicted as the oldest) is free
again and the lowest free number is reused.

`plan()` is pure (testable without Docker); services/orchestrator.py applies the plan through
the Docker Engine API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SLOT_MIN = 1
SLOT_MAX = 99


def slot_name(number: int) -> str:
    return f"obs-{number:02d}"


def slot_number(name: str) -> int | None:
    if not name.startswith("obs-"):
        return None
    try:
        number = int(name[4:])
    except ValueError:
        return None
    return number if SLOT_MIN <= number <= SLOT_MAX else None


@dataclass
class ManagedContainer:
    slot: str
    container_id: str
    running: bool
    age_s: float  # seconds since the container was created


@dataclass
class Plan:
    start: list[str] = field(default_factory=list)  # slots to start containers for
    stop: list[str] = field(default_factory=list)  # container ids to stop (idle standby)
    evict_oldest: int = 0  # observers to evict so that slots become free


def free_slots(used: set[str], limit: int) -> list[str]:
    top = min(max(limit, SLOT_MIN), SLOT_MAX)
    return [slot_name(n) for n in range(SLOT_MIN, top + 1) if slot_name(n) not in used]


def plan(
    *,
    limit: int,
    active_ids: list[str],
    pending_placements: int,
    initial_remaining: int,
    containers: list[ManagedContainer],
    warm_standby: int = 0,
    idle_stop_after_s: float = 60.0,
    start_grace_s: float = 30.0,
) -> Plan:
    """Decide which observer containers to start / stop.

    demand  = observers that need a container now: queued placements + initial observers not
              yet assigned + the configured number of warm standby containers
    waiting = running managed containers whose slot is not (yet) an active observer: they are
              starting up or waiting for a placement and will take the next placements
    """
    result = Plan()
    active = set(active_ids)
    running = [c for c in containers if c.running]
    waiting = [c for c in running if c.slot not in active]
    demand = pending_placements + initial_remaining + warm_standby
    shortfall = demand - len(waiting)
    if shortfall > 0:
        used = active | {c.slot for c in running}
        slots = free_slots(used, limit)
        result.start = slots[:shortfall]
        missing = shortfall - len(result.start)
        if missing > 0 and len(active) >= min(limit, SLOT_MAX):
            result.evict_oldest = missing  # all slots busy: free the oldest (history kept)
    elif shortfall < 0:
        surplus = -shortfall
        idle = sorted(
            (c for c in waiting if c.age_s > max(idle_stop_after_s, start_grace_s)),
            key=lambda c: -c.age_s,
        )
        result.stop = [c.container_id for c in idle[:surplus]]
    return result
