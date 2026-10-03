from __future__ import annotations

import asyncio
from collections import OrderedDict, deque

from aqua_drift.models import (
    DopplerObservation,
    EstimateMode,
    ObserverRecord,
    ObserverState,
    ScenarioConfig,
    Snapshot,
    TargetState,
    TrackEstimate,
)


class ObserverRejected(RuntimeError):
    """Raised when an observer can no longer publish active observations."""


class SimulationState:
    def __init__(self, config: ScenarioConfig | None = None) -> None:
        self.lock = asyncio.Lock()
        self.config = config or ScenarioConfig()
        self.tick = 0
        self.target: TargetState | None = None
        self.observers: OrderedDict[str, ObserverRecord] = OrderedDict()
        self.archived_observer_ids: list[str] = []
        self._archived_set: set[str] = set()
        self.doppler: deque[DopplerObservation] = deque(maxlen=50_000)
        self.estimates: dict[EstimateMode, TrackEstimate] = {}

    async def set_config(self, config: ScenarioConfig) -> None:
        async with self.lock:
            self.config = config

    async def set_tick(self, tick: int) -> None:
        async with self.lock:
            self.tick = max(self.tick, tick)

    async def set_target(self, target: TargetState) -> None:
        async with self.lock:
            if self.target is None or target.tick >= self.target.tick:
                self.target = target

    async def set_observer(self, observer: ObserverState) -> str | None:
        async with self.lock:
            observer_id = observer.observer_id
            if observer_id in self._archived_set:
                raise ObserverRejected(f"observer {observer_id} is archived")

            evicted_id: str | None = None
            record = self.observers.get(observer_id)
            if record is None:
                if len(self.observers) >= self.config.observer_limit:
                    evicted_id, _ = self.observers.popitem(last=False)
                    self._archive(evicted_id)
                record = ObserverRecord(
                    state=observer,
                    registered_tick=observer.tick,
                    last_tick=observer.tick,
                )
                self.observers[observer_id] = record
            elif observer.tick - record.registered_tick >= self.config.max_observation_seconds:
                self.observers.pop(observer_id, None)
                self._archive(observer_id)
                raise ObserverRejected(f"observer {observer_id} exceeded its observation limit")
            else:
                record.state = observer
                record.last_tick = observer.tick
            return evicted_id

    async def add_doppler(self, observation: DopplerObservation) -> None:
        async with self.lock:
            self.doppler.append(observation)

    async def set_estimate(self, estimate: TrackEstimate) -> None:
        async with self.lock:
            previous = self.estimates.get(estimate.mode)
            if previous is None or estimate.tick >= previous.tick:
                self.estimates[estimate.mode] = estimate

    async def snapshot(self) -> Snapshot:
        async with self.lock:
            return Snapshot(
                tick=self.tick,
                config=self.config,
                target=self.target,
                observers=list(self.observers.values()),
                doppler=list(self.doppler)[-2_000:],
                estimates=list(self.estimates.values()),
                archived_observer_ids=list(self.archived_observer_ids),
            )

    async def reset_runtime(self) -> None:
        async with self.lock:
            self.target = None
            self.doppler.clear()
            self.estimates.clear()

    def _archive(self, observer_id: str) -> None:
        if observer_id not in self._archived_set:
            self._archived_set.add(observer_id)
            self.archived_observer_ids.append(observer_id)
