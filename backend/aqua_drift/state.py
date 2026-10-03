from __future__ import annotations

import asyncio
from collections import OrderedDict, deque

from aqua_drift.deployment import default_position
from aqua_drift.models import (
    CpaResult,
    CurrentEstimate,
    DopplerBatch,
    EstimatorFeed,
    EstimatorOutput,
    EstimatorSettings,
    ObserverFix,
    ObserverPlacement,
    ObserverRecord,
    ObserverState,
    Position,
    ScenarioConfig,
    SimState,
    Snapshot,
    TargetState,
    TrackEstimate,
)

BATCH_RETENTION_TICKS = 900


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
        self.batches: deque[DopplerBatch] = deque(maxlen=BATCH_RETENTION_TICKS)
        self.estimates: list[TrackEstimate] = []
        self.cpa: list[CpaResult] = []
        self.current_estimate: CurrentEstimate | None = None
        self.placements: deque[ObserverPlacement] = deque()
        self.assignments: dict[str, Position] = {}
        self.assignment_counter = 0
        self.generation = 0

    async def set_config(self, config: ScenarioConfig) -> None:
        async with self.lock:
            self.config = config
            while len(self.observers) > config.observer_limit:
                evicted_id, _ = self.observers.popitem(last=False)
                self._archive(evicted_id)

    async def set_tick(self, tick: int) -> None:
        async with self.lock:
            self.tick = max(self.tick, tick)

    async def set_target(self, target: TargetState) -> None:
        async with self.lock:
            if self.target is None or target.tick >= self.target.tick:
                self.target = target

    # ---------------------------------------------------------------- observers
    async def queue_placement(self, placement: ObserverPlacement) -> int:
        async with self.lock:
            self.placements.append(placement)
            return len(self.placements)

    async def assign_position(self, observer_id: str) -> Position:
        """Initial position for an observer container: an explicitly queued placement first,
        otherwise the configured default pattern. Stable for a given observer id."""
        async with self.lock:
            if observer_id in self.assignments:
                return self.assignments[observer_id]
            if self.placements:
                position = self.placements.popleft().position
            else:
                position = default_position(self.config, self.assignment_counter)
                self.assignment_counter += 1
            self.assignments[observer_id] = position
            return position

    async def set_observer(self, observer: ObserverState) -> str | None:
        """Register/update an observer. When a new observer exceeds the limit (1..100) the
        oldest is evicted; each observer may observe for at most max_observation_seconds.
        Evicted / expired observers are archived and their history is kept."""
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

    async def observer_record(self, observer_id: str) -> ObserverRecord:
        async with self.lock:
            return self.observers[observer_id].model_copy()

    # ---------------------------------------------------------------- observations
    async def add_batch(self, batch: DopplerBatch) -> None:
        async with self.lock:
            if self.batches and batch.tick <= self.batches[-1].tick:
                return
            active = set(self.observers)
            batch.observations = [o for o in batch.observations if o.observer_id in active]
            batch.truth = [t for t in batch.truth if t.observer_id in active]
            self.batches.append(batch)

    async def estimator_feed(self, after_tick: int) -> EstimatorFeed:
        async with self.lock:
            batches = [
                DopplerBatch(tick=b.tick, observations=b.observations)  # truth stripped
                for b in self.batches
                if b.tick > after_tick
            ]
            return EstimatorFeed(
                tick=self.tick,
                generation=self.generation,
                settings=EstimatorSettings.from_config(self.config),
                observers=[
                    ObserverFix(
                        observer_id=r.state.observer_id, tick=r.state.tick, position=r.state.position
                    )
                    for r in self.observers.values()
                ],
                archived_observer_ids=list(self.archived_observer_ids),
                batches=batches,
            )

    async def set_estimator_output(self, output: EstimatorOutput) -> None:
        async with self.lock:
            self.estimates = output.estimates
            self.cpa = output.cpa
            self.current_estimate = output.current

    async def snapshot(self) -> Snapshot:
        async with self.lock:
            return Snapshot(
                tick=self.tick,
                generation=self.generation,
                config=self.config,
                target=self.target,
                observers=list(self.observers.values()),
                doppler=self.batches[-1] if self.batches else None,
                estimates=list(self.estimates),
                cpa=list(self.cpa),
                current_estimate=self.current_estimate,
                archived_observer_ids=list(self.archived_observer_ids),
            )

    async def sim_state(self) -> SimState:
        async with self.lock:
            return SimState(
                tick=self.tick,
                generation=self.generation,
                config=self.config,
                target=self.target,
                observers=list(self.observers.values()),
            )

    async def reset_runtime(self) -> None:
        async with self.lock:
            self.generation += 1
            self.target = None
            self.batches.clear()
            self.estimates = []
            self.cpa = []
            self.current_estimate = None

    def _archive(self, observer_id: str) -> None:
        if observer_id not in self._archived_set:
            self._archived_set.add(observer_id)
            self.archived_observer_ids.append(observer_id)
