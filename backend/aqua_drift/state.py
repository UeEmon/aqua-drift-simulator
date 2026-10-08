from __future__ import annotations

import asyncio
import math
from collections import OrderedDict, deque

import numpy as np

from aqua_drift.deployment import default_position
from aqua_drift.models import (
    BearingReport,
    ClockStatus,
    CpaResult,
    CurrentEstimate,
    DeploymentFeed,
    DeploymentRecord,
    DeploymentRequest,
    DeploymentStatus,
    DopplerBatch,
    DropTask,
    EstimationControl,
    EstimatorFeed,
    EstimatorOutput,
    EstimatorSettings,
    LayerFeed,
    LayerState,
    LayerUpdate,
    LloydDepthEstimate,
    ObserverAssignment,
    ObserverFix,
    ObserverPlacement,
    ObserverRecord,
    ObserverState,
    OrchestratorFeed,
    Position,
    ScenarioConfig,
    SimState,
    Snapshot,
    TargetState,
    TrackEstimate,
)
from aqua_drift.optimal_deployment import LayerAvailability
from aqua_drift.physics import local_offset_m

BATCH_RETENTION_TICKS = 900
# above 1x the clock waits for the estimator when it falls this many ticks behind the newest batch
ESTIMATOR_SLACK_TICKS = 10


class ObserverRejected(RuntimeError):
    """Raised when an observer can no longer publish active observations."""


class SimulationState:
    def __init__(
        self, config: ScenarioConfig | None = None, autostart_estimation: bool = True
    ) -> None:
        self.lock = asyncio.Lock()
        self.config = config or ScenarioConfig()
        self.tick = 0
        self.time_scale = 1.0
        self.estimator_tick = -1
        self.target: TargetState | None = None
        self.observers: OrderedDict[str, ObserverRecord] = OrderedDict()
        self.archived_observer_ids: list[str] = []  # "obs-03#1" = slot obs-03, session 1
        self._archived_set: set[str] = set()
        self.sessions: dict[str, int] = {}  # observer slot -> current session
        self.batches: deque[DopplerBatch] = deque(maxlen=BATCH_RETENTION_TICKS)
        self.estimates: list[TrackEstimate] = []
        self.cpa: list[CpaResult] = []
        self.current_estimate: CurrentEstimate | None = None
        self.lloyd: LloydDepthEstimate | None = None
        self.placements: deque[ObserverPlacement] = deque()
        self.assignments: dict[str, Position] = {}
        self.assignment_counter = 0
        self.explicit_ids: set[str] = set()
        self.generation = 0
        self.estimation = EstimationControl(running=autostart_estimation)
        self.standby: dict[str, int] = {}  # observer_id -> tick of last assignment poll
        self.deploy_history: deque[DeploymentRecord] = deque(maxlen=20)
        self.last_deploy_tick: int | None = None
        self.bearings: dict[str, BearingReport] = {}
        self.tasks: list[DropTask] = []  # additional observers laid by the layer (設標者)
        self.task_counter = 0
        self.layer_state: LayerState | None = None

    async def set_config(self, config: ScenarioConfig) -> None:
        async with self.lock:
            self.config = config
            if config.layer.approval == "auto":
                for task in self.tasks:
                    if task.status == "PROPOSED":
                        self._approve(task)
            while len(self.observers) > config.observer_limit:
                evicted_id, _ = self.observers.popitem(last=False)
                self._archive(evicted_id)

    async def set_tick(self, tick: int) -> None:
        async with self.lock:
            self.tick = max(self.tick, tick)

    async def set_time_scale(self, time_scale: float) -> None:
        async with self.lock:
            self.time_scale = time_scale

    async def clock_status(self) -> ClockStatus:
        async with self.lock:
            return ClockStatus(tick=self.tick, time_scale=self.time_scale, synced=self._synced())

    def _synced(self) -> bool:
        """True when the target, every observer and the Doppler engine have published the current
        tick and the estimator is not falling behind, so the next tick can start without any
        container skipping an epoch."""
        tick = self.tick
        if self.target is None or self.target.tick < tick:
            return False
        if any(r.state.tick < tick for r in self.observers.values()):
            return False
        last_batch = self.batches[-1].tick if self.batches else -1
        if self.observers and tick % self.config.doppler_interval_seconds == 0 and last_batch < tick:
            return False
        estimator_behind = self.estimator_tick < last_batch - ESTIMATOR_SLACK_TICKS
        return not (
            self.estimation.running and last_batch > self.estimation.started_tick and estimator_behind
        )

    async def set_target(self, target: TargetState) -> None:
        async with self.lock:
            if self.target is None or target.tick >= self.target.tick:
                self.target = target

    # ---------------------------------------------------------------- observers
    async def queue_placement(self, placement: ObserverPlacement) -> int:
        """Operator placement. With the layer enabled the layer flies there and lays the
        observer (the operator chose the point, so no further approval); a placement for a
        named observer container, or without the layer, is queued at once."""
        async with self.lock:
            if self.config.layer.enabled and placement.observer_id is None:
                task = self._new_task(placement.position, "manual", "operator placement")
                task.planned_tick = task.requested_tick = placement.planned_tick
                return len(self.placements) + self._open_task_count()
            self.placements.append(placement)
            return len(self.placements)

    async def assign_position(self, observer_id: str) -> ObserverAssignment | None:
        """Start position for an observer container, stable per observer id and session:
        1. a queued placement (operator or forward deployment),
        2. the default pattern for the first `deployment.initial_count` observers,
        3. otherwise None: the container waits as a standby observer until a placement
           is queued (used by the forward deployment).
        A slot whose current session ended (evicted / 3 h limit) is reused with a new session;
        the history of the old session stays archived as "<slot>#<session>"."""
        async with self.lock:
            session = self.sessions.setdefault(observer_id, 0)
            if self._session_key(observer_id, session) in self._archived_set:
                session += 1  # reuse the slot
                self.sessions[observer_id] = session
                self.assignments.pop(observer_id, None)
                self.explicit_ids.discard(observer_id)
            if observer_id in self.assignments:
                return self._assignment(observer_id, self.assignments[observer_id])
            if self.placements:
                placement = self.placements.popleft()
                position = placement.position
                if placement.source == "manual":
                    self.explicit_ids.add(observer_id)
            elif self.assignment_counter < self.config.deployment.initial_count:
                position = default_position(self.config, self.assignment_counter)
                self.assignment_counter += 1
            else:
                self.standby[observer_id] = self.tick
                return None
            self.standby.pop(observer_id, None)
            self.assignments[observer_id] = position
            return self._assignment(observer_id, position)

    def _assignment(self, observer_id: str, position: Position) -> ObserverAssignment:
        return ObserverAssignment(
            **position.model_dump(), observer_id=observer_id, session=self.sessions[observer_id]
        )

    @staticmethod
    def _session_key(observer_id: str, session: int) -> str:
        return f"{observer_id}#{session}"

    async def evict_oldest(self, count: int = 1) -> list[str]:
        """Free observer slots: archive the oldest active observers (history kept)."""
        async with self.lock:
            evicted = []
            for _ in range(min(count, len(self.observers))):
                observer_id, _record = self.observers.popitem(last=False)
                self._archive(observer_id)
                evicted.append(observer_id)
            return evicted

    async def orchestrator_feed(self) -> OrchestratorFeed:
        async with self.lock:
            return OrchestratorFeed(
                tick=self.tick,
                limit=self.config.observer_limit,
                active_ids=list(self.observers),
                standby_ids=[k for k, last in self.standby.items() if self.tick - last <= 10],
                pending_placements=len(self.placements),
                initial_remaining=max(
                    0, self.config.deployment.initial_count - self.assignment_counter
                ),
            )

    def _standby_count(self) -> int:
        return sum(1 for last in self.standby.values() if self.tick - last <= 10)

    # ---------------------------------------------------------------- forward deployment
    async def queue_deployment(self, request: DeploymentRequest, source: str = "forward") -> DeploymentRecord:
        """A deployment plan. With the layer enabled every point becomes a drop task that is
        proposed to the operator (approved at once in automatic approval mode); the observer is
        in the water when the layer reaches the point. Without the layer: queued at once."""
        async with self.lock:
            planned = request.planned_ticks or [None] * len(request.positions)
            for position, planned_tick in zip(request.positions, planned, strict=False):
                if self.config.layer.enabled:
                    task = self._new_task(position, source, request.reason)
                    task.planned_tick = task.requested_tick = planned_tick
                else:
                    self.placements.append(ObserverPlacement(position=position, source="forward"))
            record = DeploymentRecord(
                tick=request.tick, positions=request.positions, reason=request.reason
            )
            self.deploy_history.append(record)
            self.last_deploy_tick = request.tick
            return record

    async def deployment_feed(self) -> DeploymentFeed:
        async with self.lock:
            return DeploymentFeed(
                tick=self.tick,
                config=self.config.forward,
                max_slant_range_yd=self.config.max_slant_range_yd,
                depth_step_ft=self.config.deployment.depth_step_ft,
                estimates=list(self.estimates),
                observer_positions=[r.state.position for r in self.observers.values()],
                pending_positions=[p.position for p in self.placements]
                + [t.position for t in self.tasks if t.status in self.OPEN_TASK_STATES],
                standby_count=self._standby_count(),
                last_deploy_tick=self.last_deploy_tick,
                free_slots=max(
                    self.config.observer_limit - len(self.observers) - len(self.placements)
                    - self._open_task_count(), 0
                ),
                source_frequency_hz=self.config.source.source_frequency_hz
                + self.config.source.shared_recognition_bias_hz,
                sound_speed_mps=self.config.source.sound_speed_mps,
                frequency_sigma_hz=self.config.estimator.model_frequency_sigma_hz,
                **self._layer_availability(),
            )

    def _layer_availability(self) -> dict:
        """When / where the layer is free for a new drop: after its open tasks."""
        layer = self.config.layer
        if not layer.enabled:
            return {"layer_enabled": False}
        ready_tick, ready_position, heading = self.tick, None, None
        if self.layer_state is not None:
            ready_position, heading = self.layer_state.position, self.layer_state.heading_deg
        for task in self.tasks:
            if task.status not in self.OPEN_TASK_STATES:
                continue
            due = task.planned_tick if task.planned_tick is not None else self.tick + int(task.eta_s or 0)
            if due >= ready_tick:
                ready_tick, ready_position, heading = due, task.position, None
        return {
            "layer_enabled": True,
            "layer_ready_tick": ready_tick,
            "layer_ready_position": ready_position or self.config.target.initial_position,
            "layer_ready_heading_deg": heading,
            "layer_speed_kt": layer.speed_kt,
            "layer_max_bank_deg": layer.max_bank_deg,
        }

    def _deployment_status(self) -> DeploymentStatus:
        self._expire_tasks()
        return DeploymentStatus(
            standby_count=self._standby_count(),
            pending_placements=len(self.placements),
            last_deploy_tick=self.last_deploy_tick,
            history=list(self.deploy_history),
            approval=self.config.layer.approval,
            tasks=self.tasks[-30:],
            layer=self.layer_state if self.config.layer.enabled else None,
        )

    # ---------------------------------------------------------------- layer (設標者)
    OPEN_TASK_STATES = ("PROPOSED", "APPROVED")

    def _open_task_count(self) -> int:
        return sum(1 for t in self.tasks if t.status in self.OPEN_TASK_STATES)

    def _new_task(self, position: Position, source: str, reason: str) -> DropTask:
        self.task_counter += 1
        task = DropTask(task_id=self.task_counter, created_tick=self.tick, source=source,
                        reason=reason, position=position, sequence=self.task_counter)
        if source == "manual" or self.config.layer.approval == "auto":
            self._approve(task)
        self.tasks.append(task)
        if len(self.tasks) > 200:
            closed = [t for t in self.tasks if t.status not in self.OPEN_TASK_STATES]
            for old in closed[: len(self.tasks) - 200]:
                self.tasks.remove(old)
        return task

    def _approve(self, task: DropTask) -> None:
        task.status = "APPROVED"
        task.approved_tick = self.tick

    def _expire_tasks(self) -> None:
        timeout = self.config.layer.proposal_timeout_s
        for task in self.tasks:
            if task.status == "PROPOSED" and self.tick - task.created_tick > timeout:
                task.status = "EXPIRED"

    async def decide_tasks(self, task_ids: list[int] | None, approve: bool) -> list[DropTask]:
        """Operator decision on proposed drops (None = all proposed)."""
        async with self.lock:
            self._expire_tasks()
            changed = []
            for task in self.tasks:
                if task.status != "PROPOSED" or (task_ids is not None and task.task_id not in task_ids):
                    continue
                if approve:
                    self._approve(task)
                else:
                    task.status = "REJECTED"
                changed.append(task)
            return changed

    async def cancel_tasks(self, task_ids: list[int] | None) -> list[DropTask]:
        """Operator cancels proposed or approved drops (None = all open drops)."""
        async with self.lock:
            changed = []
            for task in self.tasks:
                if task.status in self.OPEN_TASK_STATES and (task_ids is None or task.task_id in task_ids):
                    task.status = "CANCELLED"
                    changed.append(task)
            return changed

    async def reschedule_task(self, task_id: int, planned_tick: int | None) -> DropTask | None:
        """Operator sets the drop time of an open drop (None = as soon as possible)."""
        async with self.lock:
            for task in self.tasks:
                if task.task_id == task_id and task.status in self.OPEN_TASK_STATES:
                    task.planned_tick = task.requested_tick = planned_tick
                    return task
            return None

    async def reorder_tasks(self, task_ids: list[int]) -> list[DropTask] | None:
        """Operator sets the drop order (設標順) of the open drops: the listed ones first in
        that order, the others after them in their current order. The drop times are planned
        again along the new order: each drop at its requested (optimal) time, or when the layer
        can be there after the previous drop (flight at the layer speed from the previous point,
        with the turn), whichever is later. Drops 'as soon as possible' stay so while only such
        drops come before them. Returns the open drops in the new order (None: unknown id)."""
        async with self.lock:
            self._expire_tasks()
            open_tasks = sorted((t for t in self.tasks if t.status in self.OPEN_TASK_STATES),
                                key=lambda t: t.flight_key())
            by_id = {t.task_id: t for t in open_tasks}
            if len(set(task_ids)) != len(task_ids) or any(i not in by_id for i in task_ids):
                return None
            order = [by_id[i] for i in task_ids] + [t for t in open_tasks if t.task_id not in task_ids]
            sequences = sorted(t.sequence or t.task_id for t in order)
            layer = self.config.layer
            if self.layer_state is not None:
                previous, heading = self.layer_state.position, self.layer_state.heading_deg
            else:
                previous, heading = self.config.target.initial_position, None
            ready, timed = float(self.tick), False
            for task, sequence in zip(order, sequences, strict=True):
                task.sequence = sequence
                if layer.enabled:
                    east, north, _ = local_offset_m(previous, task.position)
                    flight = LayerAvailability(ready_s=0.0, position=previous, speed_kt=layer.speed_kt,
                                               max_bank_deg=layer.max_bank_deg, heading_deg=heading)
                    ready += float(flight.earliest_s(previous, np.array([[east, north, 0.0]]))[0])
                timed = timed or task.requested_tick is not None
                if timed:
                    task.planned_tick = max(task.requested_tick or 0, math.ceil(ready))
                    ready = float(task.planned_tick)
                else:
                    task.planned_tick = None
                previous, heading = task.position, None
            return order

    async def layer_feed(self) -> LayerFeed:
        async with self.lock:
            self._expire_tasks()
            online = next((e for e in self.estimates if e.mode.value == "ONLINE"), None)
            datum = online.current_position if online and online.current_position else None
            if datum is None:
                datum = self.config.target.initial_position  # operator's datum until an estimate exists
            current = self.current_estimate.base_velocity if self.current_estimate else None
            approved = sorted((t for t in self.tasks if t.status == "APPROVED"), key=lambda t: t.flight_key())
            return LayerFeed(
                tick=self.tick,
                generation=self.generation,
                config=self.config.layer,
                tasks=approved,
                datum=datum,
                current_east_kt=current.east_kt if current else 0.0,
                current_north_kt=current.north_kt if current else 0.0,
                state=self.layer_state,
            )

    async def set_layer_update(self, update: LayerUpdate) -> list[int]:
        """Layer position every tick; drop points drift; reached points become observers."""
        async with self.lock:
            self.layer_state = update.state
            done = []
            by_id = {t.task_id: t for t in self.tasks}
            for task_id, position in update.task_positions.items():
                task = by_id.get(task_id)
                if task and task.status == "APPROVED":
                    task.position = position
                    task.eta_s = update.task_eta_s.get(task_id)
            for task_id, position in update.completed.items():
                task = by_id.get(task_id)
                if not task or task.status != "APPROVED":
                    continue
                task.status = "DONE"
                task.done_tick = update.state.tick
                task.position = position
                task.eta_s = 0.0
                self.placements.append(ObserverPlacement(
                    position=position, source="manual" if task.source == "manual" else "forward"
                ))
                done.append(task_id)
            return done

    async def set_observer(self, observer: ObserverState) -> str | None:
        """Register/update an observer. When a new observer exceeds the limit (1..99) the
        oldest is evicted; each observer may observe for at most max_observation_seconds.
        Evicted / expired observers are archived and their history is kept; posts from an
        ended session (a container that should have stopped) are rejected."""
        async with self.lock:
            observer_id = observer.observer_id
            current = self.sessions.setdefault(observer_id, observer.session)
            if self._session_key(observer_id, observer.session) in self._archived_set:
                raise ObserverRejected(f"observer {observer_id} session {observer.session} ended")
            if observer.session < current:
                raise ObserverRejected(f"observer {observer_id} session {observer.session} is stale")
            self.sessions[observer_id] = observer.session

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
            for obs in batch.observations:
                if obs.bearing_deg is not None:
                    self.bearings[obs.observer_id] = BearingReport(
                        observer_id=obs.observer_id,
                        tick=obs.tick,
                        bearing_deg=obs.bearing_deg,
                        observer_position=obs.observer_position,
                    )

    async def estimator_feed(self, after_tick: int) -> EstimatorFeed:
        async with self.lock:
            batches = [
                DopplerBatch(tick=b.tick, observations=b.observations)  # truth stripped
                for b in self.batches
                if b.tick > after_tick
            ]
            return EstimatorFeed(
                tick=self.tick,
                time_scale=self.time_scale,
                generation=self.generation,
                estimation=self.estimation,
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

    # ---------------------------------------------------------------- estimation control
    async def start_estimation(self) -> EstimationControl:
        """Start a new estimation run from the current tick (previous output cleared)."""
        async with self.lock:
            self.estimation = EstimationControl(
                running=True, run_id=self.estimation.run_id + 1, started_tick=self.tick
            )
            self.estimates, self.cpa, self.current_estimate = [], [], None
            self.lloyd = None
            return self.estimation

    async def stop_estimation(self) -> EstimationControl:
        """Stop the estimator; the last estimate stays displayed (frozen)."""
        async with self.lock:
            if self.estimation.running:
                self.estimation = self.estimation.model_copy(
                    update={"running": False, "stopped_tick": self.tick}
                )
            return self.estimation

    async def set_estimator_output(self, output: EstimatorOutput) -> None:
        async with self.lock:
            self.estimator_tick = max(self.estimator_tick, output.tick)
            if not self.estimation.running:
                return  # late output from a stopped run
            self.estimates = output.estimates
            self.cpa = output.cpa
            self.current_estimate = output.current
            self.lloyd = output.lloyd

    async def snapshot(self) -> Snapshot:
        async with self.lock:
            return Snapshot(
                tick=self.tick,
                generation=self.generation,
                time_scale=self.time_scale,
                deployment=self._deployment_status(),
                estimation=self.estimation,
                bearings=[
                    b for b in self.bearings.values()
                    if self.tick - b.tick <= self.config.bearing.interval_s
                ],
                config=self.config,
                target=self.target,
                observers=list(self.observers.values()),
                doppler=self.batches[-1] if self.batches else None,
                estimates=list(self.estimates),
                cpa=list(self.cpa),
                current_estimate=self.current_estimate,
                archived_observer_ids=list(self.archived_observer_ids),
                lloyd=self.lloyd,
            )

    async def sim_state(self) -> SimState:
        async with self.lock:
            return SimState(
                tick=self.tick,
                generation=self.generation,
                time_scale=self.time_scale,
                config=self.config,
                target=self.target,
                observers=list(self.observers.values()),
            )

    async def reset_runtime(self, replace_observers: bool = True) -> None:
        """Restart the scenario from the configured initial target state. Observers return
        to their start points; default-placed observers are re-placed around the (possibly
        new) initial target position. Event history in the database is retained."""
        async with self.lock:
            self.generation += 1
            self.target = None
            self.batches.clear()
            self.bearings.clear()
            # automatic (forward) placements belong to the old run; operator placements stay
            self.placements = deque(p for p in self.placements if p.source == "manual")
            self.deploy_history.clear()
            self.last_deploy_tick = None
            # automatic drop plans belong to the old run; operator placements are still flown
            self.tasks = [t for t in self.tasks if t.source == "manual" and t.status == "APPROVED"]
            self.layer_state = None
            self.standby.clear()
            if replace_observers:
                explicit = {k: v for k, v in self.assignments.items() if k in self.explicit_ids}
                self.assignments = explicit
                self.assignment_counter = 0
            self.observers.clear()  # observers re-register (3 h limit restarts)
            if self.estimation.running:
                self.estimation = EstimationControl(
                    running=True, run_id=self.estimation.run_id + 1, started_tick=self.tick
                )
            self.estimates = []
            self.cpa = []
            self.current_estimate = None
            self.lloyd = None

    def _archive(self, observer_id: str) -> None:
        key = self._session_key(observer_id, self.sessions.get(observer_id, 0))
        if key not in self._archived_set:
            self._archived_set.add(key)
            self.archived_observer_ids.append(key)
