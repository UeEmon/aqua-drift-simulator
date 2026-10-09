"""In-process scenario runner.

Uses exactly the same physics functions as the target / observer / doppler containers, but
steps them in one process. Used by tests, by the observation-mode comparison study and for
quick offline what-if runs:

    python -m aqua_drift.scenario --seconds 3600 --observers 5
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

from aqua_drift.deployment import default_position
from aqua_drift.estimation.engine import TrackingEngine
from aqua_drift.forward_deployment import plan_forward_deployment_scheduled
from aqua_drift.layer import advance as advance_layer
from aqua_drift.layer import initial_state as initial_layer_state
from aqua_drift.layer import ready_pose
from aqua_drift.models import (
    DopplerBatch,
    DropTask,
    EstimatorOutput,
    EstimatorSettings,
    LayerFeed,
    LayerState,
    ObserverState,
    PlanBasis,
    Position,
    ReplanFeed,
    ScenarioConfig,
    TargetState,
)
from aqua_drift.optimal_deployment import LayerAvailability, SensorModel
from aqua_drift.physics import (
    LevelNoise,
    SourceSignal,
    advance_observer,
    advance_target,
    doppler_observation,
    initial_target,
    local_offset_m,
)
from aqua_drift.replanning import basis_for, plan_replacement

YD_TO_M = 0.9144


@dataclass
class ScenarioRun:
    config: ScenarioConfig
    observer_count: int
    observer_positions: list[Position] | None = None
    target: TargetState | None = None
    observers: list[ObserverState] = field(default_factory=list)
    engine: TrackingEngine | None = None
    truth_track: list[TargetState] = field(default_factory=list)
    tick: int = 0
    rng: random.Random | None = None
    level_noise: LevelNoise | None = None
    signal: SourceSignal | None = None  # emitted history (propagation delay, fluctuation)
    forward: bool = False  # automatic forward (前程) deployment from the estimate
    deploy_check_s: int = 10
    deployments: list[tuple[int, int]] = field(default_factory=list)  # (tick, count)
    last_deploy_tick: int | None = None
    last_replan_tick: int | None = None
    replans: list[tuple[int, list[int]]] = field(default_factory=list)  # (tick, replaced drops)
    tasks: list[DropTask] = field(default_factory=list)  # drops flown by the layer (設標者)
    layer_state: LayerState | None = None
    layer_rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.target = initial_target(self.config)
        positions = self.observer_positions or [
            default_position(self.config, index) for index in range(self.observer_count)
        ]
        self.observers = [
            ObserverState(observer_id=f"obs-{index:03d}", tick=0, position=position)
            for index, position in enumerate(positions)
        ]
        self.engine = TrackingEngine(EstimatorSettings.from_config(self.config))
        self.rng = random.Random(self.config.bearing.random_seed)
        self.level_noise = LevelNoise(self.config.lloyd.random_seed)
        self.signal = SourceSignal(self.config.source.random_seed)
        self.signal.add(self.config, self.target)

    def step(self) -> DopplerBatch:
        self.tick += 1
        self.target = advance_target(self.config, self.target, 1.0)
        self.target.tick = self.tick
        self.observers = [advance_observer(self.config, item, 1.0) for item in self.observers]
        for item in self.observers:
            item.tick = self.tick
        self.signal.add(self.config, self.target)
        observations, truth = [], []
        for observer in self.observers:
            obs, tr = doppler_observation(
                self.config, self.target, observer, self.rng, self.level_noise, self.signal
            )
            observations.append(obs)
            truth.append(tr)
        batch = DopplerBatch(tick=self.tick, observations=observations, truth=truth)
        self.truth_track.append(self.target)
        self.engine.process(batch)
        if self.forward and self.tick % self.deploy_check_s == 0:
            self._forward_deploy()
        if self.config.layer.enabled and (self.tasks or self.layer_state is not None or self.forward):
            self._layer_step()
        return batch

    def _layer_step(self) -> None:
        """The layer (設標者) flies to the approved drops (automatic approval) and lays them."""
        engine = self.engine
        datum = self.config.target.initial_position
        if engine.pf.initialized and engine.frame is not None:  # estimate only (no truth)
            datum = engine.frame.to_geo(*engine.pf.mean_state()[0:3])
        base = engine.current_fit.base
        approved = [t for t in self.tasks if t.status == "APPROVED"]
        feed = LayerFeed(
            tick=self.tick, config=self.config.layer, tasks=approved, datum=datum,
            current_east_kt=float(base[0]) / 0.5144444444444445,
            current_north_kt=float(base[1]) / 0.5144444444444445,
        )
        if self.layer_rng is None:
            self.layer_rng = random.Random(self.config.layer.random_seed)
        if self.layer_state is None:
            self.layer_state = initial_layer_state(self.config.layer, datum, self.tick - 1, self.layer_rng)
        self.layer_state, update = advance_layer(feed, self.layer_state, self.layer_rng)
        for task in approved:
            if task.task_id in update.completed:
                task.status = "DONE"
                task.done_tick = self.tick
                task.position = update.completed[task.task_id]
                self._add_observer(task.position)
            elif task.task_id in update.task_positions:
                task.position = update.task_positions[task.task_id]
                task.eta_s = update.task_eta_s.get(task.task_id)

    def _layer_availability(self) -> LayerAvailability | None:
        if not self.config.layer.enabled:
            return None
        layer = self.config.layer
        queue = sorted((t for t in self.tasks if t.status == "APPROVED"), key=lambda t: t.flight_key())
        ready, where, heading = ready_pose(self.layer_state, layer, queue, self.tick,
                                           self.config.target.initial_position)
        state = self.layer_state
        return LayerAvailability(
            ready_s=float(ready - self.tick), position=where, speed_kt=layer.speed_kt,
            max_bank_deg=layer.max_bank_deg, heading_deg=heading,
            now_position=state.position if state is not None else None,
            now_heading_deg=state.heading_deg if state is not None else None,
            queue=[(t.position, None if t.planned_tick is None else float(t.planned_tick - self.tick)) for t in queue],
        )

    def _add_observer(self, position: Position) -> None:
        self.observers.append(
            ObserverState(observer_id=f"fwd-{len(self.deployments):02d}-{len(self.observers):03d}",
                          tick=self.tick, position=position)
        )
        while len(self.observers) > self.config.observer_limit:
            self.observers.pop(0)  # oldest first; its history stays in the engine

    def _forward_deploy(self) -> None:
        output = self.engine.output()
        estimate = next((e for e in output.estimates if e.mode.value == "ONLINE"), None)
        sensor = SensorModel(
            source_frequency_hz=self.config.source.source_frequency_hz + self.config.source.shared_recognition_bias_hz,
            sound_speed_mps=self.config.source.sound_speed_mps,
            frequency_sigma_hz=self.config.estimator.model_frequency_sigma_hz,
            use_bearing=self.config.estimator.use_bearing and self.config.bearing.enabled,
            bearing_sigma_deg=self.config.estimator.bearing_sigma_deg,
            bearing_interval_s=self.config.bearing.interval_s,
            gate_sigma_yd=self.config.estimator.range_gate_softness_yd,
        )
        open_tasks = [t for t in self.tasks if t.status == "APPROVED"]
        free_slots = max(self.config.observer_limit - len(self.observers) - len(open_tasks), 0)
        if self.config.layer.enabled:  # first: do the open drops still match the estimate?
            replan = ReplanFeed(tasks=open_tasks, layer_state=self.layer_state, layer=self.config.layer,
                                datum=self.config.target.initial_position, last_replan_tick=self.last_replan_tick)
            decision, _ = plan_replacement(
                self.tick, estimate, [o.position for o in self.observers], [], replan, self.config.forward,
                self.config.max_slant_range_yd, free_slots, sensor, self.config.estimator.max_target_depth_ft,
            )
            if decision is not None:
                for task in open_tasks:
                    if task.task_id in decision.replaces:
                        task.status = "REPLACED"
                self._queue(decision.positions, decision.planned_ticks, decision.bases, decision.revision)
                self.replans.append((self.tick, decision.replaces))
                self.last_replan_tick = self.tick
                return
        positions, _, planned = plan_forward_deployment_scheduled(
            self.tick,
            estimate,
            [o.position for o in self.observers],
            [t.position for t in open_tasks],
            self.config.forward,
            self.config.max_slant_range_yd,
            self.last_deploy_tick,
            self.config.deployment.depth_step_ft,
            free_slots=free_slots,
            source_frequency_hz=sensor.source_frequency_hz,
            sound_speed_mps=sensor.sound_speed_mps,
            frequency_sigma_hz=sensor.frequency_sigma_hz,
            layer=self._layer_availability(),
            sensor=sensor,
            max_depth_ft=self.config.estimator.max_target_depth_ft,
        )
        if positions:
            self._queue(positions, planned, [basis_for(estimate, p) for p in positions], 0)

    def _queue(self, positions: list[Position], planned: list[int] | None, bases: list[PlanBasis],
               revision: int) -> None:
        for index, position in enumerate(positions):
            if self.config.layer.enabled:  # laid when the layer gets there (automatic approval)
                self.tasks.append(DropTask(
                    task_id=len(self.tasks) + 1, created_tick=self.tick, source="forward",
                    position=position, status="APPROVED", approved_tick=self.tick,
                    planned_tick=planned[index] if planned else None, basis=bases[index], revision=revision,
                ))
            else:
                self._add_observer(position)
        self.deployments.append((self.tick, len(positions)))
        self.last_deploy_tick = self.tick

    def run(self, seconds: int, output_every: int = 0) -> EstimatorOutput:
        for _ in range(seconds):
            self.step()
            if output_every and self.tick % output_every == 0:
                self.report(self.engine.output())
        return self.engine.output()

    def error(self, output: EstimatorOutput) -> dict[str, float]:
        estimate = output.estimates[0]
        if estimate.current_position is None:
            return {}
        east, north, down = local_offset_m(estimate.current_position, self.target.position)
        return {
            "horizontal_error_yd": math.hypot(east, north) / YD_TO_M,
            "depth_error_ft": down / 0.3048,
            "ground_speed_error_kt": estimate.ground_speed_kt - self.target.ground_speed_kt,
            "water_speed_error_kt": estimate.through_water_speed_kt
            - self.target.through_water_speed_kt,
            "cog_error_deg": (estimate.cog_deg - self.target.cog_deg + 180) % 360 - 180,
            "hdg_error_deg": (estimate.hdg_deg - self.target.hdg_deg + 180) % 360 - 180,
        }

    def report(self, output: EstimatorOutput) -> None:
        estimate = output.estimates[0]
        err = self.error(output)
        if not err:
            print(f"t={self.tick:5d}s {estimate.observability_status}")
            return
        u = estimate.uncertainty
        print(
            f"t={self.tick:5d}s {estimate.observability_status:22s} "
            f"posErr={err['horizontal_error_yd']:7.0f}YD (1σmaj {u.horizontal_major_yd:6.0f}) "
            f"depthErr={err['depth_error_ft']:6.0f}Ft (σ{u.depth_sigma_ft:5.0f}) "
            f"SOG {estimate.ground_speed_kt:5.1f}/{self.target.ground_speed_kt:5.1f}kt "
            f"COG {estimate.cog_deg:5.1f}/{self.target.cog_deg:5.1f} "
            f"STW {estimate.through_water_speed_kt:5.1f}/{self.target.through_water_speed_kt:5.1f} "
            f"HDG {estimate.hdg_deg:5.1f}/{self.target.hdg_deg:5.1f} "
            f"regions={len(estimate.presence_region.components)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AQUA-DRIFT scenario in-process")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--observers", type=int, default=5)
    parser.add_argument("--report-every", type=int, default=60)
    parser.add_argument("--bias-hz", type=float, default=0.0)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--no-bearing", action="store_true", help="estimator ignores bearings")
    parser.add_argument("--forward", action="store_true", help="automatic forward deployment")
    args = parser.parse_args()
    config = ScenarioConfig()
    config.source.shared_recognition_bias_hz = args.bias_hz
    if args.no_bearing:
        config.estimator.use_bearing = False
    if args.particles:
        config.estimator.particle_count = args.particles
    run = ScenarioRun(config, args.observers, forward=args.forward)
    output = run.run(args.seconds, args.report_every)
    for cpa in output.cpa:
        print(
            f"CPA {cpa.observer_id} pass{cpa.pass_index} t={cpa.cpa_tick:7.1f}±{cpa.cpa_tick_sigma_s:4.1f}s "
            f"R={cpa.cpa_slant_range_yd:6.0f}±{cpa.cpa_slant_range_sigma_yd:5.0f}YD "
            f"Vrel={cpa.relative_speed_kt:5.2f}kt final={cpa.final}"
        )


if __name__ == "__main__":
    main()
