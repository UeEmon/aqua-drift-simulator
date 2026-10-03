"""In-process scenario runner.

Uses exactly the same physics functions as the target / observer / doppler containers, but
steps them in one process. Used by tests, by the observation-mode comparison study and for
quick offline what-if runs:

    python -m aqua_drift.scenario --seconds 3600 --observers 5
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from aqua_drift.deployment import default_position
from aqua_drift.estimation.engine import TrackingEngine
from aqua_drift.models import (
    DopplerBatch,
    EstimatorOutput,
    EstimatorSettings,
    ObserverState,
    Position,
    ScenarioConfig,
    TargetState,
)
from aqua_drift.physics import (
    advance_observer,
    advance_target,
    doppler_observation,
    initial_target,
    local_offset_m,
)

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

    def step(self) -> DopplerBatch:
        self.tick += 1
        self.target = advance_target(self.config, self.target, 1.0)
        self.target.tick = self.tick
        self.observers = [advance_observer(self.config, item, 1.0) for item in self.observers]
        for item in self.observers:
            item.tick = self.tick
        observations, truth = [], []
        for observer in self.observers:
            obs, tr = doppler_observation(self.config, self.target, observer)
            observations.append(obs)
            truth.append(tr)
        batch = DopplerBatch(tick=self.tick, observations=observations, truth=truth)
        self.truth_track.append(self.target)
        self.engine.process(batch)
        return batch

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
    args = parser.parse_args()
    config = ScenarioConfig()
    config.source.shared_recognition_bias_hz = args.bias_hz
    if args.particles:
        config.estimator.particle_count = args.particles
    run = ScenarioRun(config, args.observers)
    output = run.run(args.seconds, args.report_every)
    for cpa in output.cpa:
        print(
            f"CPA {cpa.observer_id} pass{cpa.pass_index} t={cpa.cpa_tick:7.1f}±{cpa.cpa_tick_sigma_s:4.1f}s "
            f"R={cpa.cpa_slant_range_yd:6.0f}±{cpa.cpa_slant_range_sigma_yd:5.0f}YD "
            f"Vrel={cpa.relative_speed_kt:5.2f}kt final={cpa.final}"
        )


if __name__ == "__main__":
    main()
