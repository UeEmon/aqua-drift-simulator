"""Generate a protocol-v2 message stream from an in-process scenario, with checkpoints of the
expected (server-side) compact tracks. Used to test the JavaScript decoder against the Python
encoder:  python -m aqua_drift.analysis.wire_fixture fixture.json [seconds]"""
from __future__ import annotations

import json
import sys

from aqua_drift.models import (
    DeploymentStatus,
    EstimationControl,
    ObserverRecord,
    ScenarioConfig,
    Snapshot,
)
from aqua_drift.scenario import ScenarioRun
from aqua_drift.wire import WireEncoder, compact_region, compact_track


def build_stream(seconds: int = 600, window_s: int = 120) -> dict:
    config = ScenarioConfig()
    config.estimator.particle_count = 1500
    config.smoothing_window_seconds = window_s  # small window -> frequent tail rewrites
    config.estimator.track_store_slots = 60
    config.lloyd.enabled = True  # exercises the optional Lloyd's mirror field of the stream
    run = ScenarioRun(config, 4, forward=True)  # includes the layer (設標者) and drop tasks
    encoder = WireEncoder(region_interval_s=3.0)
    messages, checkpoints, full_sizes = [], [], []
    for _ in range(seconds):
        batch = run.step()
        output = run.engine.output()
        snapshot = Snapshot(
            tick=run.tick,
            estimation=EstimationControl(running=True, run_id=1, started_tick=0),
            config=config,
            target=run.target,
            observers=[ObserverRecord(state=o, registered_tick=0, last_tick=run.tick) for o in run.observers],
            doppler=batch,
            estimates=output.estimates,
            cpa=output.cpa,
            current_estimate=output.current,
            archived_observer_ids=[],
            lloyd=output.lloyd,
            deployment=DeploymentStatus(
                approval=config.layer.approval, tasks=[t.model_copy() for t in run.tasks[-30:]],
                layer=run.layer_state.model_copy() if run.layer_state else None,
            ),
        )
        message = encoder.encode(snapshot, float(run.tick))
        messages.append(json.dumps(message, separators=(",", ":")))
        full_sizes.append(len(snapshot.model_dump_json()))
        if run.tick % 30 == 0 or run.tick == seconds:
            data = snapshot.model_dump(mode="json")
            checkpoints.append({
                "index": len(messages) - 1,
                "tracks": {e["mode"]: compact_track(e["track"]) for e in data["estimates"]},
                "region": compact_region(data["estimates"][0]["presence_region"]) if data["estimates"] else None,
                "observers": len(data["observers"]),
                "tick": run.tick,
            })
    return {"messages": messages, "checkpoints": checkpoints, "full_sizes": full_sizes}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "wire-fixture.json"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(build_stream(seconds), handle)
    print("written", out)
