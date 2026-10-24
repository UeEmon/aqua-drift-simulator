import json

from aqua_drift.analysis.wire_fixture import build_stream
from aqua_drift.wire import track_diff


def apply(rows: list, diff: dict | None) -> list:
    if diff is None:
        return rows
    if diff.get("reset"):
        return [tuple(p) for p in diff["pts"]]
    return rows[: diff["keep"]] + [tuple(p) for p in diff["pts"]]


def test_track_diff_append_tail_and_reset() -> None:
    a = [(1, 0, 0, 0, 0, 0, 0, 0), (2, 1, 1, 1, 1, 1, 1, 1)]
    assert track_diff(a, a) is None
    appended = [*a, (3, 2, 2, 2, 2, 2, 2, 2)]
    assert track_diff(a, appended) == {"keep": 2, "pts": [[3, 2, 2, 2, 2, 2, 2, 2]]}
    tail = [a[0], (2, 9, 9, 9, 9, 9, 9, 9), (3, 2, 2, 2, 2, 2, 2, 2)]
    assert apply(a, track_diff(a, tail)) == tail
    thinned = [(2, 1, 1, 1, 1, 1, 1, 1), (3, 2, 2, 2, 2, 2, 2, 2)]
    assert track_diff(a, thinned)["reset"] is True
    assert track_diff(appended, a)["reset"] is True  # shortened


def test_stream_reconstructs_tracks_and_is_small() -> None:
    stream = build_stream(240, window_s=60)
    tracks: dict[str, list] = {}
    region = None
    checkpoints = {c["index"]: c for c in stream["checkpoints"]}
    for index, raw in enumerate(stream["messages"]):
        message = json.loads(raw)
        for body in message["est"]:
            if "trk" in body:
                tracks[body["mode"]] = apply(tracks.get(body["mode"], []), body["trk"])
        if "rgn" in message:
            region = message["rgn"]
        if index in checkpoints:
            expected = checkpoints[index]
            for mode, rows in expected["tracks"].items():
                assert tracks.get(mode, []) == [tuple(r) for r in rows], f"{mode} at {index}"
            if expected["region"] is not None and region is not None:
                assert region["pct"] == expected["region"]["pct"]
    later = stream["messages"][120:]
    full = stream["full_sizes"][120:]
    mean_delta = sum(map(len, later)) / len(later)
    mean_full = sum(full) / len(full)
    assert mean_delta < 0.35 * mean_full, (mean_delta, mean_full)


def test_layer_planned_path_sent_only_when_changed() -> None:
    from aqua_drift.models import DeploymentStatus, LayerState, Position, ScenarioConfig, Snapshot
    from aqua_drift.wire import WireEncoder

    def snapshot(tick: int, path: list[tuple[float, float]]) -> Snapshot:
        layer = LayerState(tick=tick, position=Position(latitude=35.0, longitude=140.0 + tick * 1e-4, depth_ft=0),
                           heading_deg=90.0, speed_kt=150.0, planned_path=path)
        return Snapshot(tick=tick, deployment=DeploymentStatus(layer=layer), config=ScenarioConfig(), target=None,
                        observers=[], doppler=None, estimates=[], cpa=[], current_estimate=None,
                        archived_observer_ids=[])

    encoder = WireEncoder()
    path = [(35.0 + k * 1e-3, 140.0 + k * 1.234567891e-3) for k in range(150)]
    first = encoder.encode(snapshot(1, path), 1.0)
    assert first["lpp"][1] == [35.001, 140.001235]  # rounded to 1e-6 deg
    assert "planned_path" not in first["lay"]
    second = encoder.encode(snapshot(2, path), 2.0)
    assert "lpp" not in second and second["lay"]["tick"] == 2  # the layer itself moves every message
    assert second["lay"]["position"]["longitude"] == 140.0002  # floats rounded to 1e-6
    # flying along the same plan: the path's start moves every second, resent every 5 s
    assert "lpp" not in encoder.encode(snapshot(3, path[1:]), 3.0)
    assert len(encoder.encode(snapshot(6, path[2:]), 6.0)["lpp"]) == 148
    # where the path leads changed (a drop added): sent at once
    assert len(encoder.encode(snapshot(7, path[2:] + [(35.5, 140.5)]), 7.0)["lpp"]) == 149
    assert encoder.encode(snapshot(8, []), 8.0)["lpp"] == []


def test_predicted_regions_sent_only_when_changed() -> None:
    from aqua_drift.models import (
        EstimateMode,
        Position,
        PredictedRegion,
        PresenceRegion,
        RegionComponent,
        ScenarioConfig,
        Snapshot,
        TrackEstimate,
    )
    from aqua_drift.wire import WireEncoder

    def snapshot(tick: int, lon: float) -> Snapshot:
        component = RegionComponent(
            probability_mass_pct=90.0, centroid=Position(latitude=35.0, longitude=lon, depth_ft=500),
            polygon=[[lon, 35.0], [lon + 0.01, 35.0], [lon, 35.01]], min_depth_ft=450, max_depth_ft=550,
            voxels=[],
        )
        predicted = [PredictedRegion(horizon_s=600, tick=tick + 600,
                                     region=PresenceRegion(probability_pct=90.0, components=[component]))]
        estimate = TrackEstimate(mode=EstimateMode.ONLINE, tick=tick, observability_status="TRACKING",
                                 presence_region=PresenceRegion(probability_pct=90.0), predicted_regions=predicted)
        return Snapshot(tick=tick, config=ScenarioConfig(), target=None, observers=[], doppler=None,
                        estimates=[estimate], cpa=[], current_estimate=None, archived_observer_ids=[])

    encoder = WireEncoder(region_interval_s=3.0)
    first = encoder.encode(snapshot(1, 140.0), now=0.0)
    assert first["prg"][0]["h"] == 600 and first["prg"][0]["c"][0]["poly"][0] == [140.0, 35.0]
    assert "predicted_regions" not in first["est"][0]
    assert "prg" not in encoder.encode(snapshot(2, 140.0), now=1.0)  # unchanged
    assert "prg" not in encoder.encode(snapshot(3, 140.1), now=2.0)  # changed, but not yet due
    assert encoder.encode(snapshot(4, 140.1), now=3.5)["prg"][0]["c"][0]["poly"][0] == [140.1, 35.0]
