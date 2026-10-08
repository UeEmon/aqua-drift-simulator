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
