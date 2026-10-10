"""Compact delta encoding of the GIS stream (WebSocket /ws, protocol v2).

The first message of a connection carries everything; later messages carry only changes:

* tracks: per estimate mode, the first point that differs from what this connection already
  has (`from` tick) and the points after it, as compact arrays; a full replacement (`reset`)
  when the beginning changed (server-side thinning, new run);
* presence region: shared by both estimate modes, sent when it changed, at most every
  `region_interval_s` (unless the run / probability changed);
* config, CPA, current estimate, deployment status, archive list, Lloyd's mirror depth, the
  layer's planned flight path: only when changed;
* observers, Doppler truth, bearings, relative kinematics: compact arrays.

Numbers are rounded to display precision (lat/lon 1e-6 deg ~ 0.1 m).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from aqua_drift.models import Snapshot

PROTOCOL_VERSION = 2

TrackRow = tuple[int, float, float, float, float, float, float, float]


def _r(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _rounded(value: Any, digits: int = 6) -> Any:
    """Floats rounded to `digits` decimals (lat/lon 1e-6 deg ~ 0.1 m) throughout a payload."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha1(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compact_track(points: list[dict[str, Any]]) -> list[TrackRow]:
    return [
        (
            int(p["tick"]),
            round(p["latitude"], 6),
            round(p["longitude"], 6),
            round(p["depth_ft"], 1),
            round(p["horizontal_sigma_yd"], 1),
            round(p["depth_sigma_ft"], 1),
            round(p["ground_speed_kt"], 2),
            round(p["cog_deg"], 1),
        )
        for p in points
    ]


def compact_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "pct": region["probability_pct"],
        "disc": region["disconnected"],
        "c": [
            {
                "m": round(c["probability_mass_pct"], 2),
                "cen": [
                    round(c["centroid"]["latitude"], 6),
                    round(c["centroid"]["longitude"], 6),
                    round(c["centroid"]["depth_ft"], 1),
                ],
                "poly": [[round(x, 6), round(y, 6)] for x, y in c["polygon"]],
                "zmin": round(c["min_depth_ft"], 1),
                "zmax": round(c["max_depth_ft"], 1),
                "vox": [[round(v[0], 6), round(v[1], 6), round(v[2], 1)] for v in c["voxels"]],
                "vs": round(c["voxel_size_yd"], 2),
                "vh": round(c["voxel_height_ft"], 2),
            }
            for c in region["components"]
        ],
    }


def track_diff(sent: list[TrackRow], new: list[TrackRow]) -> dict[str, Any] | None:
    """None if unchanged; {"reset": True, "pts": all} when the beginning changed or points were
    removed at the end; otherwise {"keep": n, "pts": tail}: keep the first n points the client
    already has and append `tail`."""
    if sent == new:
        return None
    limit = min(len(sent), len(new))
    index = 0
    while index < limit and sent[index] == new[index]:
        index += 1
    if index == 0 or index == len(new):
        return {"reset": True, "pts": [list(p) for p in new]}
    return {"keep": index, "pts": [list(p) for p in new[index:]]}


class WireEncoder:
    """Per-connection encoder: remembers what this client already has."""

    def __init__(self, region_interval_s: float = 3.0, path_interval_s: float = 5.0) -> None:
        self.region_interval_s = region_interval_s
        self.path_interval_s = path_interval_s
        self.path_key: tuple[Any, ...] | None = None
        self.path_sent_at = -1e9
        self.tracks: dict[str, list[TrackRow]] = {}
        self.hashes: dict[str, str] = {}
        self.region_hash: str | None = None
        self.region_key: tuple[Any, ...] | None = None
        self.region_sent_at = -1e9
        self.run_key: tuple[Any, ...] | None = None

    def _changed(self, name: str, value: Any) -> bool:
        digest = _digest(value)
        if self.hashes.get(name) == digest:
            return False
        self.hashes[name] = digest
        return True

    def encode(self, snapshot: Snapshot, now: float) -> dict[str, Any]:
        data = snapshot.model_dump(mode="json")
        run_key = (data["generation"], data["estimation"]["run_id"])
        if run_key != self.run_key:
            self.run_key = run_key
            self.tracks.clear()
            self.region_hash = None
        message: dict[str, Any] = {
            "v": PROTOCOL_VERSION,
            "t": data["tick"],
            "gen": data["generation"],
            "clk": data["time_scale"],
            "ep": data["epoch_s"],
            "ctl": data["estimation"],
            "target": data["target"],
            "obs": [
                [
                    r["state"]["observer_id"],
                    round(r["state"]["position"]["latitude"], 6),
                    round(r["state"]["position"]["longitude"], 6),
                    round(r["state"]["position"]["depth_ft"], 1),
                    r["registered_tick"],
                    r["last_tick"],
                ]
                for r in data["observers"]
            ],
            "brg": [
                [
                    b["observer_id"],
                    b["tick"],
                    round(b["bearing_deg"], 2),
                    round(b["observer_position"]["latitude"], 6),
                    round(b["observer_position"]["longitude"], 6),
                    round(b["observer_position"]["depth_ft"], 1),
                ]
                for b in data["bearings"]
            ],
        }
        doppler = data["doppler"]
        message["dop"] = None if doppler is None else {
            "t": doppler["tick"],
            "det": [o["observer_id"] for o in doppler["observations"] if o["detected"]],
            "tr": [
                [
                    t["observer_id"],
                    round(t["relative_speed_kt"], 2),
                    round(t["slant_range_yd"], 1),
                    round(t["true_bearing_deg"], 2),
                    t["tick"],
                ]
                for t in doppler["truth"]
            ],
        }
        # the layer (設標者) moves every second: its state travels in every message, apart from
        # the deployment status (tasks, history), which is sent only when it changed
        layer = data["deployment"].pop("layer", None)
        path = None
        if layer is not None:  # the guidance' own path bookkeeping is not shown
            for key in ("approach_key", "path_sides", "path_lengths_m"):
                layer.pop(key, None)
            # the planned flight path (up to a few hundred points) changes only when the plan
            # does: sent separately, only when changed, instead of in every message
            path = [[round(lat, 6), round(lon, 6)] for lat, lon in layer.pop("planned_path", [])]
        message["lay"] = layer
        # the path starts at the layer, so it changes every second: resend it at most every
        # `path_interval_s` (the client starts it at the layer's current position), at once when
        # where it leads changes (new / reordered / finished drops, path cleared)
        key = (None,) if not path else (len(path) > 1, tuple(path[-1]))
        due = now - self.path_sent_at >= self.path_interval_s
        if (key != self.path_key or due) and self._changed("lpp", path):
            message["lpp"] = path
            self.path_key = key
            self.path_sent_at = now
        for task in data["deployment"].get("tasks", []):  # the replanner's bookkeeping is not shown
            task.pop("basis", None)
            for key in ("cancel_suggestion", "cancel_suggested_tick"):  # only sent when proposed
                if task.get(key) is None:
                    task.pop(key, None)
        for name, key in (
            ("cfg", "config"),
            ("cpa", "cpa"),
            ("cur", "current_estimate"),
            ("dep", "deployment"),
            ("arch", "archived_observer_ids"),
            ("lly", "lloyd"),
        ):
            if self._changed(name, data[key]):
                message[name] = data[key]

        estimates = []
        region = None
        for estimate in data["estimates"]:
            mode = estimate["mode"]
            region = region or estimate["presence_region"]
            body = {k: v for k, v in estimate.items() if k not in ("track", "presence_region", "relative")}
            body["rel"] = [
                [r["observer_id"], round(r["relative_speed_kt"], 2), round(r["slant_range_yd"], 1),
                 r["detected"]]
                for r in estimate["relative"]
            ]
            new_track = compact_track(estimate["track"])
            diff = track_diff(self.tracks.get(mode, []), new_track)
            self.tracks[mode] = new_track
            if diff is not None:
                body["trk"] = diff
            estimates.append(body)
        message["est"] = estimates
        if not data["estimates"]:
            self.tracks.clear()

        if region is not None:
            compact = compact_region(region)
            digest = _digest(compact)
            key = (run_key, region["probability_pct"])
            due = now - self.region_sent_at >= self.region_interval_s
            if digest != self.region_hash and (due or key != self.region_key or self.region_hash is None):
                message["rgn"] = compact
                self.region_hash = digest
                self.region_key = key
                self.region_sent_at = now
        elif self.region_hash is not None:
            message["rgn"] = None  # region cleared (no estimate)
            self.region_hash = None
        return _rounded(message)
