"""Estimated presence region (推定存在圏) at a user-set probability.

The region is the highest-density region (HDR) of the particle posterior: the smallest set of
3-D voxels whose probability mass reaches p %. It is NOT forced into an ellipse / ellipsoid,
so curved, elongated and split (ambiguous) regions are shown as they are. Each connected part
is reported with its mass, its voxels and a horizontal outline extruded over its depth span.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from aqua_drift.estimation.frame import FT_TO_M, YD_TO_M, LocalFrame
from aqua_drift.models import Position, PresenceRegion, RegionComponent

MAX_VOXELS_PER_COMPONENT = 1500


def presence_region(
    frame: LocalFrame,
    points: np.ndarray,
    weights: np.ndarray,
    probability_pct: float,
) -> PresenceRegion:
    p = probability_pct / 100.0
    n = len(points)
    if n == 0:
        return PresenceRegion(probability_pct=probability_pct)
    lo = np.percentile(points, 0.2, axis=0)
    hi = np.percentile(points, 99.8, axis=0)
    span = np.maximum(hi - lo, np.array([30.0, 30.0, 6.0]))
    lo -= 0.05 * span
    hi += 0.05 * span
    per_axis = int(np.clip(round((n / 6.0) ** (1.0 / 3.0)), 6, 28))
    horizontal_cell = max(span[0], span[1]) * 1.1 / per_axis
    horizontal_cell = max(horizontal_cell, 20.0)
    vertical_cell = max(span[2] * 1.1 / max(per_axis // 2, 3), 3.0)
    edges = [
        np.arange(lo[0], hi[0] + horizontal_cell, horizontal_cell),
        np.arange(lo[1], hi[1] + horizontal_cell, horizontal_cell),
        np.arange(max(lo[2], 0.0), hi[2] + vertical_cell, vertical_cell),
    ]
    edges = [e if len(e) >= 2 else np.array([e[0], e[0] + 1.0]) for e in edges]
    mass, _ = np.histogramdd(points, bins=edges, weights=weights)
    total = mass.sum()
    if total <= 0:
        return PresenceRegion(probability_pct=probability_pct)
    mass /= total
    flat = mass.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    count = int(np.searchsorted(cumulative, p) + 1)
    selected = np.zeros(flat.shape, dtype=bool)
    selected[order[:count]] = True
    selected = selected.reshape(mass.shape)
    labels, components = ndimage.label(selected, structure=np.ones((3, 3, 3)))
    results: list[RegionComponent] = []
    centers = [(e[:-1] + e[1:]) / 2.0 for e in edges]
    for label in range(1, components + 1):
        idx = np.argwhere(labels == label)
        cell_mass = mass[labels == label]
        component_mass = float(cell_mass.sum())
        xyz = np.stack(
            [centers[0][idx[:, 0]], centers[1][idx[:, 1]], centers[2][idx[:, 2]]], axis=1
        )
        centroid = (cell_mass[:, None] * xyz).sum(axis=0) / max(component_mass, 1e-12)
        # horizontal outline from the voxel corners
        half = horizontal_cell / 2.0
        corners = np.concatenate(
            [xyz[:, :2] + np.array([sx * half, sy * half]) for sx in (-1, 1) for sy in (-1, 1)]
        )
        corners = np.unique(np.round(corners, 1), axis=0)
        try:
            hull = ConvexHull(corners)
            outline = corners[hull.vertices]
        except (QhullError, ValueError):
            outline = corners
        lon, lat = frame.to_lonlat_array(outline[:, 0], outline[:, 1])
        polygon = [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)]
        order_v = np.argsort(cell_mass)[::-1][:MAX_VOXELS_PER_COMPONENT]
        vlon, vlat = frame.to_lonlat_array(xyz[order_v, 0], xyz[order_v, 1])
        voxels = [
            [float(a), float(b), float(c / FT_TO_M)]
            for a, b, c in zip(vlon, vlat, xyz[order_v, 2], strict=True)
        ]
        centroid_geo = frame.to_geo(*centroid)
        results.append(
            RegionComponent(
                probability_mass_pct=component_mass * 100.0,
                centroid=Position(
                    latitude=centroid_geo.latitude,
                    longitude=centroid_geo.longitude,
                    depth_ft=centroid_geo.depth_ft,
                ),
                polygon=polygon,
                min_depth_ft=float(max(xyz[:, 2].min() - vertical_cell / 2, 0.0) / FT_TO_M),
                max_depth_ft=float((xyz[:, 2].max() + vertical_cell / 2) / FT_TO_M),
                voxels=voxels,
                voxel_size_yd=horizontal_cell / YD_TO_M,
                voxel_height_ft=vertical_cell / FT_TO_M,
            )
        )
    results.sort(key=lambda item: item.probability_mass_pct, reverse=True)
    return PresenceRegion(
        probability_pct=probability_pct,
        components=results,
        disconnected=len(results) > 1,
    )
