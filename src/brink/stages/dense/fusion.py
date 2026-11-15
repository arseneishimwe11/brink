"""Fuse per-view depth maps into one cloud, keeping only points that agree across views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from brink.geometry import PointCloud


@dataclass
class DepthView:
    """A depth map with the pinhole camera it belongs to.

    ``rotation`` maps world to camera; ``depth`` is distance along the optical
    axis, 0 where unknown.
    """

    depth: NDArray[np.float32]
    color: NDArray[np.uint8]  # (H, W, 3) RGB
    K: NDArray[np.float64]
    rotation: NDArray[np.float64]
    center: NDArray[np.float64]
    group: int = 0  # views from the same source frame share a group


def _backproject(view: DepthView, stride: int) -> tuple[NDArray, NDArray, NDArray]:
    h, w = view.depth.shape
    v, u = np.mgrid[stride // 2 : h : stride, stride // 2 : w : stride]
    z = view.depth[v, u]
    ok = z > 0
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok].astype(np.float64)
    x = (u - view.K[0, 2]) / view.K[0, 0] * z
    y = (v - view.K[1, 2]) / view.K[1, 1] * z
    cam = np.stack([x, y, z], axis=1)
    world = cam @ view.rotation + view.center
    return world, z, view.color[v.astype(int), u.astype(int)]


def _agreement(points: NDArray, other: DepthView, tolerance: float) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Which points ``other`` also measured, and the relative depth difference it saw."""
    cam = (points - other.center) @ other.rotation.T
    z = cam[:, 2]
    h, w = other.depth.shape
    safe = np.where(z > 1e-6, z, 1.0)
    u = np.round(other.K[0, 0] * cam[:, 0] / safe + other.K[0, 2]).astype(np.int64)
    v = np.round(other.K[1, 1] * cam[:, 1] / safe + other.K[1, 2]).astype(np.int64)
    inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    agree = np.zeros(len(points), dtype=bool)
    offset = np.zeros(len(points))
    d = other.depth[v[inside], u[inside]]
    rel = (d - z[inside]) / z[inside]
    good = (d > 0) & (np.abs(rel) <= tolerance)
    agree[inside] = good
    offset[inside] = np.where(good, rel, 0.0)
    return agree, offset


def fuse(
    views: list[DepthView],
    cameras: NDArray,
    neighbours: int = 6,
    min_agree: int = 2,
    tolerance: float = 0.012,
    stride: int = 2,
) -> PointCloud:
    """Keep each depth sample only if enough other views measured the same place, and average them."""
    centers = np.array([v.center for v in views])
    xyz, rgb, view_dir = [], [], []
    for i, view in enumerate(views):
        points, _, colour = _backproject(view, stride)
        if len(points) == 0:
            continue
        other = [j for j in np.argsort(np.linalg.norm(centers - centers[i], axis=1)) if views[j].group != view.group]
        votes = np.zeros(len(points), dtype=np.int32)
        shift = np.zeros(len(points))
        for j in other[:neighbours]:
            agree, offset = _agreement(points, views[j], tolerance)
            votes += agree
            shift += offset
        keep = votes >= min_agree
        if not keep.any():
            continue
        # Move each point to the average of all views that measured it. The
        # views look from nearly the same side, so a relative depth difference
        # in a neighbour is close to the same difference along this view's ray.
        stretch = 1.0 + shift[keep] / (votes[keep] + 1)
        p = view.center + (points[keep] - view.center) * stretch[:, None]
        to_cam = view.center - p
        xyz.append(p)
        rgb.append(colour[keep])
        view_dir.append(to_cam / np.linalg.norm(to_cam, axis=1, keepdims=True))
    if not xyz:
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), cameras=cameras)
    return PointCloud(np.concatenate(xyz), np.concatenate(rgb), cameras=cameras, view=np.concatenate(view_dir))
