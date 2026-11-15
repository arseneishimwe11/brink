"""Orientation and scale: which way is up, and how big is a model unit.

Angles only need "up". Lengths also need scale. Each estimate carries its
source and an honest uncertainty, because both feed straight into the
measurements shown to people.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from brink.domain import GravityEstimate, ScaleEstimate
from brink.geometry import PointCloud
from brink.stages.sfm.base import SfMResult

ASSUMED_PHONE_HEIGHT_M = 1.35  # "chest height" in the capture guidance
PHONE_HEIGHT_REL_UNCERTAINTY = 0.12
SENSOR_BIAS_DEG = 0.7  # accelerometer mounting and calibration error that does not average out


def _unit(v: NDArray) -> NDArray:
    return v / np.linalg.norm(v)


def _angle_deg(a: NDArray, b: NDArray) -> float:
    return float(np.degrees(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0))))


# --------------------------------------------------------------------------
# Gravity
# --------------------------------------------------------------------------


def load_sensor_gravity(sidecar: str | Path) -> tuple[NDArray, NDArray] | None:
    """Read (times, down vectors in camera axes) from a Brink sensor sidecar."""
    path = Path(sidecar)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("format") != "brink-sensors-1":
        return None
    samples = data["samples"]
    return np.array([s["t"] for s in samples]), np.array([s["gravity"] for s in samples])


def gravity_from_sensors(sfm: SfMResult, frame_times: NDArray, sensor: tuple[NDArray, NDArray]) -> GravityEstimate | None:
    """Average the phone's gravity readings after rotating each into the model frame."""
    times, down_cam = sensor
    nearest = np.clip(np.searchsorted(times, frame_times), 1, len(times) - 1)
    nearest = np.where(np.abs(times[nearest - 1] - frame_times) < np.abs(times[nearest] - frame_times), nearest - 1, nearest)
    up_world = -np.einsum("nij,nj->ni", sfm.rotations, down_cam[nearest])
    up_world /= np.linalg.norm(up_world, axis=1, keepdims=True)
    mean = _unit(np.median(up_world, axis=0))
    dev = np.degrees(np.arccos(np.clip(up_world @ mean, -1, 1)))
    keep = dev <= max(3 * np.median(dev), 1.0)
    if keep.sum() < 3:
        return None
    mean = _unit(up_world[keep].mean(0))
    spread = float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(up_world[keep] @ mean, -1, 1))) ** 2)))
    uncertainty = float(np.hypot(spread / np.sqrt(keep.sum()), SENSOR_BIAS_DEG))
    return GravityEstimate(up=tuple(mean), source="sensor", uncertainty_deg=uncertainty)


def _floor_normal(cloud: PointCloud, up_guess: NDArray) -> tuple[NDArray, float] | None:
    """Normal of the largest roughly level plane, and the share of points on it."""
    if len(cloud) < 500:
        return None
    step = max(1, len(cloud) // 60000)
    pts = cloud.xyz[::step].astype(np.float64)
    extent = np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0))
    remaining = pts
    best = None
    for _ in range(4):
        if len(remaining) < 300:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining)
        plane, inliers = pcd.segment_plane(distance_threshold=0.006 * extent, ransac_n=3, num_iterations=400)
        normal = np.asarray(plane[:3])
        normal = normal if normal @ up_guess > 0 else -normal
        if _angle_deg(normal, up_guess) < 35 and (best is None or len(inliers) > best[1]):
            best = (_unit(normal), len(inliers))
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
    if best is None or best[1] < 0.04 * len(pts):
        return None
    return best[0], best[1] / len(pts)


def gravity_from_scene(sfm: SfMResult, cloud: PointCloud) -> GravityEstimate:
    """Guess "up" without sensors.

    Two weak clues are combined: people hold a phone roughly level (so the
    image's sideways axis is horizontal), and the floor they walk on is roughly
    level. Neither is exact, hence the wide uncertainty.
    """
    right = sfm.rotations[:, :, 0]
    up_guess = _unit(-sfm.rotations[:, :, 1].mean(0))
    scatter = right.T @ right / len(right)
    floor = _floor_normal(cloud, up_guess)
    if floor is not None:
        normal, share = floor
        weight = 0.35 + 0.65 * min(1.0, share / 0.3)
        scatter = scatter + weight * (np.eye(3) - np.outer(normal, normal))
        prior = normal
        uncertainty = 3.0
    else:
        scatter = scatter + 0.15 * (np.eye(3) - np.outer(up_guess, up_guess))
        prior = up_guess
        uncertainty = 6.0
    values, vectors = np.linalg.eigh(scatter)
    up = vectors[:, 0]
    up = up if up @ prior > 0 else -up
    return GravityEstimate(up=tuple(_unit(up)), source="scene", uncertainty_deg=uncertainty)


def rotation_to_z_up(up: NDArray) -> NDArray:
    """Rotation matrix taking ``up`` to +Z with the least possible turn."""
    up = _unit(np.asarray(up, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), up @ z
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * k + (1 - c) * k @ k


# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------


def scale_from_camera_height(cloud_up: PointCloud) -> ScaleEstimate | None:
    """Rough scale from the capture guidance "hold the phone at chest height".

    ``cloud_up`` must already have +Z up. The floor under the walking line is
    found from points close to the camera track.
    """
    cams = cloud_up.cameras.astype(np.float64)
    if len(cams) < 3 or len(cloud_up) < 500:
        return None
    step = max(1, len(cloud_up) // 200000)
    pts = cloud_up.xyz[::step].astype(np.float64)
    below = pts[pts[:, 2] < np.median(cams[:, 2])]
    if len(below) < 200:
        return None
    # Distance in plan from each low point to the nearest camera.
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(cams[:, :2]).query(below[:, :2])
    reach = np.percentile(dist, 25)
    near = below[dist <= max(reach, 1e-9)]
    if len(near) < 100:
        return None
    floor_z = float(np.percentile(near[:, 2], 20))
    height = float(np.median(cams[:, 2]) - floor_z)
    if height <= 0:
        return None
    return ScaleEstimate(
        factor=ASSUMED_PHONE_HEIGHT_M / height, source="camera_height",
        relative_uncertainty=PHONE_HEIGHT_REL_UNCERTAINTY,
    )
