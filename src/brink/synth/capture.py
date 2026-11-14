"""Simulated handheld walks: where the phone is and where it points.

Camera convention is OpenCV's: +X right, +Y down, +Z forward. ``rotations``
hold camera-to-world matrices, so their columns are the camera axes in world
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from brink.synth.scene import SceneSpec


class CaptureSpec(BaseModel):
    """How one simulated video was filmed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    path: str = "along"  # "along": walk parallel to the face; "arc": walk around it
    standoff: float = 4.0  # distance from the toe, metres
    camera_height: float = 1.35
    coverage: float = 1.1  # along: fraction of face length walked; arc: fraction of 180 degrees
    duration_s: float = 60.0
    fps: float = 5.0
    aim_height: float = 0.45  # fraction of face height the camera points at
    shake_deg: float = 1.0  # RMS hand tremor
    wander: float = 0.15  # RMS drift of the walking line, metres
    speed_variation: float = 0.25
    width: int = 960
    height: int = 540
    hfov_deg: float = 66.0
    lighting: str = "overcast"  # "overcast" | "sun" | "low"
    blur: float = 0.3  # 0 none, 1 heavy
    noise: float = 0.01  # sensor noise, fraction of full scale

    @property
    def focal_px(self) -> float:
        return 0.5 * self.width / np.tan(np.radians(self.hfov_deg) / 2)


@dataclass
class CameraPath:
    times: NDArray[np.float64]
    positions: NDArray[np.float64]  # (N, 3)
    rotations: NDArray[np.float64]  # (N, 3, 3) camera-to-world

    def __len__(self) -> int:
        return len(self.times)


def _smooth_signal(t: NDArray, rng: np.random.Generator, periods: tuple[float, ...], rms: float) -> NDArray:
    """Zero-mean wobble built from a few sinusoids, scaled to the given RMS."""
    if rms == 0:
        return np.zeros_like(t)
    total = np.zeros_like(t)
    for period in periods:
        total += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    return total * rms / (np.sqrt(np.mean(total**2)) + 1e-9)


def look_at(position: NDArray, target: NDArray) -> NDArray:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def _rotvec(v: NDArray) -> NDArray:
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def camera_path(scene: SceneSpec, capture: CaptureSpec) -> CameraPath:
    rng = np.random.default_rng(capture.seed)
    n = max(2, int(round(capture.duration_s * capture.fps)))
    t = np.arange(n) / capture.fps

    # Progress along the route with uneven walking speed.
    speed = 1 + _smooth_signal(t, rng, (7.0, 13.0, 23.0), capture.speed_variation)
    speed = np.clip(speed, 0.2, None)
    u = np.cumsum(speed)
    u = (u - u[0]) / (u[-1] - u[0])  # 0..1

    mid_y = scene.crest_y / 2
    drift = _smooth_signal(t, rng, (9.0, 17.0, 31.0), capture.wander)
    bob = 0.02 * np.sin(2 * np.pi * 1.8 * t + rng.uniform(0, 6.28))  # footsteps
    bob += _smooth_signal(t, rng, (11.0, 19.0), 0.03)

    if capture.path == "arc":
        half = np.radians(90.0 * capture.coverage)
        ang = -half + 2 * half * u
        radius = capture.standoff + mid_y + drift
        x = radius * np.sin(ang)
        y = mid_y - radius * np.cos(ang)
        tx = 0.3 * scene.length * np.sin(ang) * 0.5
        ty = np.full(n, mid_y)
    else:
        half = 0.5 * capture.coverage * scene.length
        x = -half + 2 * half * u
        y = -capture.standoff + drift
        lead = _smooth_signal(t, rng, (15.0, 27.0), 0.08 * scene.length)
        tx = np.clip(x + lead, -scene.length / 2, scene.length / 2)
        ty = np.full(n, mid_y)
    if rng.random() < 0.5:  # walk either way
        x, tx = -x, -tx
    z = capture.camera_height + bob
    tz = capture.aim_height * scene.height + _smooth_signal(t, rng, (8.0, 21.0), 0.06 * scene.height)

    positions = np.stack([x, y, z], axis=1)
    targets = np.stack([tx, ty, tz], axis=1)
    shake = np.radians(capture.shake_deg)
    jitter = np.stack(
        [_smooth_signal(t, rng, (0.7, 1.9, 4.3), shake) for _ in range(3)], axis=1
    )
    rotations = np.empty((n, 3, 3))
    for i in range(n):
        rotations[i] = look_at(positions[i], targets[i]) @ _rotvec(jitter[i])
    return CameraPath(t, positions, rotations)
