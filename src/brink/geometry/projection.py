"""Project points from the analysis frame into the solved video frames."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from brink.stages.sfm.base import SfMResult


def project_points(
    points: NDArray, sfm: SfMResult, index: int, rotation: NDArray, scale: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Pixel coordinates (u, v) and depth of analysis-frame ``points`` in frame ``index``.

    Analysis coordinates are ``scale * rotation @ sfm coordinates``.
    """
    world = (np.asarray(points, dtype=np.float64) / scale) @ rotation
    cam = (world - sfm.positions[index]) @ sfm.rotations[index]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    k1, k2 = sfm.distortion[0], sfm.distortion[1]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return sfm.fx * x * d + sfm.cx, sfm.fy * y * d + sfm.cy, z


def camera_centres(sfm: SfMResult, rotation: NDArray, scale: float) -> NDArray:
    """Camera positions in the analysis frame."""
    return scale * sfm.positions @ rotation.T
