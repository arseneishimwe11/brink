"""Shared helpers for hazards that need the video frames as well as the 3D shape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.geometry.projection import camera_centres, project_points
from brink.stages.sfm.base import SfMResult


@dataclass
class FrameSet:
    """A spread of video frames with their cameras, ready for sampling."""

    sfm: SfMResult
    rotation: NDArray
    scale: float
    index: list[int]
    gray: list[NDArray[np.float32]]  # 0..255
    centres: NDArray  # camera centres in the analysis frame, one per kept frame

    @classmethod
    def load(cls, sfm: SfMResult, frames_dir: Path, rotation: NDArray, scale: float, count: int = 14) -> FrameSet:
        pick = np.unique(np.linspace(0, sfm.registered - 1, min(count, sfm.registered)).astype(int)).tolist()
        gray = []
        for i in pick:
            image = cv2.imread(str(Path(frames_dir) / sfm.names[i]), cv2.IMREAD_GRAYSCALE)
            gray.append(image.astype(np.float32))
        return cls(sfm, rotation, scale, pick, gray, camera_centres(sfm, rotation, scale)[pick])

    def sample(self, k: int, image: NDArray, points: NDArray, margin: int = 6) -> tuple[NDArray, NDArray]:
        """Values of ``image`` (same size as frame ``k``) at the projections of ``points``.

        Returns (values, valid). Points behind the camera or outside the frame are invalid.
        """
        u, v, z = project_points(points, self.sfm, self.index[k], self.rotation, self.scale)
        h, w = image.shape[:2]
        valid = (z > 0) & (u >= margin) & (u < w - margin) & (v >= margin) & (v < h - margin)
        values = np.zeros(len(points), dtype=np.float32)
        values[valid] = image[v[valid].astype(int), u[valid].astype(int)]
        return values, valid


def local_std(gray: NDArray, size: int) -> NDArray[np.float32]:
    mean = cv2.blur(gray, (size, size))
    var = cv2.blur(gray * gray, (size, size)) - mean * mean
    return np.sqrt(np.maximum(var, 0)).astype(np.float32)


def components(mask: NDArray[np.bool_]) -> tuple[NDArray, int]:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, count - 1
