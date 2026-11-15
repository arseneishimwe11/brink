"""Camera solving (structure from motion): interface and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class SfMResult:
    """Solved cameras and sparse points, in an arbitrary frame and scale.

    ``rotations`` are camera-to-world; ``positions`` are camera centres.
    Intrinsics describe an undistorted pinhole after applying ``distortion``
    (OpenCV order k1, k2, p1, p2).
    """

    names: list[str]
    rotations: NDArray[np.float64]  # (N, 3, 3)
    positions: NDArray[np.float64]  # (N, 3)
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: NDArray[np.float64]
    width: int
    height: int
    points: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    point_colors: NDArray[np.uint8] = field(default_factory=lambda: np.zeros((0, 3), np.uint8))
    # For each image, indices into ``points`` of the sparse points it observes.
    visible: list[NDArray[np.int64]] = field(default_factory=list)
    frames_given: int = 0
    mean_reprojection_error: float = float("nan")
    method: str = ""

    @property
    def registered(self) -> int:
        return len(self.names)

    @property
    def K(self) -> NDArray[np.float64]:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, names=np.array(self.names), rotations=self.rotations, positions=self.positions,
            intr=np.array([self.fx, self.fy, self.cx, self.cy, self.width, self.height]),
            distortion=self.distortion, points=self.points, point_colors=self.point_colors,
            visible=np.array(self.visible, dtype=object), frames_given=self.frames_given,
            err=self.mean_reprojection_error, method=self.method,
        )

    @classmethod
    def load(cls, path: str | Path) -> SfMResult:
        with np.load(path, allow_pickle=True) as z:
            fx, fy, cx, cy, w, h = z["intr"]
            return cls(
                [str(n) for n in z["names"]], z["rotations"], z["positions"], float(fx), float(fy), float(cx), float(cy),
                z["distortion"], int(w), int(h), z["points"], z["point_colors"], list(z["visible"]),
                int(z["frames_given"]), float(z["err"]), str(z["method"]),
            )


class SfMSolver(Protocol):
    name: str

    def solve(self, image_dir: Path, names: list[str], work_dir: Path, masks_dir: Path | None = None) -> SfMResult | None:
        """Return solved cameras, or None when no usable reconstruction was found."""
        ...


def make_solver(name: str, device: str = "cpu") -> SfMSolver:
    from brink.stages.sfm.colmap import ColmapSolver

    presets = {
        "colmap_sift": {"features": "sift", "mapper": "incremental"},
        "colmap_sift_global": {"features": "sift", "mapper": "global"},
    }
    # Learned features (ALIKED + LightGlue) exist in COLMAP 4 but the PyPI
    # build of pycolmap has no ONNX runtime and aborts if asked for them.
    if name not in presets:
        raise ValueError(f"unknown SfM method {name!r}; choose from {sorted(presets)}")
    import os

    threads = int(os.environ.get("BRINK_THREADS", os.cpu_count() or 4))
    return ColmapSolver(name=name, device=device, threads=threads, **presets[name])
