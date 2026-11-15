"""Dense reconstruction: interface shared by all methods."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from brink.geometry import PointCloud
from brink.stages.sfm.base import SfMResult


class DenseReconstructor(Protocol):
    name: str

    def reconstruct(self, sfm: SfMResult, image_dir: Path, masks_dir: Path | None = None) -> PointCloud:
        """Dense coloured points in the SfM frame, with a view vector per point."""
        ...


def make_dense(name: str, device: str = "cpu") -> DenseReconstructor:
    if name == "stereo":
        from brink.stages.dense.stereo import StereoDense

        return StereoDense()
    if name == "monodepth":
        from brink.stages.dense.monodepth import MonoDepthDense

        return MonoDepthDense(device=device)
    raise ValueError(f"unknown dense method {name!r}; choose 'stereo' or 'monodepth'")
