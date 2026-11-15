"""Dense reconstruction from single-image depth prediction.

A neural network (Depth Anything V2 Small, Apache-2.0) guesses relative depth
for each frame on its own. The guess has no scale, so each frame's depth is
fitted to the sparse points that camera solving placed in that frame. The
resulting depth maps go through the same cross-view fusion as stereo.

Kept as a comparison method. It needs the optional ``learned`` dependencies.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.geometry import PointCloud
from brink.stages.dense.fusion import DepthView, fuse
from brink.stages.masking import mask_name
from brink.stages.sfm.base import SfMResult

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"  # the Base and Large variants are non-commercial


def _fit_inverse_depth(relative: NDArray, inverse_depth: NDArray) -> tuple[float, float] | None:
    """Robust line ``inverse_depth = a * relative + b`` through sparse anchors."""
    weights = np.ones(len(relative))
    a = b = 0.0
    for _ in range(8):
        design = np.stack([relative, np.ones_like(relative)], axis=1) * weights[:, None]
        solution, *_ = np.linalg.lstsq(design, inverse_depth * weights, rcond=None)
        a, b = float(solution[0]), float(solution[1])
        residual = inverse_depth - (a * relative + b)
        spread = 1.4826 * np.median(np.abs(residual)) + 1e-9
        weights = 1.0 / np.maximum(1.0, np.abs(residual) / (2 * spread))
    if a <= 0:
        return None
    return a, b


class MonoDepthDense:
    name = "monodepth"

    def __init__(self, device: str = "cpu", tolerance: float = 0.02, min_agree: int = 2, stride: int = 1):
        self.device = device
        self.tolerance = tolerance
        self.min_agree = min_agree
        self.stride = stride  # use every n-th frame

    def reconstruct(self, sfm: SfMResult, image_dir: Path, masks_dir: Path | None = None) -> PointCloud:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(self.device).eval()
        K = sfm.K
        views: list[DepthView] = []
        for i in range(0, sfm.registered, self.stride):
            bgr = cv2.imread(str(Path(image_dir) / sfm.names[i]))
            bgr = cv2.undistort(bgr, K, sfm.distortion)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                inputs = processor(images=rgb, return_tensors="pt").to(self.device)
                predicted = model(**inputs).predicted_depth
                relative = torch.nn.functional.interpolate(
                    predicted[None], size=rgb.shape[:2], mode="bicubic", align_corners=False
                )[0, 0].cpu().numpy()

            # Anchor to the sparse points this frame observes.
            world_to_cam = sfm.rotations[i].T
            seen = sfm.points[sfm.visible[i]]
            cam = (seen - sfm.positions[i]) @ sfm.rotations[i]
            ok = cam[:, 2] > 1e-6
            cam = cam[ok]
            u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
            inside = (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])
            if inside.sum() < 30:
                continue
            fit = _fit_inverse_depth(relative[v[inside], u[inside]].astype(np.float64), 1.0 / cam[inside, 2])
            if fit is None:
                continue
            inverse = fit[0] * relative + fit[1]
            z = cam[inside, 2]
            valid = inverse > 1.0 / (3.0 * np.percentile(z, 99))  # drop sky and anything far beyond the scene
            depth = np.zeros(relative.shape, np.float32)
            depth[valid] = (1.0 / inverse[valid]).astype(np.float32)
            if masks_dir is not None:
                mask = cv2.imread(str(Path(masks_dir) / mask_name(sfm.names[i])), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    depth[mask < 128] = 0
            views.append(DepthView(depth, rgb, K, world_to_cam, sfm.positions[i].copy(), group=i))
        if not views:
            return PointCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), cameras=sfm.positions)
        return fuse(views, sfm.positions, min_agree=self.min_agree, tolerance=self.tolerance)
