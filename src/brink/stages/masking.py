"""Masking: mark image areas that are not bare ground, so later stages ignore them.

Phase 0 uses simple colour and texture rules, with no model to license:
- sky: smooth, bright, low-colour area connected to the top of the frame;
- vegetation: clearly green pixels.

People and machines are not detected here. Anything that moves between frames
is already dropped by the cross-view check in dense reconstruction; parked
machines are not, and will be reconstructed as if they were ground. A
segmentation model is the place to fix that (see docs/LICENSES.md for the
ones that may and may not be used).

A mask is an 8-bit image: 255 = use this pixel, 0 = ignore it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class MaskingParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    sky_max_texture: float = 3.0  # local standard deviation, 0..255 scale
    sky_max_saturation: float = 0.35
    vegetation_min_excess_green: float = 0.10
    grow_px: int = 5  # masks are grown a little to cover blended edge pixels


def ground_mask(bgr: NDArray[np.uint8], params: MaskingParams) -> NDArray[np.uint8]:
    image = bgr.astype(np.float32)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(grey, (0, 0), 1.5)
    mean = cv2.blur(smooth, (11, 11))
    texture = np.sqrt(np.maximum(cv2.blur(smooth * smooth, (11, 11)) - mean * mean, 0))

    high, low = image.max(axis=2), image.min(axis=2)
    saturation = (high - low) / (high + 1.0)
    blue_or_neutral = (image[:, :, 0] >= image[:, :, 2] - 6) | (saturation < 0.12)  # B >= R, or nearly grey
    candidate = (
        (texture < params.sky_max_texture) & (saturation < params.sky_max_saturation)
        & blue_or_neutral & (grey > np.percentile(grey, 40))
    ).astype(np.uint8)
    # Keep only candidate areas that reach the top edge of the frame.
    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    touching = np.unique(labels[0, :])
    sky = np.isin(labels, touching[touching > 0])

    total = image.sum(axis=2) + 1.0
    excess_green = (2 * image[:, :, 1] - image[:, :, 0] - image[:, :, 2]) / total
    vegetation = excess_green > params.vegetation_min_excess_green
    vegetation = cv2.morphologyEx(vegetation.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    ignore = (sky | vegetation).astype(np.uint8)
    if params.grow_px > 0:
        size = 2 * params.grow_px + 1
        ignore = cv2.dilate(ignore, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return np.where(ignore > 0, 0, 255).astype(np.uint8)


def mask_name(frame_name: str) -> str:
    """COLMAP's convention: the mask of ``a.jpg`` is ``a.jpg.png``."""
    return frame_name + ".png"


def write_masks(image_dir: Path, names: list[str], out_dir: Path, params: MaskingParams | None = None) -> dict[str, float]:
    """Write one mask per frame. Returns the share of pixels ignored, per reason."""
    params = params or MaskingParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ignored = []
    for name in names:
        bgr = cv2.imread(str(Path(image_dir) / name))
        mask = ground_mask(bgr, params)
        cv2.imwrite(str(out_dir / mask_name(name)), mask)
        ignored.append(float((mask == 0).mean()))
    return {"ignored_share": float(np.mean(ignored)) if ignored else 0.0}
