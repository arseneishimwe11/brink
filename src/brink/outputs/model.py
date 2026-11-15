"""Colour the reconstructed cloud by hazard and save it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from brink.domain import HazardType
from brink.hazards.engine import GeometryResult
from brink.outputs.colors import HAZARD_RGB


def hazard_colours(geometry: GeometryResult) -> NDArray[np.uint8]:
    """Per-point RGB: muted photo colour, with hazards painted over it."""
    cloud = geometry.cloud
    base = cloud.rgb if cloud.rgb is not None else np.full((len(cloud), 3), 150, np.uint8)
    grey = base.astype(np.float32).mean(1, keepdims=True)
    colours = 0.55 * base + 0.45 * grey  # mute, so hazard colours stand out

    def paint(mask_or_index, rgb, strength):
        colours[mask_or_index] = (1 - strength) * colours[mask_or_index] + strength * np.array(rgb, np.float32)

    face_index = {face.id: i for i, face in enumerate(geometry.faces)}
    for hazard in geometry.hazards:
        if hazard.type is HazardType.OVERSTEEP and hazard.face_id in face_index:
            paint(geometry.segments[face_index[hazard.face_id]].index, HAZARD_RGB[hazard.type], 0.45)
    for hazard in geometry.hazards:
        if hazard.type is HazardType.OVERHANG and hazard.face_id in face_index and hazard.measurement("lip_height") is None:
            paint(geometry.segments[face_index[hazard.face_id]].index, HAZARD_RGB[hazard.type], 0.55)
    if geometry.hazard_of_point is not None:
        for i, hazard in enumerate(geometry.hazards):
            mask = geometry.hazard_of_point == i
            if mask.any():
                paint(mask, HAZARD_RGB[hazard.type], 0.85)
    return np.clip(colours, 0, 255).astype(np.uint8)


def save_model(geometry: GeometryResult, path: str | Path) -> NDArray[np.uint8]:
    colours = hazard_colours(geometry)
    geometry.cloud.save_ply(path, rgb=colours)
    return colours
