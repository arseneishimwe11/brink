"""A photo from the capture with the hazards drawn on it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.domain import Hazard, Severity
from brink.hazards.engine import GeometryResult
from brink.outputs.colors import HAZARD_NAME, HAZARD_RGB
from brink.stages.sfm.base import SfMResult

_RANK = {Severity.DANGER: 2, Severity.WARNING: 1, Severity.NOTE: 0}


def _project(points_analysis: NDArray, sfm: SfMResult, i: int, rotation: NDArray, scale: float):
    """Pixel coordinates and depth of analysis-frame points in frame ``i``."""
    world = (points_analysis / scale) @ rotation  # undo: analysis = scale * R @ sfm
    cam = (world - sfm.positions[i]) @ sfm.rotations[i]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    x, y = cam[:, 0] / safe, cam[:, 1] / safe
    k1, k2 = sfm.distortion[0], sfm.distortion[1]
    r2 = x * x + y * y
    d = 1 + k1 * r2 + k2 * r2 * r2
    return sfm.fx * x * d + sfm.cx, sfm.fy * y * d + sfm.cy, z


def _best_frame(sfm: SfMResult, target: NDArray, outward: NDArray | None, rotation: NDArray, scale: float) -> int:
    """Frame that looks most squarely at ``target`` with it near the image centre."""
    seen = np.array([[c[0] for c in _project(target[None], sfm, i, rotation, scale)] for i in range(sfm.registered)])
    u, v, z = seen[:, 0], seen[:, 1], seen[:, 2]
    off = np.hypot((u - sfm.width / 2) / sfm.width, (v - sfm.height / 2) / sfm.height)
    cost = off + np.where(z > 0, 0, 10)
    if outward is not None:
        axes = sfm.rotations[:, :, 2] @ rotation.T  # optical axes in analysis frame
        cost += 0.6 * (1 + axes @ outward)  # facing the face: axis opposes outward
    return int(np.argmin(cost))


def _label(image: NDArray, text: str, anchor: tuple[int, int], rgb: tuple[int, int, int]) -> None:
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    (w, h), base = cv2.getTextSize(text, font, size, thick)
    x = int(np.clip(anchor[0] - w // 2, 4, image.shape[1] - w - 4))
    y = int(np.clip(anchor[1], h + 6, image.shape[0] - 6))
    cv2.rectangle(image, (x - 4, y - h - 5), (x + w + 4, y + base + 2), (255, 255, 255), -1)
    cv2.rectangle(image, (x - 4, y - h - 5), (x + w + 4, y + base + 2), rgb[::-1], 2)
    cv2.putText(image, text, (x, y), font, size, (20, 20, 20), thick, cv2.LINE_AA)


def short_label(hazard: Hazard, face_label: str, metric: bool) -> str:
    unit = "m" if metric else "units"
    name = HAZARD_NAME[hazard.type]
    depth = hazard.measurement("depth")
    slope = hazard.measurement("slope")
    if depth is not None:
        bound = ">=" if depth.is_lower_bound else ""
        return f"{face_label}: {name} {bound}{depth.value:.2f} {unit}"
    if slope is not None:
        return f"{face_label}: {name} {slope.value:.0f} deg"
    return f"{face_label}: {name}"


def save_hazard_image(
    geometry: GeometryResult, colours: NDArray[np.uint8], sfm: SfMResult, frames_dir: Path,
    rotation: NDArray, scale: float, metric: bool, path: str | Path,
) -> int | None:
    """Write the annotated photo. Returns the index of the frame used."""
    hazards = sorted(geometry.hazards, key=lambda h: (-_RANK[h.severity], -h.confidence))
    faces = {f.id: f for f in geometry.faces}
    if hazards:
        focus = hazards[0]
        face = faces.get(focus.face_id) if focus.face_id else None
        target = np.array(focus.location)
    elif geometry.faces:
        face = geometry.faces[0]
        target = np.array(face.centroid)
    else:
        return None
    outward = np.array(face.outward) if face is not None else None
    index = _best_frame(sfm, target, outward, rotation, scale)
    image = cv2.imread(str(Path(frames_dir) / sfm.names[index]))
    if image is None:
        return None

    # Paint hazard points over the photo, nearest surface only.
    xyz = geometry.cloud.xyz.astype(np.float64)
    base = geometry.cloud.rgb if geometry.cloud.rgb is not None else np.full_like(colours, 150)
    marked = np.abs(colours.astype(int) - (0.55 * base + 0.45 * base.mean(1, keepdims=True)).astype(int)).sum(1) > 40
    u, v, z = _project(xyz, sfm, index, rotation, scale)
    ok = (z > 0) & (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    order = np.flatnonzero(ok)[np.argsort(-z[ok])]  # far first, near last
    depth = np.full(image.shape[:2], np.inf, np.float32)
    overlay = image.copy()
    ui, vi = u[order].astype(int), v[order].astype(int)
    radius = 2
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu = np.clip(ui + du, 0, image.shape[1] - 1)
            vv = np.clip(vi + dv, 0, image.shape[0] - 1)
            depth[vv, uu] = z[order]
            sel = marked[order]
            overlay[vv[sel], uu[sel]] = colours[order][sel][:, ::-1]
            overlay[vv[~sel], uu[~sel]] = image[vv[~sel], uu[~sel]]
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    for n, hazard in enumerate(hazards[:6]):
        hu, hv, hz = _project(np.array(hazard.location)[None], sfm, index, rotation, scale)
        if hz[0] <= 0:
            continue
        label = faces[hazard.face_id].label if hazard.face_id in faces else "?"
        _label(image, short_label(hazard, label, metric), (int(hu[0]), int(hv[0]) - 14 - 26 * (n % 3)), HAZARD_RGB[hazard.type])
    cv2.imwrite(str(path), image)
    return index
