"""Cracks along the top edge of a face.

A tension crack behind a crest shows up as a thin dark line in the images and
as a thin gap or trough in the 3D surface. This detector needs to see the top
surface behind the edge. From the foot of a tall face that surface is hidden,
and the detector then reports that it could not check, rather than nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.domain import Face
from brink.geometry import PointCloud
from brink.hazards.faces import FaceSegment
from brink.hazards.imagery import FrameSet, components
from brink.hazards.params import HazardParams


@dataclass
class CrackRegion:
    centre: NDArray  # analysis frame
    length: float
    offset_from_edge: float
    image_score: float  # darkness of the line, in robust sigmas over the top surface
    has_gap: bool  # a matching thin gap exists in the 3D surface
    cells: int


@dataclass
class CrestCheck:
    visible: bool  # was enough of the top surface seen to look for cracks?
    cracks: list[CrackRegion]


def detect_cracks(
    cloud: PointCloud, slope: NDArray, face: Face, segment: FaceSegment,
    frames: FrameSet, voxel: float, params: HazardParams,
) -> CrestCheck:
    xyz = cloud.xyz.astype(np.float64)
    s_all, t_all = xyz @ segment.s_dir, xyz @ segment.t_dir
    fs = s_all[segment.index]
    ft = t_all[segment.index]
    s_lo, s_hi = np.percentile(fs, [3, 97])
    # Back edge of the face at crest height.
    top_band = xyz[segment.index, 2] > face.crest_z - 0.15 * face.height
    crest_t = float(np.percentile(ft[top_band], 90)) if top_band.sum() > 20 else float(np.percentile(ft, 98))
    depth_back = float(np.clip(0.6 * face.height, 0.5, 2.0))

    flat = slope < params.steep_deg
    zone = (
        flat & (s_all >= s_lo) & (s_all <= s_hi) & (t_all >= crest_t - 0.05) & (t_all <= crest_t + depth_back)
        & (np.abs(xyz[:, 2] - face.crest_z) < max(0.15 * face.height, 0.15))
    )
    cell = max(params.crack_cell_voxels * voxel, 0.015)
    s_edges = np.arange(s_lo, s_hi + cell, cell)
    t_edges = np.arange(crest_t - 0.05, crest_t + depth_back + cell, cell)
    ns, nt = len(s_edges) - 1, len(t_edges) - 1
    if ns < 5 or nt < 5 or zone.sum() < 200:
        return CrestCheck(False, [])
    si = np.clip(((s_all[zone] - s_edges[0]) / cell).astype(int), 0, ns - 1)
    ti = np.clip(((t_all[zone] - t_edges[0]) / cell).astype(int), 0, nt - 1)
    filled = np.zeros((ns, nt), dtype=bool)
    filled[si, ti] = True
    # The strip just behind the edge must be mostly reconstructed to count as seen.
    strip = filled[:, : max(3, int(0.3 / cell))]
    if strip.mean() < params.crest_min_coverage:
        return CrestCheck(False, [])
    top_z = float(np.median(xyz[zone, 2]))

    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
    thin_gap = closed & ~filled

    # Dark thin lines in the images, sampled on the top surface.
    gs, gt = np.meshgrid(np.arange(ns), np.arange(nt), indexing="ij")
    centres = (
        (0.5 * (s_edges[:-1] + s_edges[1:]))[gs.ravel(), None] * segment.s_dir
        + (0.5 * (t_edges[:-1] + t_edges[1:]))[gt.ravel(), None] * segment.t_dir
    )
    centres[:, 2] = top_z
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    total = np.zeros(ns * nt)
    hits = np.zeros(ns * nt)
    for k, gray in enumerate(frames.gray):
        if frames.centres[k][2] < top_z + 0.1:  # camera must be above the top surface
            continue
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.0)
        dark = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel) / (float(np.median(gray)) + 1e-3)
        v, valid = frames.sample(k, dark, centres)
        total += np.where(valid, v, 0.0)
        hits += valid
    seen = (hits >= 2).reshape(ns, nt) & closed
    if seen.sum() < 50:
        return CrestCheck(False, [])
    darkness = (total / np.maximum(hits, 1)).reshape(ns, nt)
    ref = darkness[seen]
    med = np.median(ref)
    spread = 1.4826 * np.median(np.abs(ref - med)) + 1e-6
    score = (darkness - med) / spread
    line = seen & (score >= params.crack_image_sigmas)
    line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8)) > 0

    cracks = []
    labels, count = components(line)
    for c in range(1, count + 1):
        m = labels == c
        if m.sum() < 5:
            continue
        pts = np.stack([gs[m], gt[m]], axis=1).astype(float) * cell
        centred = pts - pts.mean(0)
        values, vectors = np.linalg.eigh(centred.T @ centred / len(pts))
        length = float(4 * np.sqrt(max(values[1], 0)))
        width = float(4 * np.sqrt(max(values[0], 0))) + cell
        along_face = abs(vectors[0, 1])  # s-component of the long axis
        if length < params.crack_min_length or length < 4 * width or along_face < np.cos(np.radians(40)):
            continue
        where = m.ravel()
        cracks.append(CrackRegion(
            centre=centres[where].mean(0), length=length,
            offset_from_edge=float(pts[:, 1].mean() - 0.05),
            image_score=float(np.median(score[m])), has_gap=bool(thin_gap[m].mean() > 0.2), cells=int(m.sum()),
        ))
    return CrestCheck(True, cracks)
