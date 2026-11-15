"""Standing water at the foot of a face.

Water gives itself away in two ways at once. In 3D, the floor has a gap there:
a mirror surface cannot be matched between views, so no points land on it.
In the images, the patch does not look the same from different places (it
reflects whatever is behind it) or is unusually smooth and bright (it reflects
sky). Either clue alone is common on dry ground, so both are required.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.domain import Face
from brink.geometry import PointCloud
from brink.hazards.faces import FaceSegment
from brink.hazards.imagery import FrameSet, components, local_std
from brink.hazards.params import HazardParams
from brink.hazards.profiles import FaceProfiles


@dataclass
class WaterRegion:
    centre: NDArray  # analysis frame
    extent_along_face: float
    area: float
    gap_share: float  # share of the patch with no 3D points
    image_score: float  # how unlike dry ground the patch looks, in robust sigmas
    cells: int


def detect_water(
    cloud: PointCloud, face: Face, segment: FaceSegment, profiles: FaceProfiles,
    frames: FrameSet, voxel: float, params: HazardParams, debug: dict | None = None,
) -> list[WaterRegion]:
    if len(profiles.slices) < 3:
        return []
    xyz = cloud.xyz.astype(np.float64)
    s_all, t_all = xyz @ segment.s_dir, xyz @ segment.t_dir

    # Toe line and floor height along the face, from the cross-sections.
    s_c = np.array([p.s_center for p in profiles.slices])
    toe_t = np.array([np.nanmin(p.t_front[: max(4, len(p.t_front) // 5)]) if np.isfinite(p.t_front[: max(4, len(p.t_front) // 5)]).any() else np.nan for p in profiles.slices])
    floor_z = np.array([p.z_floor for p in profiles.slices])
    ok = np.isfinite(toe_t)
    if ok.sum() < 3:
        return []
    s_c, toe_t, floor_z = s_c[ok], toe_t[ok], floor_z[ok]

    reach = float(np.clip(0.8 * face.height, 0.8, 3.0))
    cell = max(params.water_cell_voxels * voxel, 0.04)
    s_edges = np.arange(s_c.min(), s_c.max() + cell, cell)
    t_rel_edges = np.arange(-reach, 0.5 * cell, cell)  # distance in front of the toe, negative = open side
    ns, nt = len(s_edges) - 1, len(t_rel_edges) - 1
    if ns < 3 or nt < 3:
        return []
    s_mid = 0.5 * (s_edges[:-1] + s_edges[1:])
    toe_at = np.interp(s_mid, s_c, toe_t)
    floor_at = np.interp(s_mid, s_c, floor_z)

    # Which cells of the floor in front of the toe hold 3D points near floor level.
    si = np.floor((s_all - s_edges[0]) / cell).astype(int)
    inside = (si >= 0) & (si < ns)
    t_rel = np.full(len(xyz), np.nan)
    t_rel[inside] = t_all[inside] - toe_at[si[inside]]
    ti = np.floor((t_rel - t_rel_edges[0]) / cell)
    near_floor = np.zeros(len(xyz), dtype=bool)
    near_floor[inside] = np.abs(xyz[inside, 2] - floor_at[si[inside]]) < max(4 * voxel, 0.08)
    use = inside & (ti >= 0) & (ti < nt) & near_floor
    filled = np.zeros((ns, nt), dtype=bool)
    filled[si[use], ti[use].astype(int)] = True
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    # Image evidence for every enclosed cell, sampled on the floor plane.
    gs, gt = np.meshgrid(np.arange(ns), np.arange(nt), indexing="ij")
    centres = (
        s_mid[gs.ravel(), None] * segment.s_dir
        + (toe_at[gs.ravel()] + 0.5 * (t_rel_edges[gt.ravel()] + t_rel_edges[gt.ravel() + 1]))[:, None] * segment.t_dir
    )
    centres[:, 2] = floor_at[gs.ravel()]
    values, texture, seen = [], [], []
    for k, gray in enumerate(frames.gray):
        level = float(np.median(gray)) + 1e-3
        # Brightness is compared after heavy blurring: the assumed floor height
        # is never exact, and at a shallow viewing angle a small height error
        # shifts where a cell lands in the image. Blurring removes the soil's
        # own texture from that comparison and leaves what water changes, the
        # broad brightness of what it reflects.
        broad = cv2.GaussianBlur(gray, (0, 0), 5.0)
        fine = cv2.GaussianBlur(gray, (0, 0), 1.0)
        contrast = local_std(fine, 9) / (cv2.blur(fine, (9, 9)) + 4.0)  # independent of light and shadow
        v, valid = frames.sample(k, broad / level, centres)
        x, _ = frames.sample(k, contrast, centres)
        # Only views from the open side, looking down onto the floor.
        cam_t = frames.centres[k] @ segment.t_dir
        valid &= cam_t < (centres @ segment.t_dir) - 0.3
        values.append(np.where(valid, v, np.nan))
        texture.append(np.where(valid, x, np.nan))
        seen.append(valid)
    values, texture = np.array(values), np.array(texture)
    views = np.array(seen).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells no frame saw are all-NaN
        brightness = np.nanmean(values, axis=0)
        change = np.nanstd(values, axis=0)
        smoothness = np.nanmean(texture, axis=0)
    # Cells at the fringe of what the cameras covered are seen by few frames
    # and are not judged.
    enough = (views >= max(4, 0.5 * views.max())).reshape(ns, nt)
    brightness, change, smoothness = (a.reshape(ns, nt) for a in (brightness, change, smoothness))

    # A gap is floor that the cameras looked at but that did not reconstruct.
    # (A puddle against the toe is not ringed by floor, so being enclosed
    # cannot be required.)
    gap = enough & ~closed
    dry = closed & enough  # reference: floor that did reconstruct
    if dry.sum() < 20:
        return []

    def sigmas(field: NDArray) -> NDArray:
        ref = field[dry]
        med = np.median(ref)
        spread = 1.4826 * np.median(np.abs(ref - med)) + 1e-6
        return (field - med) / spread

    z_change, z_bright, z_smooth = sigmas(change), sigmas(brightness), -sigmas(smoothness)
    odd = np.maximum(z_change, np.abs(z_bright))
    wet = cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if debug is not None:
        debug.update(filled=filled, gap=gap, odd=odd, wet=wet, z_change=z_change,
                     z_bright=z_bright, z_smooth=z_smooth, views=views.reshape(ns, nt), cell=cell)
    # Each gap is judged as a whole. Shadow and damp soil still have texture
    # and reconstruct, so they never become gaps; a gap whose overall
    # brightness differs from the dry floor, or changes as the camera moves,
    # is a reflecting surface.
    regions = []
    labels, count = components(wet)
    ring_kernel = np.ones((7, 7), np.uint8)
    for c in range(1, count + 1):
        m = labels == c
        area = float(m.sum() * cell * cell)
        if area < params.water_min_area or m.sum() < params.water_min_cells:
            continue
        # A missing patch bigger than half the floor that did reconstruct says
        # the floor was modelled badly, not that it is under water.
        if m.sum() > 0.5 * dry.sum():
            continue
        # A puddle sits in floor that did reconstruct. A patch surrounded by
        # more missing floor is just ground too far or too oblique to model.
        ring = (cv2.dilate(m.astype(np.uint8), ring_kernel) > 0) & ~m & enough
        if ring.sum() == 0 or closed[ring].mean() < params.water_min_surround:
            continue
        score = float(max(np.nanmedian(z_change[m]), abs(np.nanmedian(z_bright[m]))))
        if score < params.water_image_sigmas:
            continue
        where = m.ravel()
        centre = centres[where].mean(0)
        extent = float((gs[m].max() - gs[m].min() + 1) * cell)
        regions.append(WaterRegion(centre, extent, area, float(gap[m].mean()), score, int(m.sum())))
    return regions
