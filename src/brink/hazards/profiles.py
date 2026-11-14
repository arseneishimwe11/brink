"""Vertical cross-sections through a face.

Each face is cut into thin slices along its length. In every slice the ground
is summarised per height bin by two numbers in the face frame (``t`` grows
into the ground):

- ``t_front``: where the face's own surface is, taken from its steep points;
- ``t_back``: how far back the visible surface reaches at that height, which
  under an overhang is the floor or back wall of the hollow.

Slope, height and undercut depth are all read from these profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from brink.geometry import PointCloud
from brink.hazards.faces import FaceSegment
from brink.hazards.params import HazardParams

NAN = float("nan")


@dataclass
class SliceProfile:
    s_center: float
    z_floor: float
    z_crest: float
    z: NDArray[np.float64]  # bin centres
    t_front: NDArray[np.float64]
    t_back: NDArray[np.float64]
    n_front: NDArray[np.int64]
    back_is_wall: NDArray[np.bool_]
    slope_deg: float = NAN
    # Best overhang found in this slice (depth <= 0 means none).
    depth: float = 0.0
    lip_z: float = NAN
    lip_t: float = NAN
    hollow_z: float = NAN
    hollow_t: float = NAN
    hollow_back_seen: bool = False

    @property
    def height(self) -> float:
        return self.z_crest - self.z_floor


@dataclass
class FaceProfiles:
    segment: FaceSegment
    slice_width: float
    bin_height: float
    slices: list[SliceProfile] = field(default_factory=list)
    plane_slope_deg: float = NAN


def _kth(values: NDArray, fraction: float, largest: bool) -> float:
    """A trimmed extreme: ignores the most extreme few values."""
    k = min(len(values) - 1, max(1, int(fraction * len(values))))
    part = np.partition(values, len(values) - 1 - k if largest else k)
    return float(part[len(values) - 1 - k] if largest else part[k])


def _front_group(t_sorted: NDArray, gap: float, skip_stray: bool) -> NDArray:
    """Leading run of a sorted array, cut at the first wide gap.

    With ``skip_stray`` a leading run of one or two points is treated as noise
    hanging in the air and passed over. Only use it where bins are well filled.
    """
    start = 0
    while start < len(t_sorted):
        breaks = np.flatnonzero(np.diff(t_sorted[start:]) > gap)
        end = start + breaks[0] + 1 if len(breaks) else len(t_sorted)
        if not skip_stray or end - start >= 3 or end == len(t_sorted):
            return t_sorted[start:end]
        start = end
    return t_sorted


def _theil_sen(z: NDArray, t: NDArray) -> float:
    i, j = np.triu_indices(len(z), k=1)
    dz = z[j] - z[i]
    ok = np.abs(dz) > 1e-9
    return float(np.median((t[j] - t[i])[ok] / dz[ok]))


def _plane_slope(xyz: NDArray, outward: NDArray) -> float:
    """Slope of the best-fit plane through the middle band of a face."""
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [30, 90])
    band = xyz[(z >= lo) & (z <= hi)]
    if len(band) < 30:
        return NAN
    centred = band - band.mean(0)
    _, _, vt = np.linalg.svd(centred[:: max(1, len(band) // 5000)], full_matrices=False)
    n = vt[2]
    if n @ outward < 0:
        n = -n
    return float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))


def compute_profiles(
    cloud: PointCloud,
    slope: NDArray,
    face_of: NDArray,
    segments: list[FaceSegment],
    face_index: int,
    voxel: float,
    spacing: float,
    params: HazardParams,
) -> FaceProfiles:
    """Build the cross-sections of ``segments[face_index]``."""
    segment = segments[face_index]
    # Other faces that cross this one (a corner, an end slope) would look like
    # ground reaching far behind it, so their points are left out.
    crossing = np.zeros(len(cloud), dtype=bool)
    limit = np.cos(np.radians(params.azimuth_tolerance_deg))
    for j, other in enumerate(segments):
        if j != face_index and other.outward @ segment.outward < limit:
            crossing |= face_of == j
    xyz = cloud.xyz.astype(np.float64)
    s_all, t_all, z_all = xyz @ segment.s_dir, xyz @ segment.t_dir, xyz[:, 2]
    fs, ft, fz = s_all[segment.index], t_all[segment.index], z_all[segment.index]
    s_lo, s_hi = np.percentile(fs, [1, 99])
    t_lo, t_hi = np.percentile(ft, [2, 98])
    z_lo, z_hi = np.percentile(fz, [2, 98])
    reach_back = max(1.0, 0.5 * (z_hi - z_lo))
    window = (
        (s_all >= s_lo)
        & (s_all <= s_hi)
        & (t_all >= t_lo - params.front_reach)
        & (t_all <= t_hi + reach_back)
        & (z_all >= z_lo - 0.5)
        & (z_all <= z_hi + 0.5)
        & ~crossing
    )
    idx = np.flatnonzero(window)
    s, t, z = s_all[idx], t_all[idx], z_all[idx]
    flat = slope[idx] < params.steep_deg
    wall = face_of[idx] == face_index

    width = params.slice_width or float(np.clip((s_hi - s_lo) / 40.0, 3 * voxel, 0.25))
    dz = params.bin_height_voxels * voxel
    n_slices = max(1, int(np.ceil((s_hi - s_lo) / width)))
    slice_of = np.clip(((s - s_lo) / width).astype(int), 0, n_slices - 1)
    order = np.argsort(slice_of, kind="stable")
    bounds = np.searchsorted(slice_of[order], np.arange(n_slices + 1))

    cams = cloud.cameras.astype(np.float64)
    cam_t, cam_z = cams @ segment.t_dir, cams[:, 2]

    out = FaceProfiles(segment, width, dz, plane_slope_deg=_plane_slope(xyz[segment.index], segment.outward))
    # Columns must stay wide enough to hold a few points per height bin.
    column_width = max(params.column_voxels * voxel, params.column_spacings * spacing)
    for i in range(n_slices):
        sel = order[bounds[i] : bounds[i + 1]]
        if len(sel) < 30 or wall[sel].sum() < 10:
            continue
        ss, ts, zs, fl, wl = s[sel], t[sel], z[sel], flat[sel], wall[sel]
        t_wall_min = np.percentile(ts[wl], 2)
        apron = fl & (ts >= t_wall_min - params.front_reach) & (ts <= t_wall_min + 0.1) & (zs < np.percentile(zs[wl], 50))
        z_floor = float(np.median(zs[apron])) if apron.sum() >= 5 else float(np.percentile(zs[wl], 2))
        z_crest = float(np.percentile(zs[wl], 98))
        if z_crest - z_floor < 3 * dz:
            continue
        n_bins = int(np.ceil((z_crest - z_floor) / dz)) + 2
        bin_of = np.floor((zs - z_floor) / dz + 0.5).astype(int)
        inside = (bin_of >= 0) & (bin_of < n_bins)
        ss, ts, fl, wl, bin_of = ss[inside], ts[inside], fl[inside], wl[inside], bin_of[inside]

        # Slice-wide profile: enough points per bin for a steady slope fit.
        t_front, t_back, n_front, back_wall = _bin_profile(ts, fl, wl, bin_of, n_bins, params, voxel, robust=True)
        prof = SliceProfile(
            float(s_lo + (i + 0.5) * width), z_floor, z_crest,
            z_floor + dz * np.arange(n_bins), t_front, t_back, n_front, back_wall,
        )

        # Overhang is measured in thin columns and the slice takes their median.
        # A wide slice across a ridge (where this face meets a side slope) would
        # pair a lip on one side with lower ground on the other and read it as
        # a hollow; within a thin column that cannot happen.
        columns = []
        col_of = np.floor((ss - (s_lo + i * width)) / column_width).astype(int)
        for c in np.unique(col_of):
            m = col_of == c
            if wl[m].sum() < 3:
                continue
            cf, cb, cn, cw = _bin_profile(ts[m], fl[m], wl[m], bin_of[m], n_bins, params, voxel, robust=False)
            col = SliceProfile(prof.s_center, z_floor, z_crest, prof.z, cf, cb, cn, cw)
            _measure_overhang(col, dz, params, cam_t, cam_z, min_front=1)
            columns.append(col)
        if len(columns) >= 2:
            columns.sort(key=lambda col: col.depth)
            mid = columns[len(columns) // 2]
            prof.depth, prof.lip_z, prof.lip_t = mid.depth, mid.lip_z, mid.lip_t
            prof.hollow_z, prof.hollow_t, prof.hollow_back_seen = mid.hollow_z, mid.hollow_t, mid.hollow_back_seen
        _measure_slope(prof, params)
        out.slices.append(prof)
    return out


def _bin_profile(
    ts: NDArray, fl: NDArray, wl: NDArray, bin_of: NDArray, n_bins: int,
    params: HazardParams, voxel: float, robust: bool,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Front and back of the surface per height bin.

    ``robust`` trims the extremes, which needs several points per bin.
    """
    t_front = np.full(n_bins, np.nan)
    t_back = np.full(n_bins, np.nan)
    n_front = np.zeros(n_bins, dtype=np.int64)
    back_wall = np.zeros(n_bins, dtype=bool)
    order = np.argsort(bin_of, kind="stable")
    edges = np.searchsorted(bin_of[order], np.arange(n_bins + 1))
    for k in range(n_bins):
        idx = order[edges[k] : edges[k + 1]]
        if len(idx) == 0:
            continue
        tb = ts[idx]
        tw = tb[wl[idx]]
        n_front[k] = len(tw)
        if robust and len(tw) >= 3:
            t_front[k] = _kth(tw, 0.05, largest=False)
        elif not robust and len(tw) >= 1:
            t_front[k] = tw.min()
        group = _front_group(np.sort(tb), params.group_gap, skip_stray=robust)
        t_back[k] = _kth(group, 0.02, largest=True) if robust and len(group) >= 3 else group[-1]
        near = np.abs(tb - t_back[k]) < 2.5 * voxel
        back_wall[k] = (~fl[idx][near]).mean() >= 0.5
    return t_front, t_back, n_front, back_wall


def _measure_overhang(
    prof: SliceProfile, dz: float, params: HazardParams, cam_t: NDArray, cam_z: NDArray, min_front: int = 3
) -> None:
    """Find the deepest place where visible ground lies behind and below the face above it."""
    n = len(prof.z)
    gap = params.lip_gap_bins
    clearance = max(2 * dz, 0.05)
    min_drop = max(params.min_hollow_bins * dz, params.min_hollow_height)
    usable = (prof.n_front >= min_front) & (prof.z >= prof.z_floor + clearance) & np.isfinite(prof.t_front)
    front = np.where(usable, prof.t_front, np.inf)
    # Suffix minimum: the furthest-forward face surface at or above each bin.
    lip_t = np.full(n, np.inf)
    lip_k = np.full(n, -1)
    best, best_k = np.inf, -1
    for k in range(n - 1, -1, -1):
        if front[k] < best:
            best, best_k = front[k], k
        lip_t[k], lip_k[k] = best, best_k

    candidates = []
    for k1 in range(n - gap):
        k2 = lip_k[k1 + gap]
        if k2 < 0 or not np.isfinite(prof.t_back[k1]):
            continue
        depth = prof.t_back[k1] - lip_t[k1 + gap]
        if depth > 0:
            candidates.append((depth, k1, k2))
    for depth, k1, k2 in sorted(candidates, reverse=True):
        # The hollow only counts if it has real height and some camera looked in
        # under the lip. Ground that merely dips behind a ridge is shallow and
        # is seen from over the top instead.
        p_t, p_z, l_t, l_z = prof.t_back[k1], prof.z[k1], prof.t_front[k2], prof.z[k2]
        drop = l_z - p_z
        if drop < min_drop:
            continue
        ok = cam_t < min(l_t, p_t) - 1e-6  # cameras standing in front of the lip
        ray_z = cam_z[ok] + (p_z - cam_z[ok]) * (l_t - cam_t[ok]) / (p_t - cam_t[ok])
        if len(cam_t) == 0 or np.any(ray_z < l_z + params.ray_tolerance * drop):
            prof.depth = float(depth)
            prof.lip_z, prof.lip_t = float(l_z), float(l_t)
            prof.hollow_z, prof.hollow_t = float(p_z), float(p_t)
            prof.hollow_back_seen = bool(prof.back_is_wall[k1])
            return


def _measure_slope(prof: SliceProfile, params: HazardParams) -> None:
    h = prof.height
    lo = prof.z_floor + 0.25 * h
    if prof.depth >= params.min_depth and np.isfinite(prof.lip_z):
        lo = max(lo, prof.lip_z)
    hi = prof.z_crest - 0.1 * h
    use = (prof.z >= lo) & (prof.z <= hi) & (prof.n_front >= 3) & np.isfinite(prof.t_front)
    if use.sum() < 4:
        return
    b = _theil_sen(prof.z[use], prof.t_front[use])
    prof.slope_deg = float(np.degrees(np.arctan2(1.0, b)))
