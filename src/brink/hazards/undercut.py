"""Undercut and overhang detection. Two interchangeable methods.

``profile``: reads vertical cross-sections; an overhang is visible ground lying
behind and below the face surface above it.
``normals``: looks for patches of surface that face downward.

Both return ``OverhangRegion`` records in the face frame; the engine turns them
into hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from brink.geometry import PointCloud
from brink.hazards.params import HazardParams
from brink.hazards.profiles import FaceProfiles


@dataclass
class OverhangRegion:
    kind: str  # "undercut" (lip low on the face) or "overhang" (lip high)
    s_min: float
    s_max: float
    depth: float
    depth_spread: float
    lip_z: float
    hollow_z: float
    t_center: float
    support: int  # slices or points backing the detection
    back_seen: bool  # False: depth is only a lower bound

    @property
    def width(self) -> float:
        return self.s_max - self.s_min

    @property
    def s_center(self) -> float:
        return 0.5 * (self.s_min + self.s_max)


class UndercutDetector(Protocol):
    name: str

    def detect(
        self, cloud: PointCloud, slope: NDArray, profiles: FaceProfiles, voxel: float, params: HazardParams
    ) -> list[OverhangRegion]: ...


class ProfileDetector:
    name = "profile"

    def detect(self, cloud, slope, profiles, voxel, params) -> list[OverhangRegion]:
        if not profiles.slices:
            return []
        return self._detect_kind("undercut", profiles, voxel, params) + self._detect_kind("overhang", profiles, voxel, params)

    def _detect_kind(self, kind: str, profiles: FaceProfiles, voxel: float, params: HazardParams) -> list[OverhangRegion]:
        slices = profiles.slices
        readings = [p.reading(kind) for p in slices]
        raw = np.array([r.depth for r in readings])
        # What a plain face with this slope would read in each slice: slightly
        # negative for a normal slope, the full lean for a face that tips out.
        slopes = np.array([p.slope_deg for p in slices])
        slope = float(np.nanmedian(slopes)) if np.isfinite(slopes).any() else 90.0
        lean = np.tan(np.radians(np.clip(slope, 20.0, 160.0) - 90.0))
        heights = np.array([p.height for p in slices])
        expected = np.where(lean > 0, heights * lean, profiles.bin_height * lean)
        excess = raw - expected
        # Noise and roughness add a constant on top. Estimate it from the
        # quietest part of the face (an undercut may span half of it, so the
        # median would swallow the hazard), and never let it exceed what the
        # scatter of the face surface itself can explain.
        floor = float(np.percentile(excess, 20))
        quiet = excess[excess <= np.median(excess)]
        mad = max(float(1.4826 * np.median(np.abs(quiet - floor))), 0.5 * voxel)
        sigmas = np.array([p.front_sigma for p in slices])
        scatter = float(np.nanmedian(sigmas)) if np.isfinite(sigmas).any() else voxel
        bias = float(np.clip(floor, 0.0, 2 * voxel + 3 * scatter))
        local = excess - bias
        flagged = (local >= params.min_depth) & (excess >= floor + params.noise_sigmas * mad)
        # Reported depth is the total overhang at that spot, lean included.
        depths = local + np.maximum(expected, 0.0)

        regions: list[OverhangRegion] = []
        i = 0
        while i < len(slices):
            if not flagged[i]:
                i += 1
                continue
            j = i
            # Extend over neighbours, bridging one unflagged slice.
            while j + 1 < len(slices) and (
                flagged[j + 1] or (j + 2 < len(slices) and flagged[j + 2])
            ) and slices[j + 1].s_center - slices[j].s_center < 2.5 * profiles.slice_width:
                j += 1
            run = [r for r, f in zip(readings[i : j + 1], flagged[i : j + 1], strict=True) if f]
            width = slices[j].s_center - slices[i].s_center + profiles.slice_width
            if width >= params.min_region_width and len(run) >= 2:
                d = depths[i : j + 1][flagged[i : j + 1]]
                regions.append(
                    OverhangRegion(
                        kind=kind,
                        s_min=slices[i].s_center - profiles.slice_width / 2,
                        s_max=slices[j].s_center + profiles.slice_width / 2,
                        depth=float(np.percentile(d, 80)),
                        depth_spread=float(1.4826 * np.median(np.abs(d - np.median(d)))),
                        lip_z=float(np.median([r.lip_z for r in run])),
                        hollow_z=float(np.median([r.hollow_z for r in run])),
                        t_center=float(np.median([0.5 * (r.lip_t + r.hollow_t) for r in run])),
                        support=len(run),
                        back_seen=float(np.mean([r.back_seen for r in run])) >= 0.5,
                    )
                )
            i = j + 1
        return regions


class NormalsDetector:
    name = "normals"

    def detect(self, cloud, slope, profiles, voxel, params) -> list[OverhangRegion]:
        seg = profiles.segment
        if not profiles.slices:
            return []
        xyz = cloud.xyz.astype(np.float64)
        s, t = xyz @ seg.s_dir, xyz @ seg.t_dir
        s_lo = profiles.slices[0].s_center - profiles.slice_width
        s_hi = profiles.slices[-1].s_center + profiles.slice_width
        face_t = t[seg.index]
        near = (
            (slope >= params.downward_deg)
            & (s >= s_lo)
            & (s <= s_hi)
            & (t >= face_t.min() - 0.3)
            & (t <= face_t.max() + 1.5)
        )
        idx = np.flatnonzero(near)
        if len(idx) < 15:
            return []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[idx])
        labels = np.asarray(pcd.cluster_dbscan(eps=4 * voxel, min_points=10))
        regions = []
        for c in range(labels.max() + 1):
            m = idx[labels == c]
            if len(m) < 15:
                continue
            ms, mt, mz = s[m], t[m], xyz[m, 2]
            depth = float(np.percentile(mt, 97) - np.percentile(mt, 3))
            width = float(np.percentile(ms, 98) - np.percentile(ms, 2))
            if depth < params.min_depth or width < params.min_region_width:
                continue
            lip = float(np.percentile(mz, 95))
            base = min(p.z_floor for p in profiles.slices)
            top = max(p.z_crest for p in profiles.slices)
            low = (lip - base) <= params.undercut_max_lip_fraction * (top - base)
            regions.append(
                OverhangRegion(
                    kind="undercut" if low else "overhang",
                    s_min=float(np.percentile(ms, 2)),
                    s_max=float(np.percentile(ms, 98)),
                    depth=depth,
                    depth_spread=float("nan"),
                    lip_z=lip,
                    hollow_z=float(np.percentile(mz, 5)),
                    t_center=float(np.median(mt)),
                    support=len(m),
                    back_seen=True,
                )
            )
        return regions


DETECTORS: dict[str, type] = {"profile": ProfileDetector, "normals": NormalsDetector}


def make_detector(name: str) -> UndercutDetector:
    return DETECTORS[name]()
