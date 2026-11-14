"""Split steep ground into faces: connected, consistently oriented patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from brink.geometry import PointCloud
from brink.hazards.params import HazardParams
from brink.hazards.preprocess import slope_degrees


@dataclass
class FaceSegment:
    """Indices into the cleaned cloud plus the face's own coordinate frame.

    ``s`` runs along the face, ``t`` points horizontally into the ground and
    ``z`` is up, so a normal slope moves to larger ``t`` as it rises.
    """

    index: NDArray[np.int64]
    outward: NDArray[np.float64]
    s_dir: NDArray[np.float64]
    t_dir: NDArray[np.float64]


def _dbscan(xyz: NDArray, eps: float, min_points: int) -> NDArray[np.int64]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points), dtype=np.int64)


def _azimuth_peaks(az: NDArray, tolerance_deg: float) -> list[float]:
    """Dominant facing directions (radians) from a circular histogram."""
    bins = 36
    hist, _ = np.histogram(az, bins=bins, range=(-np.pi, np.pi))
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), kernel / kernel.sum(), mode="valid")
    centres = -np.pi + (np.arange(bins) + 0.5) * 2 * np.pi / bins
    peaks: list[float] = []
    order = np.argsort(smooth)[::-1]
    for i in order:
        if smooth[i] < 0.08 * smooth.max() or smooth[i] < smooth[(i - 1) % bins] or smooth[i] < smooth[(i + 1) % bins]:
            continue
        if all(abs(np.angle(np.exp(1j * (centres[i] - p)))) > np.radians(tolerance_deg) for p in peaks):
            peaks.append(float(centres[i]))
    return peaks


def segment_faces(cloud: PointCloud, voxel: float, spacing: float, params: HazardParams) -> tuple[list[FaceSegment], NDArray[np.int64]]:
    """Return face segments and, per cloud point, the index of its face (-1 if none)."""
    assert cloud.normals is not None
    slope = slope_degrees(cloud.normals)
    steep_idx = np.flatnonzero(slope >= params.steep_deg)
    if len(steep_idx) < params.min_face_points:
        return [], np.full(len(cloud), -1, dtype=np.int64)
    eps = max(2.5 * voxel, params.cluster_eps_spacings * spacing)
    labels = _dbscan(cloud.xyz[steep_idx], eps, 8)

    faces: list[FaceSegment] = []
    tol = np.radians(params.azimuth_tolerance_deg)
    for cid in range(labels.max() + 1):
        members = steep_idx[labels == cid]
        if len(members) < params.min_face_points:
            continue
        n = cloud.normals[members]
        # Faces that lean out have normals pointing down and back; their
        # horizontal part still points out of the ground.
        az = np.arctan2(n[:, 1], n[:, 0])
        for peak in _azimuth_peaks(az, params.azimuth_tolerance_deg):
            diff = np.abs(np.angle(np.exp(1j * (az - peak))))
            part = members[diff <= tol]
            if len(part) < params.min_face_points:
                continue
            sub = _dbscan(cloud.xyz[part], eps * 1.5, 8)
            for sid in range(sub.max() + 1):
                idx = part[sub == sid]
                if len(idx) < params.min_face_points:
                    continue
                horizontal = cloud.normals[idx][:, :2].mean(0)
                norm = np.linalg.norm(horizontal)
                if norm < 1e-6:
                    continue
                outward = np.array([horizontal[0] / norm, horizontal[1] / norm, 0.0])
                s_dir = np.array([-outward[1], outward[0], 0.0])
                xyz = cloud.xyz[idx]
                z_extent = np.percentile(xyz[:, 2], 98) - np.percentile(xyz[:, 2], 2)
                s = xyz @ s_dir
                s_extent = np.percentile(s, 98) - np.percentile(s, 2)
                if z_extent < params.min_face_height or s_extent < params.min_face_length:
                    continue
                faces.append(FaceSegment(idx, outward, s_dir, -outward))
    faces.sort(key=lambda f: -len(f.index))
    faces = _drop_hollow_walls(faces, cloud)
    face_of = np.full(len(cloud), -1, dtype=np.int64)
    for i, face in enumerate(faces):
        face_of[face.index] = i
    return faces, face_of


def _drop_hollow_walls(faces: list[FaceSegment], cloud: PointCloud) -> list[FaceSegment]:
    """Remove small segments that sit inside a bigger face's footprint.

    The back wall of an undercut faces the same way as the face it is cut into
    and would otherwise be reported as a tiny vertical face of its own.
    """
    keep: list[FaceSegment] = []
    boxes = []
    for face in faces:  # largest first
        xyz = cloud.xyz[face.index].astype(np.float64)
        z_lo, z_hi = np.percentile(xyz[:, 2], [2, 98])
        absorbed = False
        for parent, (p_s_lo, p_s_hi, p_z_lo, p_z_hi) in zip(keep, boxes, strict=True):
            if parent.outward @ face.outward < np.cos(np.radians(30)):
                continue
            s = xyz @ parent.s_dir
            s_lo, s_hi = np.percentile(s, [2, 98])
            inside = s_lo >= p_s_lo and s_hi <= p_s_hi and z_lo >= p_z_lo - 0.1 and z_hi <= p_z_hi
            if inside and (z_hi - z_lo) < 0.5 * (p_z_hi - p_z_lo):
                absorbed = True
                break
        if absorbed:
            continue
        s_own = xyz @ face.s_dir
        boxes.append((*np.percentile(s_own, [2, 98]), z_lo, z_hi))
        keep.append(face)
    return keep
