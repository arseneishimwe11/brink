"""Clean a raw reconstruction cloud and give it reliable normals."""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from brink.geometry import PointCloud
from brink.hazards.params import HazardParams


def choose_voxel(cloud: PointCloud) -> float:
    """Pick a working resolution from how tall the scene is."""
    z = cloud.xyz[:, 2]
    extent = float(np.percentile(z, 99) - np.percentile(z, 1))
    return float(np.clip(extent / 120.0, 0.01, 0.04))


def _to_o3d(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


def preprocess(cloud: PointCloud, params: HazardParams) -> tuple[PointCloud, float, float]:
    """Downsample, drop stray points, estimate normals facing the cameras.

    Normals are estimated on a coarser copy of the cloud and handed down to the
    fine points. Depth noise in a phone reconstruction is about as large as the
    fine point spacing, so normals fitted at the fine scale would be useless.

    Returns the cleaned cloud, the voxel size used, and the typical spacing
    between neighbouring points.
    """
    voxel = params.voxel or choose_voxel(cloud)
    pcd = _to_o3d(cloud.xyz)
    if cloud.rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(cloud.rgb.astype(np.float64) / 255.0)
    if cloud.view is not None:
        # Carry the view vectors through downsampling in the normals slot.
        pcd.normals = o3d.utility.Vector3dVector(cloud.view.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=params.outlier_std_ratio)
    xyz = np.asarray(pcd.points)
    view = np.asarray(pcd.normals).copy() if cloud.view is not None else None

    tree = cKDTree(xyz)
    probe = xyz[:: max(1, len(xyz) // 5000)]
    spacing = float(np.median(tree.query(probe, k=2)[0][:, 1]))

    coarse_voxel = max(params.coarse_factor * voxel, 2.0 * spacing)
    coarse = pcd.voxel_down_sample(coarse_voxel)
    coarse_view = np.asarray(coarse.normals).copy() if view is not None else None
    coarse.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=params.normal_radius_coarse * coarse_voxel, max_nn=60)
    )
    c_xyz = np.asarray(coarse.points)
    c_normals = np.asarray(coarse.normals).copy()
    if coarse_view is not None:
        to_cam = coarse_view
    elif len(cloud.cameras):
        _, nearest = cKDTree(cloud.cameras).query(c_xyz)
        to_cam = cloud.cameras[nearest] - c_xyz
    else:
        to_cam = np.tile([0.0, 0.0, 1.0], (len(c_xyz), 1))
    c_normals[(c_normals * to_cam).sum(1) < 0] *= -1
    _, parent = cKDTree(c_xyz).query(xyz)
    normals = c_normals[parent]

    rgb = None
    if cloud.rgb is not None:
        rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return PointCloud(xyz, rgb, normals, cloud.cameras, view), voxel, spacing


def slope_degrees(normals: np.ndarray) -> np.ndarray:
    """Slope of the surface each normal belongs to: 0 flat, 90 vertical, >90 facing down."""
    return np.degrees(np.arccos(np.clip(normals[:, 2], -1.0, 1.0)))
