"""Turn a synthetic scene into the point cloud a reconstruction might give.

Used by the geometry tier of the benchmark: it isolates the hazard geometry
code from camera solving and dense matching, while still honouring what the
cameras could and could not see.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from brink.geometry import PointCloud
from brink.synth.scene import Scene


class CloudNoise(BaseModel):
    """Imperfections applied to the ideal surface samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "typical"
    depth_sigma_rel: float = 0.004  # depth error as a fraction of range
    outlier_fraction: float = 0.005
    outlier_sigma: float = 0.15
    hole_count: int = 4
    hole_radius: float = 0.12
    spacing_rel: float = 0.0035  # point spacing as a fraction of camera range
    min_views: int = 3
    gravity_error_deg: float = 0.0
    scale_error: float = 0.0


NOISE_LEVELS = {
    "ideal": CloudNoise(name="ideal", depth_sigma_rel=0.0015, outlier_fraction=0.0, hole_count=0, spacing_rel=0.003),
    "typical": CloudNoise(name="typical"),
    "poor": CloudNoise(name="poor", depth_sigma_rel=0.008, outlier_fraction=0.02, hole_count=10, hole_radius=0.2, spacing_rel=0.006),
}


def raycasting_scene(scene: Scene) -> o3d.t.geometry.RaycastingScene:
    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(
        o3d.core.Tensor(scene.vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(scene.triangles.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
    )
    return rs


def _sample_surface(scene: Scene, lo: NDArray, hi: NDArray, density: float, rng: np.random.Generator):
    v, tri = scene.vertices, scene.triangles[scene.materials == 0]
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    centre = (a + b + c) / 3
    keep = np.all((centre >= lo) & (centre <= hi), axis=1)
    a, b, c = a[keep], b[keep], c[keep]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    n = int(area.sum() * density)
    pick = rng.choice(len(area), size=n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
    pts = w[:, :1] * a[pick] + w[:, 1:2] * b[pick] + w[:, 2:] * c[pick]
    normals = cross[pick] / (2 * area[pick, None] + 1e-12)
    return pts.astype(np.float32), normals.astype(np.float32)


def sample_cloud(
    scene: Scene,
    cameras: NDArray,
    noise: CloudNoise,
    seed: int = 0,
    region_margin: float = 2.5,
) -> PointCloud:
    """Sample the surface, keep what enough cameras saw, then add errors."""
    rng = np.random.default_rng(seed)
    spec = scene.spec
    span = spec.length / 2 + spec.height / np.tan(np.radians(40)) + 0.3
    lo = np.array([-span, min(0.0, spec.crest_y) - region_margin, -1.0])
    hi = np.array([span, max(spec.crest_y, 0.0) + 3.5, spec.height + 1.0])
    cams = np.asarray(cameras, dtype=np.float32)
    centre = np.array([0.0, spec.crest_y / 2, spec.height / 2])
    camera_range = float(np.median(np.linalg.norm(cams - centre, axis=1)))
    density = 1.0 / (noise.spacing_rel * camera_range) ** 2
    pts, normals = _sample_surface(scene, lo, hi, density, rng)

    views = cams[np.linspace(0, len(cams) - 1, min(24, len(cams))).astype(int)]
    rs = raycasting_scene(scene)
    seen = np.zeros(len(pts), dtype=np.int32)
    chosen = np.zeros(len(pts), dtype=np.int32)
    best = np.full(len(pts), np.inf, dtype=np.float32)
    for k, cam in enumerate(views):
        to_pt = pts - cam
        dist = np.linalg.norm(to_pt, axis=1)
        direction = to_pt / dist[:, None]
        facing = -(normals * direction).sum(1) > 0.12
        rays = np.concatenate([np.broadcast_to(cam, pts.shape), direction], axis=1).astype(np.float32)
        hit = rs.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        visible = facing & (hit > dist - 0.01)
        seen += visible
        closer = visible & (dist < best)
        best[closer] = dist[closer]
        chosen[closer] = k
    keep = seen >= min(noise.min_views, len(views))
    pts, chosen, best = pts[keep], chosen[keep], best[keep]

    # Depth error along the viewing ray, growing with range.
    ray = pts - views[chosen]
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    pts = pts + ray * (rng.normal(0, 1, len(pts)) * noise.depth_sigma_rel * best)[:, None]
    if noise.outlier_fraction > 0:
        bad = rng.random(len(pts)) < noise.outlier_fraction
        pts[bad] += ray[bad] * rng.normal(0, noise.outlier_sigma, (int(bad.sum()), 1))
    for _ in range(noise.hole_count):
        centre = pts[rng.integers(len(pts))]
        keep = np.linalg.norm(pts - centre, axis=1) > noise.hole_radius * rng.uniform(0.5, 1.5)
        pts, ray = pts[keep], ray[keep]

    cloud = PointCloud(pts, cameras=cams, view=-ray)
    if noise.gravity_error_deg or noise.scale_error:
        axis = rng.normal(size=3)
        axis[2] = 0
        axis /= np.linalg.norm(axis)
        ang = np.radians(noise.gravity_error_deg)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k
        cloud = cloud.transformed(rot, 1.0 + noise.scale_error)
    return cloud
