"""Synthetic pit faces and soil piles with exactly known geometry.

Frame: +X runs along the face (strike), +Y points into the ground behind the
face, +Z is up. The toe of the test face is the line y = 0, z = 0, and the open
side where a person walks is y < 0.

The ground is described by a signed distance function (negative inside solid
ground) and meshed with marching cubes. Because every hazard is carved by a
parameter, its true size is known without measuring the mesh.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from skimage import measure

from brink.synth import noise


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UndercutSpec(_Spec):
    center_x: float
    width: float
    depth: float  # horizontal reach behind the face surface at the notch top
    height: float  # height of the notch opening above the toe


class CrackSpec(_Spec):
    center_x: float
    length: float
    width: float
    depth: float
    offset: float  # distance behind the crest edge


class PuddleSpec(_Spec):
    center_x: float
    radius_x: float
    radius_y: float


class SceneSpec(_Spec):
    """Everything needed to rebuild a scene exactly."""

    name: str
    seed: int
    kind: str = "face"  # "face": bench continues behind the crest; "pile": free-standing mound
    material: str = "damp_soil"
    height: float = 4.0
    slope_deg: float = 60.0
    length: float = 8.0
    top_width: float = 1.0  # pile only: width of the flat top
    back_slope_deg: float = 38.0  # pile only: angle of the back and end slopes
    roughness: float = 0.02  # amplitude of surface bumps, metres
    voxel: float = 0.025
    undercut: UndercutSpec | None = None
    crack: CrackSpec | None = None
    puddle: PuddleSpec | None = None

    @property
    def crest_y(self) -> float:
        return self.height / np.tan(np.radians(self.slope_deg))

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


@dataclass
class Scene:
    spec: SceneSpec
    vertices: NDArray[np.float32]
    triangles: NDArray[np.int32]
    # Per-triangle material: 0 ground, 1 water.
    materials: NDArray[np.uint8]

    def ground_truth(self) -> dict:
        return ground_truth(self.spec)


# --------------------------------------------------------------------------
# Signed distance function
# --------------------------------------------------------------------------

_END_SLOPE_DEG = 40.0
_PUDDLE_DEPTH = 0.07
_WATER_LEVEL = 0.4  # fraction of the depression depth left above the water


_FLOOR_BUMP = 0.03


def _ground_height(x: NDArray, y: NDArray, spec: SceneSpec, grid: NDArray, detail: bool = True) -> NDArray:
    """Height of the floor in front of the face: gentle undulation plus puddle hollow."""
    if detail:
        pts = np.stack([x, y, np.zeros_like(x)], axis=-1)
        g = _FLOOR_BUMP * noise.fbm(pts, grid, wavelength=2.5, octaves=3)
    else:
        g = np.zeros_like(x, dtype=np.float32)
    if spec.puddle is not None:
        p = spec.puddle
        cy = -0.75 * p.radius_y
        r2 = ((x - p.center_x) / p.radius_x) ** 2 + ((y - cy) / p.radius_y) ** 2
        g = g * (1 - np.exp(-r2)) - _PUDDLE_DEPTH * np.exp(-r2)
    return g.astype(np.float32)


def _box(px: NDArray, py: NDArray, pz: NDArray, half: tuple[float, float, float], radius: float) -> NDArray:
    """Signed distance to an axis-aligned rounded box centred at the origin."""
    qx = np.abs(px) - (half[0] - radius)
    qy = np.abs(py) - (half[1] - radius)
    qz = np.abs(pz) - (half[2] - radius)
    outside = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return outside + inside - radius


def sdf(
    points: NDArray, spec: SceneSpec, grid: NDArray | None = None, detail: bool = True
) -> NDArray[np.float32]:
    """Signed distance (approximate, exact in sign) at ``points`` (..., 3).

    With ``detail=False`` the noise terms are skipped. That version is cheap and
    never differs from the full one by more than ``detail_band(spec)``.
    """
    if grid is None:
        grid = noise.lattice(spec.seed)
    p = np.asarray(points, dtype=np.float32)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    s = np.radians(spec.slope_deg)
    h = spec.height

    d_face = -np.sin(s) * y + np.cos(s) * z
    d_top = z - h
    a = np.radians(_END_SLOPE_DEG)
    d_end = np.cos(a) * (z - h) + np.sin(a) * (np.abs(x) - spec.length / 2)
    wall = np.maximum(np.maximum(d_face, d_top), d_end)
    if spec.kind == "pile":
        b = np.radians(spec.back_slope_deg)
        back_toe = spec.crest_y + spec.top_width + h / np.tan(b)
        d_back = np.sin(b) * (y - back_toe) + np.cos(b) * z
        wall = np.maximum(wall, d_back)

    # Surface roughness is applied before carving, so carved features keep
    # their exact dimensions.
    if detail:
        wall = wall + spec.roughness * noise.fbm(p, grid, wavelength=12 * spec.roughness + 0.15, octaves=4)

    if spec.undercut is not None:
        u = spec.undercut
        lip_y = u.height / np.tan(s)
        back_y = lip_y + u.depth
        front_y = min(0.0, lip_y) - 0.5
        half = (u.width / 2, (back_y - front_y) / 2, u.height / 2 + 0.25)
        radius = 0.3 * min(u.height, u.depth, u.width)
        notch = _box(x - u.center_x, y - (back_y + front_y) / 2, z - (u.height / 2 - 0.25), half, radius)
        wall = np.maximum(wall, -notch)

    if spec.crack is not None:
        c = spec.crack
        wander = 0.25 * c.offset * np.sin(2 * np.pi * (x - c.center_x) / (0.6 * c.length) + spec.seed)
        cy = spec.crest_y + c.offset + wander
        # Taper the slot to nothing at both ends, as real cracks do.
        taper = np.clip(1 - (np.abs(x - c.center_x) / (c.length / 2)) ** 4, 0, 1)
        half_w = np.maximum(c.width / 2 * taper, 1e-4)
        slot = np.maximum(
            np.maximum(np.abs(y - cy) - half_w, (h - c.depth * taper) - z),
            np.abs(x - c.center_x) - c.length / 2,
        )
        wall = np.maximum(wall, -slot)

    ground = z - _ground_height(x, y, spec, grid, detail)
    return np.minimum(wall, ground).astype(np.float32)


def detail_band(spec: SceneSpec) -> float:
    return spec.roughness + _FLOOR_BUMP


# --------------------------------------------------------------------------
# Meshing
# --------------------------------------------------------------------------


def _volume_bounds(spec: SceneSpec) -> tuple[NDArray, NDArray]:
    end_run = spec.height / np.tan(np.radians(_END_SLOPE_DEG))
    lean = min(0.0, spec.crest_y)  # negative when the face overhangs
    lo = np.array([-spec.length / 2 - end_run - 0.3, lean - 0.8, -0.2])
    if spec.kind == "pile":
        back = spec.crest_y + spec.top_width + spec.height / np.tan(np.radians(spec.back_slope_deg))
        hi_y = back + 0.4
    else:
        hi_y = max(spec.crest_y, 0.0) + 3.0
    if spec.puddle is not None:
        lo[1] = min(lo[1], -2.2 * spec.puddle.radius_y - 0.3)
    hi = np.array([spec.length / 2 + end_run + 0.3, hi_y, spec.height + 0.3])
    return lo, hi


def _marching_cubes(spec: SceneSpec, grid: NDArray) -> tuple[NDArray, NDArray]:
    lo, hi = _volume_bounds(spec)
    v = spec.voxel
    nx, ny, nz = (np.ceil((hi - lo) / v).astype(int) + 1).tolist()
    xs = (lo[0] + v * np.arange(nx)).astype(np.float32)
    ys = (lo[1] + v * np.arange(ny)).astype(np.float32)
    zs = (lo[2] + v * np.arange(nz)).astype(np.float32)
    volume = np.empty((nx, ny, nz), dtype=np.float32)
    step = 12
    for k in range(0, nz, step):
        gx, gy, gz = np.meshgrid(xs, ys, zs[k : k + step], indexing="ij")
        pts = np.stack([gx, gy, gz], axis=-1)
        # Noise only matters near the surface, so evaluate it in a thin shell.
        slab = sdf(pts, spec, grid, detail=False)
        near = np.abs(slab) < detail_band(spec) + 2 * v
        slab[near] = sdf(pts[near], spec, grid)
        volume[:, :, k : k + step] = slab
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(v, v, v), allow_degenerate=False)
    verts = verts + lo.astype(np.float32)
    return verts.astype(np.float32), faces.astype(np.int32)


def _floor_mesh(spec: SceneSpec, grid: NDArray, margin: float) -> tuple[NDArray, NDArray]:
    """Coarse floor outside the marching-cubes box, out to where people walk."""
    lo, hi = _volume_bounds(spec)
    cell = max(4 * spec.voxel, 0.08)
    xs = np.arange(lo[0] - margin, hi[0] + margin + cell, cell, dtype=np.float32)
    ys = np.arange(lo[1] - margin, hi[1] + margin + cell, cell, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gz = _ground_height(gx, gy, spec, grid)
    verts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    quads = np.stack([a, b, c, d], axis=-1).reshape(-1, 4)
    # Drop cells that lie wholly inside the detailed volume.
    cx = (gx[:-1, :-1] + gx[1:, 1:]).ravel() / 2
    cy = (gy[:-1, :-1] + gy[1:, 1:]).ravel() / 2
    inner = (cx > lo[0] + cell) & (cx < hi[0] - cell) & (cy > lo[1] + cell) & (cy < hi[1] - cell)
    quads = quads[~inner]
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return verts.astype(np.float32), tris.astype(np.int32)


def _water_mesh(spec: SceneSpec) -> tuple[NDArray, NDArray] | None:
    if spec.puddle is None:
        return None
    p = spec.puddle
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cy = -0.75 * p.radius_y
    z = -_PUDDLE_DEPTH * _WATER_LEVEL
    ring = np.stack([p.center_x + 1.05 * p.radius_x * np.cos(ang), cy + 1.05 * p.radius_y * np.sin(ang), np.full_like(ang, z)], axis=-1)
    verts = np.concatenate([[[p.center_x, cy, z]], ring]).astype(np.float32)
    k = np.arange(len(ang))
    tris = np.stack([np.zeros_like(k), 1 + k, 1 + (k + 1) % len(ang)], axis=-1).astype(np.int32)
    return verts, tris


def _orient_outward(verts: NDArray, tris: NDArray, spec: SceneSpec, grid: NDArray) -> NDArray:
    """Flip winding if needed so triangle normals point out of the ground."""
    sample = tris[:: max(1, len(tris) // 2000)]
    a, b, c = verts[sample[:, 0]], verts[sample[:, 1]], verts[sample[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    centre = (a + b + c) / 3
    outward = sdf(centre + 0.5 * spec.voxel * n, spec, grid) > sdf(centre - 0.5 * spec.voxel * n, spec, grid)
    return tris if outward.mean() > 0.5 else tris[:, ::-1].copy()


def build_scene(spec: SceneSpec, floor_margin: float = 10.0, cache_dir: str | Path | None = None) -> Scene:
    """Build (or load from cache) the triangle mesh for ``spec``."""
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"scene_{spec.digest()}_{floor_margin:g}.npz"
        if cache.exists():
            with np.load(cache) as z:
                return Scene(spec, z["vertices"], z["triangles"], z["materials"])

    grid = noise.lattice(spec.seed)
    verts, tris = _marching_cubes(spec, grid)
    tris = _orient_outward(verts, tris, spec, grid)
    fv, ft = _floor_mesh(spec, grid, floor_margin)
    parts_v, parts_t, mats = [verts, fv], [tris, ft + len(verts)], [np.zeros(len(tris) + len(ft), np.uint8)]
    water = _water_mesh(spec)
    if water is not None:
        wv, wt = water
        parts_t.append(wt + len(verts) + len(fv))
        parts_v.append(wv)
        mats.append(np.ones(len(wt), np.uint8))
    scene = Scene(spec, np.concatenate(parts_v), np.concatenate(parts_t), np.concatenate(mats))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, vertices=scene.vertices, triangles=scene.triangles, materials=scene.materials)
    return scene


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def ground_truth(spec: SceneSpec) -> dict:
    """True faces and hazards for scoring, in scene coordinates (metres)."""
    faces = [
        {
            "id": "main",
            "slope_deg": spec.slope_deg,
            "height": spec.height,
            "length": spec.length,
            "outward": [0.0, -1.0, 0.0],
            "centroid": [0.0, spec.crest_y / 2, spec.height / 2],
        }
    ]
    hazards: list[dict] = []
    if spec.undercut is not None:
        u = spec.undercut
        lip_y = u.height / np.tan(np.radians(spec.slope_deg))
        hazards.append(
            {
                "type": "undercut",
                "face_id": "main",
                "location": [u.center_x, lip_y + u.depth / 2, u.height / 2],
                "extent_along_face": u.width,
                "depth": u.depth,
                "height": u.height,
            }
        )
    if spec.slope_deg > 90.5:
        hazards.append(
            {
                "type": "overhang",
                "face_id": "main",
                "location": [0.0, spec.crest_y / 2, spec.height / 2],
                "extent_along_face": spec.length,
                "depth": -spec.crest_y,
            }
        )
    if spec.crack is not None:
        c = spec.crack
        hazards.append(
            {
                "type": "crest_crack",
                "face_id": "main",
                "location": [c.center_x, spec.crest_y + c.offset, spec.height],
                "extent_along_face": c.length,
                "length": c.length,
                "width": c.width,
                "depth": c.depth,
            }
        )
    if spec.puddle is not None:
        p = spec.puddle
        hazards.append(
            {
                "type": "water_at_toe",
                "face_id": "main",
                "location": [p.center_x, -0.75 * p.radius_y, 0.0],
                "extent_along_face": 2 * p.radius_x,
                "radius_x": p.radius_x,
                "radius_y": p.radius_y,
            }
        )
    return {"spec": json.loads(spec.model_dump_json()), "faces": faces, "hazards": hazards}
