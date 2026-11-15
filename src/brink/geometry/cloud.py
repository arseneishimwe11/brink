"""Point cloud container shared by reconstruction, hazard geometry and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]


@dataclass
class PointCloud:
    """Points with optional colour and normals, plus the cameras that saw them.

    ``cameras`` holds camera centres in the same frame as the points. Hazard
    geometry uses them to reason about what was visible.

    ``view`` holds, per point, a unit vector toward a camera that actually saw
    it. Normals are oriented with it: the nearest camera is not good enough,
    because on a walk around a pile it is often one behind the surface.
    """

    xyz: F32
    rgb: NDArray[np.uint8] | None = None
    normals: F32 | None = None
    cameras: F32 = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    view: F32 | None = None

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        if self.rgb is not None:
            self.rgb = np.ascontiguousarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        if self.normals is not None:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32).reshape(-1, 3)
        self.cameras = np.ascontiguousarray(self.cameras, dtype=np.float32).reshape(-1, 3)
        if self.view is not None:
            self.view = np.ascontiguousarray(self.view, dtype=np.float32).reshape(-1, 3)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def select(self, mask_or_index: NDArray) -> PointCloud:
        return PointCloud(
            self.xyz[mask_or_index],
            None if self.rgb is None else self.rgb[mask_or_index],
            None if self.normals is None else self.normals[mask_or_index],
            self.cameras,
            None if self.view is None else self.view[mask_or_index],
        )

    def transformed(self, rotation: NDArray, scale: float = 1.0, shift: NDArray | None = None) -> PointCloud:
        """Return ``scale * R @ p + shift`` applied to points and cameras."""
        r = np.asarray(rotation, dtype=np.float64)
        t = np.zeros(3) if shift is None else np.asarray(shift, dtype=np.float64)
        xyz = (self.xyz.astype(np.float64) @ r.T) * scale + t
        cams = (self.cameras.astype(np.float64) @ r.T) * scale + t
        normals = None if self.normals is None else self.normals.astype(np.float64) @ r.T
        view = None if self.view is None else self.view.astype(np.float64) @ r.T
        return PointCloud(xyz, self.rgb, normals, cams, view)

    def thinned(self, voxel: float) -> PointCloud:
        """One averaged point per ``voxel`` cube. Colours and view vectors are averaged too."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz.astype(np.float64))
        if self.rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.rgb.astype(np.float64) / 255.0)
        if self.view is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.view.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel)
        rgb = None if self.rgb is None else (np.asarray(pcd.colors) * 255).astype(np.uint8)
        view = None
        if self.view is not None:
            view = np.asarray(pcd.normals)
            view = view / (np.linalg.norm(view, axis=1, keepdims=True) + 1e-12)
        return PointCloud(np.asarray(pcd.points), rgb, None, self.cameras, view)

    def without_floaters(self, cell: float, min_share: float = 0.003, min_points: int = 300) -> PointCloud:
        """Drop small clumps of points that are not attached to the main surfaces.

        Stereo matching leaves stray clumps in mid-air, mostly where ground
        meets sky. Points are binned into ``cell``-sized cubes; cubes that touch
        form a clump, and clumps holding too few points are removed.
        """
        from scipy import ndimage

        if len(self) == 0:
            return self
        lo = self.xyz.min(0)
        span = self.xyz.max(0) - lo
        cell = max(cell, float(span.max()) / 500.0)  # keep the grid a sane size
        idx = np.floor((self.xyz - lo) / cell).astype(np.int64)
        shape = idx.max(0) + 1
        occupied = np.zeros(shape, dtype=bool)
        occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        labels, count = ndimage.label(occupied, structure=np.ones((3, 3, 3), dtype=bool))
        point_label = labels[idx[:, 0], idx[:, 1], idx[:, 2]]
        sizes = np.bincount(point_label, minlength=count + 1)
        keep = sizes[point_label] >= max(min_points, min_share * len(self))
        return self.select(keep)

    # -- persistence -------------------------------------------------------

    def save_npz(self, path: str | Path) -> None:
        arrays: dict[str, NDArray] = {"xyz": self.xyz, "cameras": self.cameras}
        if self.rgb is not None:
            arrays["rgb"] = self.rgb
        if self.normals is not None:
            arrays["normals"] = self.normals
        if self.view is not None:
            arrays["view"] = self.view
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> PointCloud:
        with np.load(path) as z:
            return cls(
                z["xyz"],
                z["rgb"] if "rgb" in z else None,
                z["normals"] if "normals" in z else None,
                z["cameras"] if "cameras" in z else np.zeros((0, 3), np.float32),
                z["view"] if "view" in z else None,
            )

    def save_ply(self, path: str | Path, rgb: NDArray[np.uint8] | None = None) -> None:
        """Write a binary little-endian PLY. ``rgb`` overrides stored colours."""
        colours = rgb if rgb is not None else self.rgb
        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if self.normals is not None:
            fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
        if colours is not None:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        data = np.empty(len(self), dtype=fields)
        data["x"], data["y"], data["z"] = self.xyz.T
        if self.normals is not None:
            data["nx"], data["ny"], data["nz"] = self.normals.T
        if colours is not None:
            data["red"], data["green"], data["blue"] = np.asarray(colours, np.uint8).T
        names = {"<f4": "float", "u1": "uchar"}
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(self)}"]
        header += [f"property {names[t]} {n}" for n, t in fields]
        header.append("end_header")
        with open(path, "wb") as fh:
            fh.write(("\n".join(header) + "\n").encode("ascii"))
            fh.write(data.tobytes())
