"""Render a simulated handheld phone video of a synthetic scene.

A small CPU ray tracer on top of Embree (through Open3D): procedural soil
colour, sun with cast shadows, sky light with ambient occlusion, mirror-like
water, then the camera's own flaws (motion blur, exposure drift, sensor noise,
H.264 compression). Output is an ordinary .mp4 plus sidecar files holding the
true camera poses and a simulated gravity sensor.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from brink.synth import noise
from brink.synth.capture import CameraPath, CaptureSpec, camera_path
from brink.synth.scene import Scene

_SOIL = {
    # base colour, tint of strata/patches (linear RGB)
    "damp_soil": ((0.30, 0.17, 0.10), (0.42, 0.26, 0.15)),
    "dry_sand": ((0.58, 0.47, 0.33), (0.66, 0.56, 0.42)),
    "clay": ((0.45, 0.22, 0.13), (0.56, 0.33, 0.20)),
    "weathered_rock": ((0.40, 0.36, 0.31), (0.52, 0.44, 0.34)),
    "hard_rock": ((0.36, 0.36, 0.37), (0.48, 0.47, 0.46)),
    "unknown": ((0.38, 0.26, 0.17), (0.48, 0.35, 0.24)),
}

_LIGHT = {
    # sun strength, sky strength, shadow softness (0 hard .. 1 no shadow), exposure, noise gain
    "overcast": (0.35, 0.80, 0.6, 1.0, 1.0),
    "sun": (1.00, 0.35, 0.0, 0.85, 1.0),
    "low": (0.20, 0.55, 0.5, 0.45, 3.0),
}


@dataclass
class RenderedCapture:
    video: Path
    truth: Path
    sensors: Path
    frames: int


class Renderer:
    def __init__(self, scene: Scene, capture: CaptureSpec):
        self.scene = scene
        self.capture = capture
        self.threads = int(os.environ.get("BRINK_RENDER_THREADS", "0"))  # 0: all cores
        self.grid = noise.lattice(scene.spec.seed + 101)
        self.grid2 = noise.lattice(scene.spec.seed + 202)
        self.rs = o3d.t.geometry.RaycastingScene()
        self.rs.add_triangles(
            o3d.core.Tensor(scene.vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(scene.triangles.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
        )
        self.vertex_normals = _vertex_normals(scene.vertices, scene.triangles)
        rng = np.random.default_rng(capture.seed + 7)
        elevation = np.radians(rng.uniform(28, 70))
        # Keep the sun on the open side so the face is not purely back-lit.
        azimuth = np.radians(rng.uniform(200, 340))
        self.sun = np.array(
            [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)],
            dtype=np.float32,
        )
        self.sun_k, self.sky_k, self.shadow_soft, self.exposure, self.noise_gain = _LIGHT[capture.lighting]
        w, h = capture.width, capture.height
        f = capture.focal_px
        u, v = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
        d = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], axis=-1)
        self.cam_dirs = (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)
        self.vertex_open = self._vertex_openness(rng)

    # -- tracing -----------------------------------------------------------

    def _cast(self, origins: NDArray, dirs: NDArray) -> dict[str, NDArray]:
        rays = o3d.core.Tensor(np.concatenate([origins, dirs], axis=1).astype(np.float32))
        hit = self.rs.cast_rays(rays, nthreads=self.threads)
        return {k: hit[k].numpy() for k in ("t_hit", "primitive_ids", "primitive_uvs")}

    def _occluded(self, origins: NDArray, dirs: NDArray, tmax: float) -> NDArray:
        t = self._cast(origins, dirs)["t_hit"]
        return t < tmax

    def _vertex_openness(self, rng: np.random.Generator, samples: int = 12) -> NDArray[np.float32]:
        """Share of nearby sky directions open at each vertex (ambient occlusion).

        Stored on the mesh so it is free of noise and identical in every frame,
        as real shading is.
        """
        v, n = self.scene.vertices, self.vertex_normals
        tangent, bitangent = _basis(n)
        origin = v + 0.004 * n
        open_count = np.zeros(len(v), dtype=np.float32)
        for _ in range(samples):
            r = rng.random((len(v), 2)).astype(np.float32)
            phi, sin_t, cos_t = 2 * np.pi * r[:, 0], np.sqrt(r[:, 1]), np.sqrt(1 - r[:, 1])
            d = (np.cos(phi) * sin_t)[:, None] * tangent + (np.sin(phi) * sin_t)[:, None] * bitangent + cos_t[:, None] * n
            open_count += ~self._occluded(origin, d.astype(np.float32), 1.5)
        return open_count / samples

    def _surface(self, hit: dict[str, NDArray], origins: NDArray, dirs: NDArray):
        valid = np.isfinite(hit["t_hit"])
        prim = hit["primitive_ids"][valid].astype(np.int64)
        tri = self.scene.triangles[prim]
        uv = hit["primitive_uvs"][valid]
        w0 = 1 - uv[:, 0] - uv[:, 1]
        n = (
            w0[:, None] * self.vertex_normals[tri[:, 0]]
            + uv[:, :1] * self.vertex_normals[tri[:, 1]]
            + uv[:, 1:] * self.vertex_normals[tri[:, 2]]
        )
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        pos = origins[valid] + dirs[valid] * hit["t_hit"][valid, None]
        openness = (
            w0 * self.vertex_open[tri[:, 0]] + uv[:, 0] * self.vertex_open[tri[:, 1]] + uv[:, 1] * self.vertex_open[tri[:, 2]]
        )
        return valid, pos.astype(np.float32), n.astype(np.float32), self.scene.materials[prim], openness

    # -- shading -----------------------------------------------------------

    def _albedo(self, p: NDArray) -> NDArray:
        base, tint = (np.array(c, dtype=np.float32) for c in _SOIL.get(self.scene.spec.material, _SOIL["unknown"]))
        big = self.scene.spec.kind == "face"
        # Layers: gently warped bands in rock and cut faces, blotches in piles.
        warp = noise.value_noise(p / np.float32(1.7), self.grid)
        band_coord = p[:, 2] * (3.0 if big else 1.2) + 1.5 * warp
        strata = 0.5 + 0.5 * np.sin(2 * np.pi * band_coord / (1.3 if big else 2.1))
        patch = 0.5 + 0.5 * noise.fbm(p, self.grid2, wavelength=0.45, octaves=3)
        mix = np.clip(0.55 * strata + 0.45 * patch, 0, 1)[:, None]
        colour = base * (1 - mix) + tint * mix
        # Grain, clods and stones, down to a few millimetres.
        grain = noise.fbm(p, self.grid, wavelength=0.04, octaves=3, gain=0.7)
        fine = noise.value_noise(p / np.float32(0.007), self.grid2)
        clods = noise.value_noise(p / np.float32(0.11), self.grid2)
        stones = noise.value_noise(p / np.float32(0.028), self.grid)
        lightness = (
            1.0 + 0.42 * grain + 0.22 * fine + 0.25 * clods
            + 0.9 * np.clip(stones - 0.45, 0, 1)  # pale stones
            - 0.6 * np.clip(-stones - 0.5, 0, 1)  # dark pits
        )
        return colour * np.clip(lightness, 0.15, 2.5)[:, None]

    def _sky(self, dirs: NDArray) -> NDArray:
        up = np.clip(dirs[:, 2], 0, 1)[:, None]
        if self.capture.lighting == "sun":
            horizon, zenith = np.array([0.75, 0.82, 0.90]), np.array([0.25, 0.45, 0.80])
        elif self.capture.lighting == "low":
            horizon, zenith = np.array([0.45, 0.42, 0.42]), np.array([0.22, 0.24, 0.30])
        else:
            horizon, zenith = np.array([0.80, 0.80, 0.80]), np.array([0.62, 0.64, 0.67])
        return (horizon * (1 - up) + zenith * up).astype(np.float32)

    def _shade(self, pos: NDArray, n: NDArray, openness: NDArray) -> NDArray:
        albedo = self._albedo(pos)
        origin = pos + 0.004 * n
        ndl = np.clip(n @ self.sun, 0, 1)
        lit = np.ones(len(pos), dtype=np.float32)
        facing = ndl > 0
        if facing.any():
            blocked = self._occluded(origin[facing], np.broadcast_to(self.sun, (int(facing.sum()), 3)), 50.0)
            lit[np.flatnonzero(facing)[blocked]] = self.shadow_soft
        sky = (0.5 + 0.5 * n[:, 2]) * (0.2 + 0.8 * openness)
        light = self.sun_k * ndl * lit + self.sky_k * sky
        return albedo * light[:, None]

    def frame(self, position: NDArray, rotation: NDArray) -> NDArray[np.float32]:
        """Linear RGB image, shape (H, W, 3), before camera effects."""
        w, h = self.capture.width, self.capture.height
        dirs = (self.cam_dirs @ rotation.T.astype(np.float32)).astype(np.float32)
        origins = np.broadcast_to(position.astype(np.float32), dirs.shape)
        image = self._sky(dirs)
        hit = self._cast(origins, dirs)
        valid, pos, n, material, openness = self._surface(hit, origins, dirs)
        index = np.flatnonzero(valid)
        ground = material == 0
        colour = np.zeros((len(pos), 3), dtype=np.float32)
        if ground.any():
            colour[ground] = self._shade(pos[ground], n[ground], openness[ground])
        water = ~ground
        if water.any():
            wp = pos[water]
            ripple = 0.02 * np.stack(
                [noise.value_noise(wp / np.float32(0.07), self.grid), noise.value_noise(wp / np.float32(0.07) + 9.1, self.grid)],
                axis=1,
            )
            wn = np.concatenate([ripple, np.ones((len(wp), 1), np.float32)], axis=1)
            wn /= np.linalg.norm(wn, axis=1, keepdims=True)
            d_in = dirs[index[water]]
            d_ref = d_in - 2 * (d_in * wn).sum(1, keepdims=True) * wn
            d_ref[:, 2] = np.abs(d_ref[:, 2])
            o_ref = wp + 0.004 * wn
            reflected = self._sky(d_ref)
            hit2 = self._cast(o_ref, d_ref.astype(np.float32))
            valid2, pos2, n2, _, open2 = self._surface(hit2, o_ref, d_ref)
            if valid2.any():
                reflected[valid2] = self._shade(pos2, n2, open2)
            fresnel = 0.25 + 0.75 * (1 - np.clip(-(d_in * wn).sum(1), 0, 1)) ** 3
            mud = np.array([0.16, 0.12, 0.08], dtype=np.float32) * (self.sun_k + self.sky_k)
            colour[water] = fresnel[:, None] * reflected + (1 - fresnel[:, None]) * mud
        image[index] = colour
        return image.reshape(h, w, 3)


def _vertex_normals(vertices: NDArray, triangles: NDArray) -> NDArray[np.float32]:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face_n = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices, dtype=np.float64)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_n)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals.astype(np.float32)


def _basis(n: NDArray) -> tuple[NDArray, NDArray]:
    helper = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]], np.float32), np.array([[1.0, 0.0, 0.0]], np.float32))
    tangent = np.cross(helper, n)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    return tangent.astype(np.float32), np.cross(n, tangent).astype(np.float32)


# --------------------------------------------------------------------------
# Camera effects and files
# --------------------------------------------------------------------------


def _pixel_motion(path: CameraPath, i: int, capture: CaptureSpec, depth: float) -> NDArray:
    """Image motion (pixels per second) at the centre of frame ``i``."""
    j = min(i + 1, len(path) - 1)
    k = max(i - 1, 0)
    if j == k:
        return np.zeros(2)
    point = path.positions[i] + depth * path.rotations[i][:, 2]

    def project(idx: int) -> NDArray:
        pc = path.rotations[idx].T @ (point - path.positions[idx])
        return capture.focal_px * pc[:2] / pc[2]

    return (project(j) - project(k)) / (path.times[j] - path.times[k])


def _motion_blur(image: NDArray, shift_px: NDArray) -> NDArray:
    length = float(np.linalg.norm(shift_px))
    if length < 0.7:
        return image
    size = int(np.ceil(length)) | 1
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = (size - 1) / 2
    direction = shift_px / length
    for step in np.linspace(-length / 2, length / 2, 2 * size):
        x, y = centre + step * direction[0], centre + step * direction[1]
        kernel[int(round(y)), int(round(x))] += 1
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def render_capture(scene: Scene, capture: CaptureSpec, out_dir: str | Path, name: str, crf: int = 23) -> RenderedCapture:
    """Render the whole walk to ``<out_dir>/<name>.mp4`` with truth and sensor sidecars."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{name}.mp4"
    path = camera_path(scene.spec, capture)
    renderer = Renderer(scene, capture)
    rng = np.random.default_rng(capture.seed + 13)
    w, h = capture.width, capture.height

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{capture.fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(video)],
        stdin=subprocess.PIPE,
    )
    exposure_s = capture.blur / 40.0 * (2.0 if capture.lighting == "low" else 1.0)
    typical_depth = capture.standoff + 0.5 * abs(scene.spec.crest_y)
    gain_wobble = 1 + 0.08 * np.sin(2 * np.pi * path.times / 9.0 + rng.uniform(0, 6.28))
    assert ffmpeg.stdin is not None
    for i in range(len(path)):
        linear = renderer.frame(path.positions[i], path.rotations[i]) * renderer.exposure * gain_wobble[i]
        linear = _motion_blur(linear, _pixel_motion(path, i, capture, typical_depth) * exposure_s)
        sigma = capture.noise * renderer.noise_gain
        linear = linear + rng.normal(0, 1, linear.shape).astype(np.float32) * sigma * np.sqrt(np.clip(linear, 0.02, 1))
        srgb = np.clip(linear, 0, 1) ** (1 / 2.2)
        ffmpeg.stdin.write((srgb * 255 + 0.5).astype(np.uint8).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the synthetic video")

    truth = out_dir / f"{name}.truth.json"
    truth.write_text(json.dumps({
        "scene": scene.ground_truth(),
        "capture": json.loads(capture.model_dump_json()),
        "intrinsics": {"fx": capture.focal_px, "fy": capture.focal_px, "cx": w / 2, "cy": h / 2, "width": w, "height": h},
        "frames": [
            {"t": float(path.times[i]), "position": path.positions[i].tolist(), "rotation": path.rotations[i].tolist()}
            for i in range(len(path))
        ],
    }))

    # Simulated gravity sensor: direction of "down" in camera axes, with a
    # fixed mounting bias and per-sample noise, as a phone accelerometer gives.
    bias = np.radians(rng.normal(0, 0.6, 3))
    samples = []
    for i in range(len(path)):
        down = path.rotations[i].T @ (-scene.spec.true_up)
        tilt = bias + np.radians(rng.normal(0, 0.8, 3))
        down = down + np.cross(tilt, down)
        samples.append({"t": float(path.times[i]), "gravity": (down / np.linalg.norm(down)).tolist()})
    sensors = out_dir / f"{name}.sensors.json"
    sensors.write_text(json.dumps({"format": "brink-sensors-1", "frame": "camera_opencv", "samples": samples}))
    return RenderedCapture(video, truth, sensors, len(path))
