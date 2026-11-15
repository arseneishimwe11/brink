"""Dense reconstruction by pairwise stereo on the CPU.

Each frame is paired with a nearby frame, both are rectified so matching
points share an image row, and OpenCV's semi-global matcher (Apache-2.0)
gives a depth map. Depth maps are then cross-checked between pairs.

This is the CPU stand-in for GPU multi-view stereo; ``make_dense`` is the
switch point for a GPU method later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from brink.geometry import PointCloud
from brink.stages.dense.fusion import DepthView, fuse
from brink.stages.masking import mask_name
from brink.stages.sfm.base import SfMResult


def _textured(gray: NDArray) -> NDArray[np.bool_]:
    """True where the image has enough local contrast to be matched reliably."""
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    mean = cv2.blur(smooth, (15, 15))
    var = cv2.blur(smooth * smooth, (15, 15)) - mean * mean
    return var > 1.5**2


class StereoDense:
    name = "stereo"

    def __init__(self, baseline_ratio: float = 0.10, max_disparities: int = 192, block: int = 5,
                 search: int = 8, tolerance: float = 0.012, min_agree: int = 2, fringe: int = 11):
        self.baseline_ratio = baseline_ratio  # wanted baseline as a share of scene depth
        self.max_disparities = max_disparities
        self.block = block
        self.search = search
        self.tolerance = tolerance
        self.min_agree = min_agree
        self.fringe = fringe  # pixels trimmed around every matched region

    # -- pairing -----------------------------------------------------------

    def _partner(self, sfm: SfMResult, i: int, depth: float, side: int) -> int | None:
        """Best partner for frame ``i`` among later (side=+1) or earlier (side=-1) frames."""
        want = self.baseline_ratio * depth
        best, best_cost = None, np.inf
        axis_i = sfm.rotations[i][:, 2]
        for step in range(1, self.search + 1):
            j = i + side * step
            if j < 0 or j >= sfm.registered:
                break
            base = sfm.positions[j] - sfm.positions[i]
            length = np.linalg.norm(base)
            if length < 0.25 * want or length > 3.0 * want:
                continue
            axis_j = sfm.rotations[j][:, 2]
            if axis_i @ axis_j < np.cos(np.radians(25)):
                continue
            mean_axis = axis_i + axis_j
            mean_axis /= np.linalg.norm(mean_axis)
            if abs(mean_axis @ base) / length > np.cos(np.radians(35)):  # moving along the view
                continue
            cost = abs(np.log(length / want))
            if cost < best_cost:
                best, best_cost = j, cost
        return best

    # -- one pair ----------------------------------------------------------

    def _pair_depth(self, sfm: SfMResult, images: dict[int, NDArray], masks: dict[int, NDArray | None],
                    a: int, b: int, z_near: float, z_far: float) -> DepthView | None:
        ca, cb = sfm.positions[a], sfm.positions[b]
        base = cb - ca
        length = float(np.linalg.norm(base))
        x = base / length
        z_mean = sfm.rotations[a][:, 2] + sfm.rotations[b][:, 2]
        y = np.cross(z_mean, x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        r_rect = np.stack([x, y, z])  # world -> rectified camera

        K, f = sfm.K, 0.5 * (sfm.fx + sfm.fy)
        w, h = sfm.width, sfm.height
        corners = cv2.undistortPoints(
            np.array([[0, 0], [w, 0], [w, h], [0, h], [w / 2, 0], [w / 2, h], [0, h / 2], [w, h / 2]], np.float64).reshape(-1, 1, 2),
            K, sfm.distortion,
        ).reshape(-1, 2)
        rays = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        lo, hi = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])
        rel = {}
        for idx in (a, b):
            rel[idx] = r_rect @ sfm.rotations[idx]  # original camera -> rectified
            d = rays @ rel[idx].T
            if (d[:, 2] <= 0.05).any():
                return None
            p = d[:, :2] / d[:, 2:]
            lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
        size = np.ceil((hi - lo) * f).astype(int)
        if size[0] > 1.6 * max(w, h) or size[1] > 1.6 * max(w, h) or size.min() < 32:
            return None
        k_new = np.array([[f, 0, -lo[0] * f], [0, f, -lo[1] * f], [0, 0, 1.0]])
        out_size = (int(size[0]), int(size[1]))

        rect, valid = {}, {}
        for idx in (a, b):
            m1, m2 = cv2.initUndistortRectifyMap(K, sfm.distortion, rel[idx], k_new, out_size, cv2.CV_32FC1)
            rect[idx] = cv2.remap(images[idx], m1, m2, cv2.INTER_LINEAR)
            ones = np.full((h, w), 255, np.uint8) if masks[idx] is None else masks[idx]
            valid[idx] = cv2.remap(ones, m1, m2, cv2.INTER_NEAREST) > 127

        d_min = f * length / z_far
        d_max = f * length / z_near
        min_disp = int(np.floor(d_min)) - 2
        num = int(np.ceil((d_max - min_disp + 2) / 16.0)) * 16
        if num > self.max_disparities:
            return None
        num = max(num, 16)
        sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp, numDisparities=num, blockSize=self.block,
            P1=8 * self.block**2, P2=32 * self.block**2, disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=120, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        ga = cv2.cvtColor(rect[a], cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(rect[b], cv2.COLOR_RGB2GRAY)
        # Pad on the left so the matcher's blind band falls outside the image.
        pad = min_disp + num
        ga_p = cv2.copyMakeBorder(ga, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        gb_p = cv2.copyMakeBorder(gb, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0)
        disp = sgbm.compute(ga_p, gb_p)[:, pad:].astype(np.float32) / 16.0
        ok = (disp > max(min_disp, 0.5)) & valid[a] & _textured(ga)
        # The matcher bleeds a surface a few pixels past its outline into
        # whatever has no texture behind it (sky above a crest). Shrinking the
        # matched area removes that fringe, which would otherwise look like
        # ground hanging over the edge.
        ok = cv2.erode(ok.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.fringe, self.fringe))) > 0
        ok &= cv2.erode(valid[a].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # Pixels straddling a jump in depth are averages of two surfaces.
        hi_d = cv2.dilate(disp, np.ones((3, 3), np.uint8))
        lo_d = cv2.erode(np.where(disp > min_disp, disp, np.float32(1e6)), np.ones((3, 3), np.uint8))
        ok &= (hi_d - lo_d) <= np.maximum(1.5, 0.03 * disp)
        depth = np.zeros(disp.shape, np.float32)
        depth[ok] = f * length / disp[ok]
        return DepthView(depth, rect[a], k_new, r_rect, ca.copy(), group=a)

    # -- all pairs ---------------------------------------------------------

    def reconstruct(self, sfm: SfMResult, image_dir: Path, masks_dir: Path | None = None) -> PointCloud:
        images: dict[int, NDArray] = {}
        masks: dict[int, NDArray | None] = {}
        for i, name in enumerate(sfm.names):
            bgr = cv2.imread(str(Path(image_dir) / name))
            images[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask_path = None if masks_dir is None else Path(masks_dir) / mask_name(name)
            masks[i] = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.exists() else None

        views: list[DepthView] = []
        for i in range(sfm.registered):
            seen = sfm.points[sfm.visible[i]] if len(sfm.visible[i]) else np.zeros((0, 3))
            if len(seen) < 15:
                continue
            z = (seen - sfm.positions[i]) @ sfm.rotations[i][:, 2]
            z = z[z > 0]
            if len(z) < 15:
                continue
            z_near, z_med, z_far = np.percentile(z, [1, 50, 99])
            z_near, z_far = 0.7 * z_near, 1.4 * z_far
            for side in (1, -1):
                j = self._partner(sfm, i, z_med, side)
                if j is None:
                    continue
                view = self._pair_depth(sfm, images, masks, i, j, z_near, z_far)
                if view is not None:
                    views.append(view)
        if not views:
            return PointCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), cameras=sfm.positions)
        return fuse(views, sfm.positions, min_agree=self.min_agree, tolerance=self.tolerance)
