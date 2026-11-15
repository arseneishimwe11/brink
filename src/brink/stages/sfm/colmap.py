"""Camera solving with COLMAP (through pycolmap, BSD-3-Clause), CPU only."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pycolmap

from brink.stages.sfm.base import SfMResult


class ColmapSolver:
    """Feature extraction, sequential matching and mapping.

    ``features``: "sift" (the only type the PyPI build of pycolmap can run).
    ``mapper``: "incremental" (classic COLMAP) or "global" (GLOMAP-style).
    """

    def __init__(self, name: str, features: str = "sift", mapper: str = "incremental", device: str = "cpu",
                 max_features: int = 4000, overlap: int = 8, threads: int = 4):
        self.name = name
        self.features = features
        self.mapper = mapper
        self.device = pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu
        self.max_features = max_features
        self.overlap = overlap
        self.threads = threads

    def solve(self, image_dir: Path, names: list[str], work_dir: Path) -> SfMResult | None:
        work_dir = Path(work_dir)
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        database = work_dir / "database.db"

        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = self.threads
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = self.threads
        if self.features != "sift":
            raise ValueError(f"unsupported feature type {self.features!r}")
        extraction.type = pycolmap.FeatureExtractorType.SIFT
        extraction.sift.max_num_features = self.max_features
        matching.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        pycolmap.extract_features(
            database, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader, extraction_options=extraction, device=self.device,
        )
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = self.overlap
        pairing.quadratic_overlap = True
        pairing.loop_detection = False
        pycolmap.match_sequential(database, matching_options=matching, pairing_options=pairing, device=self.device)

        sparse = work_dir / "sparse"
        sparse.mkdir()
        if self.mapper == "global":
            options = pycolmap.GlobalPipelineOptions()
            models = pycolmap.global_mapping(database, image_dir, sparse, options)
        else:
            options = pycolmap.IncrementalPipelineOptions()
            options.num_threads = self.threads
            options.min_model_size = 5
            options.multiple_models = False
            options.extract_colors = True
            # Fewer full bundle adjustments. Measured on a 55-frame capture:
            # 64 s -> 47 s with the same camera accuracy (see docs/EXPERIMENTS.md).
            options.ba_global_frames_ratio = 1.4
            options.ba_global_points_ratio = 1.4
            options.ba_global_max_num_iterations = 25
            options.ba_global_max_refinements = 2
            options.ba_local_max_num_iterations = 15
            models = pycolmap.incremental_mapping(database, image_dir, sparse, options)
        if not models:
            return None
        model = max(models.values(), key=lambda m: m.num_reg_images())
        return _to_result(model, len(names), self.name)


def _to_result(model: pycolmap.Reconstruction, frames_given: int, method: str) -> SfMResult | None:
    images = sorted((im for im in model.images.values() if im.has_pose), key=lambda im: im.name)
    if len(images) < 3:
        return None
    camera = model.cameras[images[0].camera_id]
    params = np.asarray(camera.params, dtype=float)
    model_name = camera.model.name
    if model_name == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        fx = fy = f
        dist = np.array([k1, 0.0, 0.0, 0.0])
    elif model_name == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
        dist = np.zeros(4)
    elif model_name == "PINHOLE":
        fx, fy, cx, cy = params
        dist = np.zeros(4)
    elif model_name in ("RADIAL", "OPENCV"):
        if model_name == "RADIAL":
            f, cx, cy, k1, k2 = params
            fx = fy = f
            dist = np.array([k1, k2, 0.0, 0.0])
        else:
            fx, fy, cx, cy, k1, k2, p1, p2 = params
            dist = np.array([k1, k2, p1, p2])
    else:
        raise ValueError(f"unsupported camera model {model_name}")

    point_ids = sorted(model.points3D.keys())
    row = {pid: i for i, pid in enumerate(point_ids)}
    points = np.array([model.points3D[pid].xyz for pid in point_ids], dtype=float).reshape(-1, 3)
    colors = np.array([model.points3D[pid].color for pid in point_ids], dtype=np.uint8).reshape(-1, 3)
    rotations, positions, visible, names = [], [], [], []
    for im in images:
        cam_from_world = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        r_cw = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
        t_cw = np.asarray(cam_from_world.translation, dtype=float)
        rotations.append(r_cw.T)
        positions.append(-r_cw.T @ t_cw)
        ids = [p.point3D_id for p in im.points2D if p.has_point3D()]
        visible.append(np.array([row[i] for i in ids if i in row], dtype=np.int64))
        names.append(im.name)
    return SfMResult(
        names, np.array(rotations), np.array(positions), float(fx), float(fy), float(cx), float(cy), dist,
        int(camera.width), int(camera.height), points, colors, visible, frames_given,
        float(model.compute_mean_reprojection_error()), method,
    )
