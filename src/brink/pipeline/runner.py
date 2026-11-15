"""Run the whole chain on one video: frames, cameras, dense cloud, up, scale, hazards.

Each expensive stage writes its result under the work directory together with
a key built from its inputs and settings. A rerun with the same key reuses the
stored result, so changing a late stage does not repeat the early ones.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from brink.domain import Capture, Reconstruction, Report, ScaleEstimate, Site
from brink.geometry import PointCloud
from brink.hazards.engine import GeometryResult, add_image_hazards, analyze_geometry
from brink.hazards.imagery import FrameSet
from brink.hazards.params import HazardParams
from brink.rules import Rules, load_rules
from brink.stages import orient
from brink.stages.dense import make_dense
from brink.stages.frames import FrameSelectionParams, SelectedFrames, load_selected, select_frames
from brink.stages.sfm import SfMResult, make_solver

LIMITATIONS = [
    "This is an automated screening aid, not an engineering assessment or certification.",
    "It reports hazards it could measure. Ground it did not flag has not been cleared.",
    "Hazard thresholds are provisional and have not been approved by an engineer.",
]


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frames: FrameSelectionParams = Field(default_factory=FrameSelectionParams)
    sfm: str = "colmap_sift"
    dense: str = "stereo"
    gravity: str = "auto"  # "auto": sensor file if present, else scene; or force "sensor" / "scene"
    scale: str = "camera_height"  # or "none"
    material: str = "unknown"
    hazards: HazardParams = Field(default_factory=HazardParams)
    device: str = "cpu"  # the switch for GPU stages later
    min_registered_share: float = 0.6
    min_registered_frames: int = 12


@dataclass
class PipelineResult:
    report: Report
    work_dir: Path
    geometry: GeometryResult | None = None
    sfm: SfMResult | None = None
    frames: SelectedFrames | None = None
    # Analysis coordinates = scale * rotation @ SfM coordinates.
    rotation: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _fresh(marker: Path, key: str) -> bool:
    return marker.exists() and marker.read_text().strip() == key


def _probe(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or None,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def run_pipeline(
    video: str | Path,
    work_dir: str | Path,
    config: PipelineConfig | None = None,
    rules: Rules | None = None,
    site: Site | None = None,
    sensors: str | Path | None = None,
    captured_at: datetime | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    rules = rules or load_rules()
    video, work_dir = Path(video), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    site = site or Site(id="unknown", material=config.material)

    sha = _sha256(video)
    info = _probe(video)
    if sensors is None:
        candidate = video.with_suffix(".sensors.json")
        sensors = candidate if candidate.exists() else None
    capture = Capture(
        id=video.stem, site_id=site.id, video_path=str(video), video_sha256=sha, captured_at=captured_at,
        duration_s=(info["frames"] / info["fps"]) if info["fps"] else None,
        width=info["width"], height=info["height"], fps=info["fps"],
        sensor_sidecar=str(sensors) if sensors else None,
    )
    recon = Reconstruction(capture_id=capture.id, methods={"sfm": config.sfm, "dense": config.dense})

    def finish(geometry: GeometryResult | None = None, **extra) -> PipelineResult:
        recon.timings_s = {k: round(v, 2) for k, v in timings.items()}
        report = Report(
            generated_at=datetime.now(UTC), site=site, capture=capture, reconstruction=recon,
            faces=geometry.faces if geometry else [], hazards=geometry.hazards if geometry else [],
            rules_version=rules.version, rules_provisional=rules.provisional, limitations=list(LIMITATIONS),
        )
        return PipelineResult(report, work_dir, geometry, **extra)

    # -- frames ------------------------------------------------------------
    start = time.perf_counter()
    frames_dir = work_dir / "frames"
    frames_key = _key(sha, config.frames.model_dump_json())
    if _fresh(frames_dir / "key", frames_key):
        frames = load_selected(frames_dir)
    else:
        frames = select_frames(video, frames_dir, config.frames)
        (frames_dir / "key").write_text(frames_key)
    timings["frames"] = time.perf_counter() - start
    recon.frames_selected = len(frames.names)

    # -- cameras -----------------------------------------------------------
    start = time.perf_counter()
    sfm_key = _key(frames_key, config.sfm)
    sfm_file = work_dir / f"sfm_{config.sfm}.npz"
    if _fresh(sfm_file.with_suffix(".key"), sfm_key):
        sfm = SfMResult.load(sfm_file) if sfm_file.exists() else None
    else:
        sfm = make_solver(config.sfm, config.device).solve(frames_dir, frames.names, work_dir / f"sfm_{config.sfm}")
        if sfm is not None:
            sfm.save(sfm_file)
        elif sfm_file.exists():
            sfm_file.unlink()
        sfm_file.with_suffix(".key").write_text(sfm_key)
    timings["sfm"] = time.perf_counter() - start
    if sfm is None:
        recon.failure_reason = "Camera positions could not be solved. Film again, slower, keeping the face in view."
        return finish(frames=frames)
    recon.frames_registered = sfm.registered
    recon.mean_reprojection_error_px = sfm.mean_reprojection_error
    if sfm.registered < max(config.min_registered_frames, config.min_registered_share * len(frames.names)):
        recon.failure_reason = (
            f"Only {sfm.registered} of {len(frames.names)} frames could be placed. "
            "Film again, slower, with more overlap between views."
        )
        return finish(sfm=sfm, frames=frames)

    # -- dense cloud -------------------------------------------------------
    start = time.perf_counter()
    dense_key = _key(sfm_key, config.dense)
    dense_file = work_dir / f"dense_{config.dense}.npz"
    if _fresh(dense_file.with_suffix(".key"), dense_key) and dense_file.exists():
        cloud = PointCloud.load_npz(dense_file)
    else:
        cloud = make_dense(config.dense, config.device).reconstruct(sfm, frames_dir)
        if len(cloud):
            ranges = np.linalg.norm(sfm.points - sfm.positions.mean(0), axis=1)
            voxel = 0.003 * float(np.median(ranges))
            cloud = cloud.thinned(voxel).without_floaters(4 * voxel)
        cloud.save_npz(dense_file)
        dense_file.with_suffix(".key").write_text(dense_key)
    timings["dense"] = time.perf_counter() - start
    recon.points = len(cloud)
    if len(cloud) < 5000:
        recon.failure_reason = "Too little of the ground could be reconstructed. Film again in better light, closer to the face."
        return finish(sfm=sfm, frames=frames)

    # -- up and scale ------------------------------------------------------
    start = time.perf_counter()
    gravity = None
    if config.gravity in ("auto", "sensor") and sensors is not None:
        readings = orient.load_sensor_gravity(sensors)
        if readings is not None:
            times = dict(zip(frames.names, frames.times, strict=True))
            gravity = orient.gravity_from_sensors(sfm, np.array([times[n] for n in sfm.names]), readings)
    if gravity is None:
        gravity = orient.gravity_from_scene(sfm, cloud)
    recon.gravity = gravity
    rotation = orient.rotation_to_z_up(np.array(gravity.up))
    upright = cloud.transformed(rotation)
    scale = orient.scale_from_camera_height(upright) if config.scale == "camera_height" else None
    recon.scale = scale or ScaleEstimate()
    metric = upright.transformed(np.eye(3), recon.scale.factor)
    timings["orient"] = time.perf_counter() - start

    # -- hazards -----------------------------------------------------------
    start = time.perf_counter()
    geometry = analyze_geometry(
        metric, rules, site.material, config.hazards, metric=recon.scale.is_metric,
        gravity_uncertainty_deg=gravity.uncertainty_deg,
    )
    image_set = FrameSet.load(sfm, frames_dir, rotation, recon.scale.factor)
    notes = add_image_hazards(geometry, image_set, rules, config.hazards, metric=recon.scale.is_metric)
    timings["hazards"] = time.perf_counter() - start
    recon.success = True
    result = finish(geometry, sfm=sfm, frames=frames, rotation=rotation, scale=recon.scale.factor)
    result.report.limitations = notes + result.report.limitations
    return result
