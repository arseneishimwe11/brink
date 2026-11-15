"""Full tier: a rendered phone video goes in, hazards come out, truth scores them.

This is the tier that tests the core claim, because it includes frame
selection, camera solving, dense reconstruction, gravity and scale.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from brink.bench.metrics import score_case
from brink.bench.suites import BenchCase
from brink.domain import Site
from brink.pipeline import PipelineConfig, run_pipeline
from brink.rules import load_rules
from brink.synth.render import render_capture
from brink.synth.scene import build_scene

VARIANTS: dict[str, dict[str, Any]] = {
    "sensor": {},  # defaults: SIFT cameras, stereo dense, gravity from the phone sensor file
    "scene": {"gravity": "scene"},
    "aliked": {"sfm": "colmap_aliked"},
    "global": {"sfm": "colmap_sift_global"},
    "monodepth": {"dense": "monodepth"},
}


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Similarity transform (rotation, scale, shift) taking ``source`` onto ``target``."""
    ms, mt = source.mean(0), target.mean(0)
    sc, tc = source - ms, target - mt
    u, sing, vt = np.linalg.svd(tc.T @ sc / len(source))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(sing) @ d) / sc.var(0).sum())
    return rotation, scale, mt - scale * rotation @ ms


def render_case(case: BenchCase, render_dir: Path, cache_dir: Path) -> Path:
    """Render the case's video unless an up-to-date one is already there."""
    key = case.model_dump_json()
    marker = render_dir / f"{case.id}.key"
    video = render_dir / f"{case.id}.mp4"
    if marker.exists() and marker.read_text() == key and video.exists():
        return video
    scene = build_scene(case.scene, cache_dir=cache_dir)
    render_capture(scene, case.capture, render_dir, case.id)
    marker.write_text(key)
    return video


def run_case(case: BenchCase, variants: list[str], derived: str) -> list[dict[str, Any]]:
    root = Path(derived)
    rules = load_rules()
    video = render_case(case, root / "renders", root / "synth_cache")
    truth = json.loads(video.with_suffix(".truth.json").read_text())
    gt_times = np.array([f["t"] for f in truth["frames"]])
    gt_positions = np.array([f["position"] for f in truth["frames"]])
    true_up = np.array(truth["scene"]["true_up"])
    records = []
    for variant in variants:
        config = PipelineConfig(material=case.scene.material, **VARIANTS[variant])
        start = time.perf_counter()
        result = run_pipeline(video, root / "runs" / case.id, config, rules, Site(id="synthetic", material=case.scene.material))
        elapsed = time.perf_counter() - start
        recon = result.report.reconstruction
        record: dict[str, Any] = {
            "case": case.id, "kind": case.kind, "variant": variant, "condition": variant,
            "reconstructed": recon.success, "failure": recon.failure_reason,
            "time_s": sum(recon.timings_s.values()), "wall_s": elapsed, "timings": recon.timings_s,
            "frames_selected": recon.frames_selected, "frames_registered": recon.frames_registered,
            "points": recon.points, "gt_slope": truth["scene"]["faces"][0]["slope_deg"],
            "lighting": case.capture.lighting, "blur": case.capture.blur, "path": case.capture.path,
        }
        if not recon.success or result.sfm is None or result.frames is None:
            record["gt_overhangs"] = [
                {"type": h["type"], "depth": h["depth"], "detected": False}
                for h in truth["scene"]["hazards"] if h["type"] in ("undercut", "overhang")
            ]
            records.append(record)
            continue
        times = dict(zip(result.frames.names, result.frames.times, strict=True))
        match = [int(np.argmin(np.abs(gt_times - times[n]))) for n in result.sfm.names]
        r_g, s_g, t_g = umeyama(result.sfm.positions, gt_positions[match])
        residual = s_g * result.sfm.positions @ r_g.T + t_g - gt_positions[match]
        to_truth = (r_g @ result.rotation.T, s_g / result.scale, t_g)
        record.update(score_case(result.report.faces, result.report.hazards, truth["scene"], rules, case.scene.material, to_truth))
        up_est = r_g @ np.array(recon.gravity.up)
        record.update(
            pose_rmse_m=float(np.sqrt((residual**2).sum(1).mean())),
            gravity_err_deg=float(np.degrees(np.arccos(np.clip(up_est @ true_up, -1, 1)))),
            gravity_source=recon.gravity.source,
            scale_err_rel=float(result.scale / s_g - 1.0) if recon.scale.is_metric else None,
            reproj_px=recon.mean_reprojection_error_px,
            hazards=[json.loads(h.model_dump_json()) for h in result.report.hazards],
        )
        records.append(record)
    return records


def _worker(args: tuple[BenchCase, list[str], str]) -> list[dict[str, Any]]:
    import cv2
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    cv2.setNumThreads(int(os.environ.get("BRINK_THREADS", "2")))
    np.seterr(all="ignore")
    try:
        return run_case(*args)
    except Exception as exc:  # one broken case must not sink a multi-hour run
        case, variants, _ = args
        return [
            {"case": case.id, "kind": case.kind, "variant": v, "condition": v, "reconstructed": False,
             "failure": f"crash: {exc!r}", "gt_overhangs": [], "crashed": True}
            for v in variants
        ]


def _render_worker(args: tuple[BenchCase, str]) -> str:
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    np.seterr(all="ignore")
    case, derived = args
    root = Path(derived)
    render_case(case, root / "renders", root / "synth_cache")
    return case.id


def run_full_tier(cases: list[BenchCase], variants: list[str], derived: str | Path, workers: int = 2) -> list[dict[str, Any]]:
    """Render every case, then run the pipeline on each.

    Rendering is one core per video, so it runs one video per core. The
    pipeline stages are multi-threaded, so fewer of those run side by side.
    """
    cores = os.cpu_count() or 4
    context = get_context("spawn")
    os.environ["BRINK_RENDER_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=cores, mp_context=context) as pool:
        list(pool.map(_render_worker, [(case, str(derived)) for case in cases]))

    jobs = [(case, variants, str(derived)) for case in cases]
    os.environ.setdefault("BRINK_THREADS", str(max(1, cores // max(workers, 1))))
    records: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            records.extend(_worker(job))
        return records
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for chunk in pool.map(_worker, jobs):
            records.extend(chunk)
    return records
