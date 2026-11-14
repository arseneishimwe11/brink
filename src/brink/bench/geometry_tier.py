"""Geometry tier: hazard code on sampled ground-truth surfaces.

No images, no camera solving. The cloud is what a reconstruction of a given
quality would produce, so errors here belong to the hazard geometry alone.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from brink.bench.metrics import score_case
from brink.bench.suites import BenchCase
from brink.hazards.engine import analyze_geometry
from brink.hazards.params import HazardParams
from brink.rules import load_rules
from brink.synth.capture import camera_path
from brink.synth.sampling import NOISE_LEVELS, sample_cloud
from brink.synth.scene import build_scene

# Reconstruction quality, plus how wrong "up" and scale are assumed to be.
CONDITIONS: dict[str, dict[str, float]] = {
    "ideal": {"gravity_error_deg": 0.0, "scale_error": 0.0},
    "typical": {"gravity_error_deg": 1.0, "scale_error": 0.05},
    "poor": {"gravity_error_deg": 3.0, "scale_error": 0.12},
}


def run_case(case: BenchCase, methods: list[str], conditions: list[str], cache_dir: str) -> list[dict[str, Any]]:
    rules = load_rules()
    scene = build_scene(case.scene, cache_dir=cache_dir)
    truth = scene.ground_truth()
    path = camera_path(case.scene, case.capture)
    records = []
    for condition in conditions:
        extra = dict(CONDITIONS[condition])
        sign = 1.0 if case.scene.seed % 2 else -1.0
        extra["scale_error"] *= sign
        noise = NOISE_LEVELS[condition].model_copy(update=extra)
        cloud = sample_cloud(scene, path.positions, noise, seed=case.scene.seed)
        for method in methods:
            start = time.perf_counter()
            result = analyze_geometry(
                cloud, rules, case.scene.material, HazardParams(undercut_method=method),
                gravity_uncertainty_deg=extra["gravity_error_deg"],
            )
            elapsed = time.perf_counter() - start
            record = score_case(result.faces, result.hazards, truth, rules, case.scene.material)
            record.update(
                case=case.id, kind=case.kind, condition=condition, method=method,
                time_s=elapsed, reconstructed=True, gt_slope=case.scene.slope_deg,
                points=len(result.cloud),
            )
            records.append(record)
    return records


def _worker(args: tuple[BenchCase, list[str], list[str], str]) -> list[dict[str, Any]]:
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    np.seterr(all="ignore")
    return run_case(*args)


def run_geometry_tier(
    cases: list[BenchCase],
    methods: list[str],
    conditions: list[str],
    cache_dir: str | Path,
    workers: int = 4,
) -> list[dict[str, Any]]:
    jobs = [(case, methods, conditions, str(cache_dir)) for case in cases]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            records.extend(_worker(job))
        return records
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for chunk in pool.map(_worker, jobs):
            records.extend(chunk)
    return records
