"""End-to-end check of the hazard geometry on a small synthetic pile."""

import numpy as np
import pytest

from brink.domain import HazardType
from brink.hazards.engine import analyze_geometry
from brink.rules import load_rules
from brink.synth.capture import CaptureSpec, camera_path
from brink.synth.sampling import NOISE_LEVELS, sample_cloud
from brink.synth.scene import SceneSpec, UndercutSpec, build_scene

pytestmark = pytest.mark.slow


def _analyse(undercut: UndercutSpec | None):
    spec = SceneSpec(
        name="t", seed=3, kind="pile", height=1.2, slope_deg=65.0, length=3.0, top_width=0.9,
        roughness=0.01, voxel=0.015, undercut=undercut,
    )
    scene = build_scene(spec)
    path = camera_path(spec, CaptureSpec(seed=1, path="arc", standoff=2.6, coverage=0.8))
    cloud = sample_cloud(scene, path.positions, NOISE_LEVELS["typical"], seed=1)
    return analyze_geometry(cloud, load_rules(), "damp_soil")


def test_control_pile_raises_no_overhang_and_measures_slope():
    result = analyze_geometry_cached(None)
    assert not [h for h in result.hazards if h.type in (HazardType.UNDERCUT, HazardType.OVERHANG)]
    main = max(result.faces, key=lambda f: -f.outward[1])
    assert abs(main.slope_deg - 65.0) < 2.0
    assert abs(main.height - 1.2) < 0.1


def test_undercut_is_found_with_about_the_right_depth():
    result = analyze_geometry_cached(UndercutSpec(center_x=0.2, width=0.8, depth=0.3, height=0.3))
    found = [h for h in result.hazards if h.type is HazardType.UNDERCUT]
    assert len(found) == 1
    assert abs(found[0].measurement("depth").value - 0.3) < 0.08
    assert abs(found[0].location[0] - 0.2) < 0.3
    assert "Do not work below it" in found[0].message


_cache: dict = {}


def analyze_geometry_cached(undercut):
    key = None if undercut is None else undercut.model_dump_json()
    if key not in _cache:
        _cache[key] = _analyse(undercut)
    return _cache[key]


def test_wilson_interval_is_sane():
    from brink.bench.metrics import wilson

    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert np.isnan(wilson(0, 0)[0])
