"""Benchmark cases, generated deterministically from (split, index).

Two splits with disjoint seeds:
- ``dev``: used while developing and tuning.
- ``test``: frozen. Reported, never tuned on.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from brink.synth.capture import CaptureSpec
from brink.synth.scene import CrackSpec, PuddleSpec, SceneSpec, UndercutSpec

_BASE_SEED = {"dev": 1000, "test": 5000}

# Kinds whose hazards are purely geometric (scored in every tier).
GEOMETRY_KINDS = [
    "pile_control", "pile_undercut", "pile_undercut", "face_undercut", "face_control",
    "pile_undercut", "face_undercut", "pile_control", "face_overhang", "pile_undercut",
    "face_control", "face_undercut", "pile_control", "pile_undercut", "face_undercut",
    "pile_control", "pile_undercut", "face_control", "face_undercut", "pile_undercut",
]
# Kinds that need image evidence (scored only where images exist).
IMAGE_KINDS = ["pile_crack", "pile_puddle", "face_puddle", "pile_crack", "pile_puddle"]

_DEPTH_BANDS = [(0.10, 0.16), (0.20, 0.30), (0.30, 0.50), (0.20, 0.30), (0.30, 0.50), (0.50, 1.00)]


class BenchCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    split: str
    kind: str
    scene: SceneSpec
    capture: CaptureSpec


def _pile(rng: np.random.Generator, name: str, seed: int, **extra) -> SceneSpec:
    return SceneSpec(
        name=name, seed=seed, kind="pile",
        material=str(rng.choice(["damp_soil", "dry_sand", "clay"])),
        height=float(rng.uniform(1.0, 1.5)), slope_deg=float(rng.uniform(45, 85)),
        length=float(rng.uniform(2.5, 4.0)), top_width=float(rng.uniform(0.7, 1.2)),
        roughness=float(rng.uniform(0.008, 0.02)), voxel=0.01, **extra,
    )


def _face(rng: np.random.Generator, name: str, seed: int, slope: tuple[float, float] = (55, 90), **extra) -> SceneSpec:
    return SceneSpec(
        name=name, seed=seed, kind="face",
        material=str(rng.choice(["weathered_rock", "clay", "damp_soil"])),
        height=float(rng.uniform(3.0, 6.0)), slope_deg=float(rng.uniform(*slope)),
        length=float(rng.uniform(6.0, 12.0)), roughness=float(rng.uniform(0.02, 0.04)),
        voxel=0.025, **extra,
    )


def _undercut(rng: np.random.Generator, scene: SceneSpec, band: tuple[float, float]) -> SceneSpec:
    big = scene.kind == "face"
    depth = float(rng.uniform(*band))
    if not big:
        depth = min(depth, 0.5)
    height = float(rng.uniform(0.4, 1.2) if big else rng.uniform(0.15, 0.35))
    height = min(height, 0.35 * scene.height)
    width = float(rng.uniform(1.5, 4.0) if big else rng.uniform(0.5, 1.4))
    room = max(scene.length / 2 - width / 2 - 0.2, 0.0)
    cut = UndercutSpec(center_x=float(rng.uniform(-room, room)), width=width, depth=depth, height=height)
    return scene.model_copy(update={"undercut": cut})


def _capture(rng: np.random.Generator, scene: SceneSpec, seed: int) -> CaptureSpec:
    if scene.kind == "pile":
        arc = rng.random() < 0.6
        return CaptureSpec(
            seed=seed, path="arc" if arc else "along",
            standoff=float(rng.uniform(2.0, 3.5)), camera_height=float(rng.uniform(1.2, 1.55)),
            coverage=float(rng.uniform(0.6, 1.1) if arc else rng.uniform(1.0, 1.6)),
            duration_s=float(rng.uniform(45, 75)), shake_deg=float(rng.uniform(0.4, 1.5)),
            lighting=str(rng.choice(["overcast", "sun", "low"], p=[0.5, 0.35, 0.15])),
            blur=float(rng.uniform(0.0, 0.8)),
        )
    arc = rng.random() < 0.2
    return CaptureSpec(
        seed=seed, path="arc" if arc else "along",
        standoff=float(rng.uniform(4.0, 8.0)), camera_height=float(rng.uniform(1.2, 1.55)),
        coverage=float(rng.uniform(0.5, 0.8) if arc else rng.uniform(0.9, 1.2)),
        duration_s=float(rng.uniform(60, 100)), shake_deg=float(rng.uniform(0.4, 1.5)),
        lighting=str(rng.choice(["overcast", "sun", "low"], p=[0.5, 0.35, 0.15])),
        blur=float(rng.uniform(0.0, 0.8)),
    )


def make_case(split: str, index: int, kinds: list[str] | None = None) -> BenchCase:
    kinds = kinds or GEOMETRY_KINDS
    kind = kinds[index % len(kinds)]
    seed = _BASE_SEED[split] + index * 7 + (0 if kinds is GEOMETRY_KINDS else 100_000)
    rng = np.random.default_rng(seed)
    name = f"{split}-{'g' if kinds is GEOMETRY_KINDS else 'i'}{index:03d}-{kind}"
    band = _DEPTH_BANDS[(index // 2) % len(_DEPTH_BANDS)]
    if kind == "pile_control":
        scene = _pile(rng, name, seed)
    elif kind == "pile_undercut":
        scene = _undercut(rng, _pile(rng, name, seed), band)
    elif kind == "face_control":
        scene = _face(rng, name, seed)
    elif kind == "face_undercut":
        scene = _undercut(rng, _face(rng, name, seed), band)
    elif kind == "face_overhang":
        scene = _face(rng, name, seed, slope=(96, 108))
    elif kind == "pile_crack":
        base = _pile(rng, name, seed)
        length = float(rng.uniform(0.8, min(2.0, base.length - 0.6)))
        crack = CrackSpec(
            center_x=float(rng.uniform(-0.3, 0.3)), length=length,
            width=float(rng.uniform(0.025, 0.06)), depth=float(rng.uniform(0.1, 0.3)),
            offset=float(rng.uniform(0.15, 0.4)),
        )
        scene = base.model_copy(update={"crack": crack, "top_width": max(base.top_width, 1.0), "height": min(base.height, 1.15)})
    elif kind in ("pile_puddle", "face_puddle"):
        base = _pile(rng, name, seed) if kind == "pile_puddle" else _face(rng, name, seed)
        big = base.kind == "face"
        puddle = PuddleSpec(
            center_x=float(rng.uniform(-0.25, 0.25) * base.length),
            radius_x=float(rng.uniform(0.8, 1.6) if big else rng.uniform(0.35, 0.6)),
            radius_y=float(rng.uniform(0.5, 0.9) if big else rng.uniform(0.25, 0.4)),
        )
        scene = base.model_copy(update={"puddle": puddle})
    else:
        raise ValueError(kind)
    capture = _capture(rng, scene, seed)
    if kind == "pile_crack":
        # A crack behind the top edge cannot be seen from chest height on a pile
        # this tall. These captures hold the phone overhead and stand closer.
        capture = capture.model_copy(
            update={"camera_height": float(rng.uniform(1.8, 2.1)), "standoff": float(rng.uniform(1.6, 2.4)), "aim_height": 0.85}
        )
    return BenchCase(id=name, split=split, kind=kind, scene=scene, capture=capture)


def geometry_cases(split: str, count: int) -> list[BenchCase]:
    return [make_case(split, i) for i in range(count)]


def image_cases(split: str, count: int) -> list[BenchCase]:
    return [make_case(split, i, IMAGE_KINDS) for i in range(count)]


def with_tilt(case: BenchCase) -> BenchCase:
    """Tip the scene 0 to 4 degrees off true gravity, as uneven pit floors are."""
    rng = np.random.default_rng(case.scene.seed + 999)
    scene = case.scene.model_copy(
        update={"tilt_deg": float(rng.uniform(0, 4)), "tilt_azimuth_deg": float(rng.uniform(0, 360))}
    )
    # Render about 150 frames per video whatever its length: enough for frame
    # selection to have a choice, without paying to render frames it will drop.
    fps = float(np.clip(150.0 / case.capture.duration_s, 1.5, 4.0))
    return case.model_copy(update={"scene": scene, "capture": case.capture.model_copy(update={"fps": fps})})


def full_cases(split: str, geometry_count: int, image_count: int) -> list[BenchCase]:
    """Cases for the video tier: geometric hazards first, then image-based ones."""
    cases = geometry_cases(split, geometry_count) + image_cases(split, image_count)
    return [with_tilt(c) for c in cases]
