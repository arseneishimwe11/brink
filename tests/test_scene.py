import numpy as np

from brink.synth.scene import SceneSpec, UndercutSpec, ground_truth, sdf


def _spec(**extra) -> SceneSpec:
    return SceneSpec(name="t", seed=1, kind="face", height=4.0, slope_deg=70.0, length=8.0, roughness=0.0, **extra)


def test_sdf_sign_matches_geometry():
    spec = _spec()
    air = np.array([[0.0, -1.0, 1.0], [0.0, 0.5, 4.5]])
    ground = np.array([[0.0, 2.0, 1.0], [0.0, 0.0, -0.1]])
    assert (sdf(air, spec) > 0).all()
    assert (sdf(ground, spec) < 0).all()


def test_undercut_carves_exactly_the_stated_depth():
    cut = UndercutSpec(center_x=0.0, width=2.0, depth=0.6, height=0.8)
    spec = _spec(undercut=cut)
    lip_y = cut.height / np.tan(np.radians(spec.slope_deg))
    just_inside = np.array([[0.0, lip_y + cut.depth - 0.03, 0.3]])
    just_beyond = np.array([[0.0, lip_y + cut.depth + 0.03, 0.3]])
    assert sdf(just_inside, spec) > 0  # air inside the notch
    assert sdf(just_beyond, spec) < 0  # solid behind its back wall
    truth = ground_truth(spec)
    assert truth["hazards"][0]["type"] == "undercut"
    assert truth["hazards"][0]["depth"] == 0.6


def test_leaning_face_is_listed_as_overhang():
    truth = ground_truth(_spec().model_copy(update={"slope_deg": 100.0}))
    assert [h["type"] for h in truth["hazards"]] == ["overhang"]
    assert np.isclose(truth["hazards"][0]["depth"], 4.0 * np.tan(np.radians(10.0)))
