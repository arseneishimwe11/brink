"""Unit tests for pipeline pieces that can be checked with plain numbers."""

import numpy as np

from brink.bench.full_tier import umeyama
from brink.geometry import PointCloud
from brink.stages import orient
from brink.stages.dense.fusion import DepthView, fuse


def _rotation(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(40, 3))
    rotation, scale, shift = _rotation([1, 2, 3], 37.0), 2.5, np.array([1.0, -2.0, 0.5])
    target = scale * source @ rotation.T + shift
    r, s, t = umeyama(source, target)
    assert np.allclose(r, rotation, atol=1e-8)
    assert np.isclose(s, scale)
    assert np.allclose(t, shift, atol=1e-8)


def test_rotation_to_z_up_maps_up_to_z():
    up = np.array([0.2, -0.9, 0.3])
    r = orient.rotation_to_z_up(up)
    assert np.allclose(r @ (up / np.linalg.norm(up)), [0, 0, 1], atol=1e-9)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_scale_from_camera_height_uses_the_floor_under_the_walk():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(-3, 3, 4000), rng.uniform(-3, 1, 4000), rng.normal(0, 0.005, 4000)])
    wall = np.column_stack([rng.uniform(-3, 3, 2000), np.full(2000, 1.0), rng.uniform(0, 2, 2000)])
    cams = np.column_stack([np.linspace(-2, 2, 30), np.full(30, -2.0), np.full(30, 0.9)])  # 0.9 model units up
    estimate = orient.scale_from_camera_height(PointCloud(np.vstack([floor, wall]), cameras=cams))
    assert estimate is not None and estimate.source == "camera_height"
    assert abs(estimate.factor - orient.ASSUMED_PHONE_HEIGHT_M / 0.9) < 0.03


def test_fusion_keeps_agreeing_depths_and_drops_lone_ones():
    k = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])

    def view(x, group, spike=False):
        depth = np.full((48, 64), 5.0, np.float32)  # a wall 5 units in front
        if spike:
            depth[20:24, 30:34] = 2.0  # something only this view claims to see
        return DepthView(depth, np.zeros((48, 64, 3), np.uint8), k, np.eye(3), np.array([x, 0.0, 0.0]), group)

    views = [view(0.0, 0, spike=True), view(0.2, 1), view(0.4, 2)]
    cloud = fuse(views, np.zeros((3, 3)), min_agree=2, stride=1)
    assert len(cloud) > 0
    assert cloud.xyz[:, 2].min() > 4.9  # the lone near patch was rejected
    assert np.allclose(np.linalg.norm(cloud.view, axis=1), 1.0, atol=1e-5)


def test_floaters_are_removed():
    rng = np.random.default_rng(2)
    surface = np.column_stack([rng.uniform(0, 4, 20000), rng.uniform(0, 4, 20000), rng.normal(0, 0.003, 20000)])
    clump = rng.normal([2, 2, 1.5], 0.02, (40, 3))
    cleaned = PointCloud(np.vstack([surface, clump])).without_floaters(0.05)
    assert len(cleaned) == 20000


def test_mask_drops_sky_and_green_but_keeps_soil():
    from brink.stages.masking import MaskingParams, ground_mask

    rng = np.random.default_rng(3)
    image = np.zeros((120, 160, 3), np.uint8)
    image[:40] = (215, 205, 195)  # pale sky (BGR), smooth
    soil = np.array([60, 95, 140]) + rng.integers(-35, 35, (80, 160, 1))  # textured brown
    image[40:] = np.clip(soil, 0, 255)
    image[80:100, 20:60] = (40, 150, 50)  # a green bush
    mask = ground_mask(image, MaskingParams(grow_px=2))
    assert (mask[:30] == 0).mean() > 0.95  # sky ignored
    assert (mask[85:95, 25:55] == 0).mean() > 0.95  # vegetation ignored
    assert (mask[50:75, 80:150] == 255).mean() > 0.95  # soil kept
