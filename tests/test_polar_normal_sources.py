import numpy as np
from pathlib import Path

from pointact.data.polar_normal_sources import (
    ambiguous_normals,
    build_cga_record,
    normals_from_depth,
    polarization_observation,
    stack_analyzers,
    viewing_directions,
)
from scripts.prepare_polar_normal_sources import hammer_group, strided_camera_intrinsics


def synthetic_analyzers(height=12, width=16, iun=0.8, dolp=0.35, aolp=0.37):
    angles = np.deg2rad([0.0, 45.0, 90.0, 135.0])
    values = [0.5 * iun * (1.0 + dolp * np.cos(2.0 * (angle - aolp))) for angle in angles]
    return np.stack(
        [np.full((height, width, 3), value, dtype=np.float32) for value in values], axis=0
    )


def test_stokes_observation_recovers_synthetic_parameters():
    analyzers = synthetic_analyzers()
    observation = polarization_observation(analyzers)
    np.testing.assert_allclose(observation["Iun"], 0.8, atol=1e-6)
    np.testing.assert_allclose(observation["DoP"], 0.35, atol=1e-6)
    np.testing.assert_allclose(observation["cos1"], np.cos(2 * 0.37), atol=1e-6)
    np.testing.assert_allclose(observation["cos2"], np.sin(2 * 0.37), atol=1e-6)
    assert observation["spec"].min() >= 0.0
    assert observation["spec"].max() <= 1.0


def test_ambiguous_normals_are_unit_camera_facing_vectors():
    dolp = np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(6, 8)
    aolp = np.linspace(0.0, np.pi, 48, dtype=np.float32).reshape(6, 8)
    candidates = ambiguous_normals(dolp, aolp).reshape(3, 3, 6, 8)
    np.testing.assert_allclose(np.linalg.norm(candidates, axis=1), 1.0, atol=2e-5)
    assert np.isfinite(candidates).all()
    assert (candidates[:, 2] <= 1e-6).all()


def test_build_cga_record_has_expected_branches_without_using_normal_as_input():
    height, width = 12, 16
    analyzers = synthetic_analyzers(height, width)
    rays = viewing_directions(height, width)
    normals = -rays
    mask = np.ones((height, width), dtype=bool)
    record = build_cga_record(analyzers, normals, mask)
    assert record["polar_observation"].shape == (11, height, width)
    assert record["physical_prior"].shape == (11, height, width)
    assert record["normal_gt"].shape == (height, width, 3)
    assert record["normal_valid_mask"].shape == (height, width)
    assert np.max(np.sum(record["normal_gt"] * rays, axis=-1)) <= 1e-6
    assert not np.shares_memory(record["physical_prior"], record["normal_gt"])


def test_shared_percentile_normalization_preserves_analyzer_ratios():
    analyzers = synthetic_analyzers() * 12.0
    normalized = stack_analyzers(analyzers, normalization="percentile")
    before = analyzers[:, 0, 0, 0] / analyzers[0, 0, 0, 0]
    after = normalized[:, 0, 0, 0] / normalized[0, 0, 0, 0]
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_depth_normals_are_camera_facing_on_a_frontoparallel_plane():
    depth = np.full((12, 16), 2.0, dtype=np.float32)
    camera_k = np.asarray([[100.0, 0.0, 7.5], [0.0, 100.0, 5.5], [0.0, 0.0, 1.0]])
    normal, valid = normals_from_depth(depth, camera_k)
    assert valid[1:-1, 1:-1].all()
    assert not valid[[0, -1]].any()
    np.testing.assert_allclose(
        normal[valid], np.broadcast_to([0.0, 0.0, -1.0], normal[valid].shape), atol=1e-5
    )


def test_camera_rays_use_pixel_centers():
    camera_k = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    rays = viewing_directions(2, 2, camera_k)
    expected = np.asarray([-0.25, -0.25, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(rays[0, 0], expected, atol=1e-7)


def test_strided_camera_intrinsics_keep_selected_pixel_rays():
    camera_k = np.asarray([[700.0, 0.0, 545.0], [0.0, 700.0, 390.0], [0.0, 0.0, 1.0]])
    original = viewing_directions(832, 1088, camera_k)
    reduced = viewing_directions(208, 272, strided_camera_intrinsics(camera_k, 4))
    np.testing.assert_allclose(reduced, original[::4, ::4], atol=1e-7)


def test_hammer_split_groups_all_trajectories_of_one_scene():
    first = Path("/data/scene2_traj1_1/polarization/pol/000.png")
    second = Path("/data/scene2_traj2_2/polarization/pol/001.png")
    other = Path("/data/scene3_traj1_1/polarization/pol/000.png")
    assert hammer_group(first) == hammer_group(second) == "hammer_scene2"
    assert hammer_group(other) != hammer_group(first)


def test_unknown_intrinsics_use_optical_axis_without_flipping_grazing_normals():
    analyzers = synthetic_analyzers()
    normal = np.broadcast_to([0.9, 0.0, -0.4358899], (12, 16, 3)).astype(np.float32)
    record = build_cga_record(
        analyzers, normal, np.ones((12, 16), dtype=bool), normal_orientation="optical_axis"
    )
    np.testing.assert_allclose(record["normal_gt"], normal, atol=1e-6)
    np.testing.assert_allclose(record["image_coordinate"][0], 0.0)
    np.testing.assert_allclose(record["image_coordinate"][1], 0.0)
    np.testing.assert_allclose(record["image_coordinate"][2], 1.0)
