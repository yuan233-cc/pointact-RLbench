import numpy as np

from pointact.data.polar_normal_sources import (
    ambiguous_normals,
    build_cga_record,
    polarization_observation,
    stack_analyzers,
    viewing_directions,
)


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
