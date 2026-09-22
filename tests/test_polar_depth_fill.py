"""Checks for image-aligned polar features on depth-filled RLBench points."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/10_rlbench"))
from polar_depth_fill import (
    add_polar_depth_filled_points, corruption_hole_pixels, fill_depth_multiscale,
)


def make_example():
    height = width = 16
    x_image = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    frame = {
        "rgb": np.stack((x_image, np.zeros_like(x_image), np.zeros_like(x_image)), axis=-1),
        "DoLP": np.full((height, width), 0.4, dtype=np.float32),
        "AoLP": np.full((height, width), 0.2, dtype=np.float32),
        "valid_mask": np.ones((height, width), dtype=bool),
        "AoLP_valid_mask": np.ones((height, width), dtype=bool),
    }
    y, x = np.mgrid[6:9, 6:9]
    pixels = (y * width + x).ravel()
    pixels = pixels[pixels != 7 * width + 7]
    xyz = np.column_stack((
        (pixels % width + 0.5 - 8) / 12,
        (pixels // width + 0.5 - 8) / 12,
        np.ones(len(pixels)),
    ))
    cloud = np.column_stack((
        xyz, np.zeros((len(pixels), 3)), np.full(len(pixels), 0.4),
        np.full(len(pixels), np.cos(0.4)), np.full(len(pixels), np.sin(0.4)),
    )).astype(np.float32)
    return frame, cloud, pixels.astype(np.int32)


def complete(cloud, pixels, frame, hole_pixels=None):
    if hole_pixels is None:
        hole_pixels = np.array([7 * 16 + 7], dtype=np.int32)
    return add_polar_depth_filled_points(
        cloud, pixels, frame, np.eye(4, dtype=np.float32), 12.0,
        np.array([8.0, 8.0]), hole_pixel_indices=hole_pixels,
    )


def test_hole_receives_same_pixel_polar_and_existing_points_stay_intact():
    frame, cloud, pixels = make_example()
    # The old renderer could mark an image pixel invalid while selecting
    # workspace points even though its AoLP and DoLP are available.
    frame["valid_mask"][7, 7] = False
    result, all_pixels, filled = complete(cloud, pixels, frame)
    np.testing.assert_array_equal(all_pixels[filled], [7 * 16 + 7])
    np.testing.assert_array_equal(result[:len(cloud)], cloud)
    np.testing.assert_array_equal(all_pixels[:len(cloud)], pixels)
    np.testing.assert_allclose(result[filled, 6], frame["DoLP"].ravel()[all_pixels[filled]])
    np.testing.assert_allclose(result[filled, 7], np.cos(0.4), atol=1e-6)
    np.testing.assert_allclose(result[filled, 8], np.sin(0.4), atol=1e-6)
    # Reproject the generated XYZ to verify the saved image correspondence.
    u = np.floor(12 * result[filled, 0] / result[filled, 2] + 8)
    v = np.floor(12 * result[filled, 1] / result[filled, 2] + 8)
    np.testing.assert_array_equal(v.astype(int) * 16 + u.astype(int), all_pixels[filled])


def test_only_corruption_holes_are_filled_even_if_other_pixels_have_estimated_depth():
    frame, cloud, pixels = make_example()
    frame["AoLP_valid_mask"][7, 7] = False
    result, all_pixels, filled = complete(cloud, pixels, frame)
    np.testing.assert_array_equal(all_pixels[filled], [7 * 16 + 7])
    hole_row = result[np.flatnonzero(all_pixels == 7 * 16 + 7)[0]]
    np.testing.assert_array_equal(hole_row[6:9], np.array([0.4, 0.0, 0.0], dtype=np.float32))
    assert len(result) == len(cloud) + int(filled.sum())


def test_corruption_holes_use_source_pixels_not_moved_point_positions():
    clean_pixels = np.array([10, 11, 12, 13], dtype=np.int32)
    retained_source_pixels = np.array([10, 12, 13], dtype=np.int32)
    np.testing.assert_array_equal(
        corruption_hole_pixels(clean_pixels, retained_source_pixels), [11])
    frame, cloud, pixels = make_example()
    result, all_pixels, filled = complete(
        cloud, pixels, frame, hole_pixels=np.array([], dtype=np.int32))
    assert not filled.any()
    np.testing.assert_array_equal(result, cloud)
    np.testing.assert_array_equal(all_pixels, pixels)


def test_empty_depth_does_not_hallucinate_geometry():
    filled = fill_depth_multiscale(np.zeros((8, 8), dtype=np.float32))
    assert np.count_nonzero(filled) == 0
