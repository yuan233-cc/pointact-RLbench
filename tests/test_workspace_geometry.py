import torch
from pointact.model.vla_pointact.workspace_geometry import center_visible_points, filter_normal_consistent_points


def test_normal_filter_preserves_pixel_rows_and_rejects_empty_samples():
    points = torch.arange(24.).reshape(6, 4)
    counts = torch.tensor([3, 3])
    pixels = torch.tensor([0, 1, 2, 0, 1, 2])
    confidence = torch.tensor([1., 0., .5, 0., .2, 0.]).reshape(2, 1, 1, 1, 3)
    selected, selected_counts, keep = filter_normal_consistent_points(points, counts, pixels, confidence)
    torch.testing.assert_close(selected, points[[0, 2, 4]])
    assert selected_counts.tolist() == [2, 1]
    assert pixels[keep].tolist() == [0, 2, 1]
    confidence[1] = 0
    try:
        filter_normal_consistent_points(points, counts, pixels, confidence)
    except ValueError as error:
        assert "No normal-consistent" in str(error)
    else:
        raise AssertionError("Must not fall back to an untrusted point")


def test_visible_centering_preserves_camera_coordinates_and_other_features():
    points = torch.tensor([[1., 2., 3., .3], [2., 3., 5., .4], [-1., 0., 2., .5]])
    counts = torch.tensor([2, 1])
    transforms = torch.eye(4).repeat(2, 1, 1, 1)
    transforms[:, :, :3, 3] = torch.tensor([[[.1, .2, .3]], [[.2, .3, .4]]])
    centered, updated = center_visible_points(points, counts, transforms)
    batch_ids = torch.tensor([0, 0, 1])
    before = points[:, :3] + transforms[batch_ids, 0, :3, 3]
    after = centered[:, :3] + updated[batch_ids, 0, :3, 3]
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(centered[:2, :3].mean(0), torch.zeros(3))
    torch.testing.assert_close(centered[2, :3], torch.zeros(3))
    torch.testing.assert_close(centered[:, 3:], points[:, 3:])
