import tempfile
from pathlib import Path
import torch
from pointact.data.workspace_geometry_cache import GeometryColumnWriter, CachedWorkspaceGeometryDataset


def test_column_cache_preserves_geometry_nan_teacher_and_split():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "teacher.pt"
        torch.save({"dummy": 1}, checkpoint)
        sample = dict(points=torch.arange(27.).reshape(3, 9), point_pixel_indices=torch.tensor([5, 9, 17]),
                      observed_depth=torch.tensor([float("nan"), 1.]).reshape(1, 1, 1, 2),
                      teacher_normals=torch.ones(3, 1, 2), teacher_feature_0=torch.ones(48, 2, 2))
        writer = GeometryColumnWriter(root / "cache", [b"0-0", b"9-0"], "tasknet", checkpoint, point_capacity=4)
        writer.append(sample)
        writer.append(sample)
        writer.finish()
        for preload in (True, False):
            dataset = CachedWorkspaceGeometryDataset(root / "cache", "tasknet", checkpoint,
                                                       split="val", preload=preload)
            assert dataset.keys == [b"9-0"]
            result = dataset[0]
            assert len(result["points"]) == 3
            assert torch.isnan(result["observed_depth"][0, 0, 0, 0])
            # Reordering never changes the association of a point with its pixel.
            original_rows = (result["points"][:, 0] / 9).long()
            torch.testing.assert_close(result["point_pixel_indices"], sample["point_pixel_indices"][original_rows])
            torch.testing.assert_close(result["teacher_normals"], sample["teacher_normals"])
            torch.testing.assert_close(result["teacher_feature_0"], sample["teacher_feature_0"])
        torch.save({"dummy": 2}, checkpoint)
        try:
            CachedWorkspaceGeometryDataset(root / "cache", "tasknet", checkpoint)
        except ValueError as error:
            assert "Stale cache" in str(error)
        else:
            raise AssertionError("Changed teacher must invalidate the cache")
