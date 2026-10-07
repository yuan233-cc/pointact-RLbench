import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from pointact.data.workspace_geometry_cache import GeometryColumnWriter, CachedWorkspaceGeometryDataset


def test_prefix_recovery_split_boundary_and_checksum_publication():
    spec = importlib.util.spec_from_file_location("cache_builder", Path(__file__).parents[1] / "scripts/cache_workspace_geometry.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    accessed_instances = []

    class Source:
        def __init__(self, root, backbone, split, **kwargs):
            self.keys = [b"0-0", b"0-1"] if split == "train" else [b"9-0"]
            self.envs = {}

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, index):
            accessed_instances.append(id(self))
            return dict(points=torch.arange(27.).reshape(3, 9),
                point_pixel_indices=torch.tensor([5, 9, 17]),
                polar_images=torch.ones(1, 7, 2, 2),
                observed_depth=torch.tensor([float("nan"), 1., 1., 1.]).reshape(1, 1, 2, 2),
                polar_workspace_mask=torch.ones(1, 2, 2, dtype=torch.bool))

    class Teacher(nn.Module):
        def __init__(self, *args):
            super().__init__()

        def cuda(self):
            return self

        def encode_teacher(self, batch):
            count = len(batch["polar_images"])
            return dict(teacher_normals=torch.ones(count, 3, 2, 2),
                **{f"teacher_feature_{i}": torch.ones(count, 1, 1, 1) for i in range(3)})

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "teacher.pt"
        torch.save({"dummy": 1}, checkpoint)
        source = Source(root, "tasknet", "train")
        sample = source[0]
        sample.update({k: v[0] for k, v in Teacher().encode_teacher({"polar_images": sample["polar_images"][None]}).items()})
        interrupted = GeometryColumnWriter(root / "interrupted", [b"0-0", b"0-1", b"9-0"], "tasknet", checkpoint)
        interrupted.append(sample)
        interrupted.append(sample)
        for column in interrupted.columns.values():
            column.flush()
        accessed_instances.clear()
        argv = ["cache", "--backbone", "tasknet", "--dataset-root", str(root),
            "--teacher-checkpoint", str(checkpoint), "--concerto-checkpoint", str(checkpoint),
            "--cache-dir", str(root / "published"), "--scratch-dir", str(root / "scratch"),
            "--reuse-prefix-cache", str(root / "interrupted"), "--reuse-prefix-rows", "2",
            "--batch-size", "1", "--workers", "0"]
        with patch.object(sys, "argv", argv), patch.object(builder, "WorkspaceGeometryDataset", Source), \
             patch.object(builder, "WorkspaceGeometryModel", Teacher), \
             patch.object(torch.Tensor, "cuda", lambda self, **kwargs: self):
            builder.main()
        assert len(set(accessed_instances)) == 1
        assert not (root / "interrupted/manifest.json").exists()
        assert not (root / "published.partial").exists()
        train = CachedWorkspaceGeometryDataset(root / "published", "tasknet", checkpoint)
        val = CachedWorkspaceGeometryDataset(root / "published", "tasknet", checkpoint, split="val")
        assert len(train) == 2 and len(val) == 1
        assert torch.isnan(val[0]["observed_depth"][0, 0, 0, 0])
