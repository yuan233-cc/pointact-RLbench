"""Check that interaction labels cover both operated and related objects."""

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/10_rlbench/export_visible_target_gt.py"
SPEC = importlib.util.spec_from_file_location("export_visible_target_gt", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_interaction_mask_includes_related_object_without_changing_original_mode(tmp_path):
    frame_path = tmp_path / "frame.npz"
    snapshot_path = tmp_path / "snapshot.json"
    coords = np.array([[[0.5, 0.5, 1.0], [1.5, 0.5, 1.0], [2.5, 0.5, 1.0]]], dtype=np.float32)
    cloud = np.pad(coords.reshape(-1, 3), ((0, 0), (0, 6)))
    np.savez(frame_path, point_cloud=coords, depth_m=np.ones((1, 3), dtype=np.float32),
             object_mask=np.array([[10, 20, 30]], dtype=np.int32))
    snapshot_path.write_text(json.dumps({
        "meshes": [{"name": "handset/0", "handle": 10},
                   {"name": "base/0", "handle": 20}],
        "cameras": {"front": {"to_world": np.eye(4).tolist(),
                              "intrinsics": np.eye(3).tolist()}},
    }))

    original = EXPORT.export_record(frame_path, snapshot_path, cloud,
                                    ("handset",), 2, 0.005, 1)
    interaction = EXPORT.export_record(frame_path, snapshot_path, cloud, {
        "manipulated": ["handset"], "related": {"base": ["base"]},
    }, 2, 0.005, 1)

    assert original["input_mask"].tolist() == [True, False, False]
    assert interaction["input_mask"].tolist() == [True, True, False]
    assert len(original["points"]) == 1
    assert len(interaction["points"]) == 2


def test_interaction_sampling_retains_small_related_group():
    large = np.arange(100, dtype=np.float32)[:, None] * np.array([[0.01, 0, 0]], dtype=np.float32)
    tiny = np.array([[0, 1, 0], [0.01, 1, 0]], dtype=np.float32)
    points = EXPORT.sample_interaction_groups([large, large + [0, 2, 0], tiny], 12, 7)
    assert len(points) == 12
    assert (points[:, 1] == 0).sum() == 6
    assert (points[:, 1] == 1).sum() == 2
    assert (points[:, 1] == 2).sum() == 4
