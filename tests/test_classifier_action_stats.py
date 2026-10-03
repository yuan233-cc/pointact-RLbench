"""Regression checks for the polar classifier's raw action targets."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE = REPO_ROOT / "experiments/10_rlbench/prepare_classifier_data_config.py"


class ClassifierActionStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stats_path = self.root / "original.json"
        self.stats = {
            "state_mean": [0.2] * 7,
            "state_std": [0.5] * 7,
            "action_min": [-0.5] * 6 + [0.0],
            "action_max": [0.5] * 6 + [1.0],
            "action_mean": [0.1] * 6 + [0.43595327657889443],
            "action_std": [0.2] * 6 + [0.4958810514821292],
        }
        self.stats_path.write_text(json.dumps(self.stats))
        self.config_path = self.root / "input.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "lerobot_datasets": [{
                "repo_id": "polar-dataset",
                "converted_rot_type": "euler",
                "is_delta_action": False,
                "point_cloud_dirname": "points_frontview_polar_filled9",
                "point_feature_mode": "xyzrgb_polar",
                "state_action_norm_file": str(self.stats_path),
            }],
        }))

    def run_prepare(self, *extra_args):
        return subprocess.run(
            [sys.executable, str(PREPARE), str(self.config_path), str(self.root / "runtime"),
             *map(str, extra_args)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )

    def test_preserves_state_and_makes_classifier_actions_raw(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        output_config = yaml.safe_load(Path(result.stdout.strip()).read_text())
        dataset = output_config["lerobot_datasets"][0]
        self.assertEqual(dataset["point_cloud_dirname"], "points_frontview_polar_filled9")
        corrected = json.loads(Path(dataset["state_action_norm_file"]).read_text())
        self.assertEqual(corrected["state_mean"], self.stats["state_mean"])
        self.assertEqual(corrected["state_std"], self.stats["state_std"])
        self.assertEqual(corrected["action_mean"], [0.0] * 7)
        self.assertEqual(corrected["action_std"], [1.0] * 7)
        self.assertEqual(json.loads(self.stats_path.read_text()), self.stats)

        # The published stats turn gripper=0 into a negative BCE target;
        # raw classifier labels stay in [0, 1].
        normalized_target = -self.stats["action_mean"][6] / self.stats["action_std"][6]
        self.assertLess(normalized_target, 0)
        logit = -10.0
        invalid_bce = max(logit, 0) - logit * normalized_target + math.log1p(math.exp(-abs(logit)))
        self.assertLess(invalid_bce, 0)
        self.assertEqual((0 - corrected["action_mean"][6]) / corrected["action_std"][6], 0)
        self.assertEqual((1 - corrected["action_mean"][6]) / corrected["action_std"][6], 1)

    def test_rejects_bad_gripper_range_without_writing_output(self):
        self.stats["action_max"][6] = 2.0
        self.stats_path.write_text(json.dumps(self.stats))
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw gripper labels", result.stderr)
        self.assertFalse((self.root / "runtime").exists())

    def test_refuses_to_overwrite_existing_runtime_config(self):
        self.assertEqual(self.run_prepare().returncode, 0)
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)

    def test_accepts_v2_incomplete9_point_cloud_config(self):
        config = yaml.safe_load(self.config_path.read_text())
        config["lerobot_datasets"][0]["point_cloud_dirname"] = "points_frontview_polar_incomplete9"
        self.config_path.write_text(yaml.safe_dump(config))
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        output_config = yaml.safe_load(Path(result.stdout.strip()).read_text())
        self.assertEqual(
            output_config["lerobot_datasets"][0]["point_cloud_dirname"],
            "points_frontview_polar_incomplete9",
        )

    def test_accepts_xyz_polar_features_from_filled9(self):
        config = yaml.safe_load(self.config_path.read_text())
        config["lerobot_datasets"][0]["point_feature_mode"] = "xyz_polar"
        self.config_path.write_text(yaml.safe_dump(config))

        result = self.run_prepare()

        self.assertEqual(result.returncode, 0, result.stderr)
        output_config = yaml.safe_load(Path(result.stdout.strip()).read_text())
        self.assertEqual(
            output_config["lerobot_datasets"][0]["point_feature_mode"],
            "xyz_polar",
        )

    def test_accepts_xyzrgb_control_from_filled9(self):
        config = yaml.safe_load(self.config_path.read_text())
        config["lerobot_datasets"][0]["point_feature_mode"] = "xyzrgb"
        self.config_path.write_text(yaml.safe_dump(config))

        result = self.run_prepare()

        self.assertEqual(result.returncode, 0, result.stderr)
        output_config = yaml.safe_load(Path(result.stdout.strip()).read_text())
        self.assertEqual(output_config["lerobot_datasets"][0]["point_feature_mode"], "xyzrgb")

    def test_mounted_dataset_root_rewrites_data_and_stats_paths(self):
        dataset_root = self.root / "polar-dataset"
        (dataset_root / "meta").mkdir(parents=True)
        (dataset_root / "meta/info.json").write_text("{}")
        stats_dir = dataset_root / "robot_state_action_stats"
        stats_dir.mkdir()
        mounted_stats = stats_dir / self.stats_path.name
        mounted_stats.write_text(json.dumps(self.stats))

        result = self.run_prepare("--dataset-root", dataset_root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output_config = yaml.safe_load(Path(result.stdout.strip()).read_text())
        dataset = output_config["lerobot_datasets"][0]
        self.assertEqual(Path(dataset["root"]), dataset_root.parent)
        corrected = json.loads(Path(dataset["state_action_norm_file"]).read_text())
        self.assertEqual(corrected["action_mean"], [0.0] * 7)
        self.assertIn(str(mounted_stats), result.stderr)


if __name__ == "__main__":
    unittest.main()
