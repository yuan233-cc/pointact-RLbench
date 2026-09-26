from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments/10_rlbench/data_configs"


class PolarAblationConfigTests(unittest.TestCase):
    def test_three_classifier_inputs_share_every_non_feature_setting(self):
        names = {
            "xyzrgb": "data-10task-rlbench-filled6-clf-no-rot.yaml",
            "xyzrgb_polar": "data-10task-polar-rlbench9-v2-filled.yaml",
            "xyz_polar": "data-10task-xyzpolar-filled6.yaml",
        }
        datasets = {
            mode: yaml.safe_load((CONFIG_DIR / filename).read_text())["lerobot_datasets"][0]
            for mode, filename in names.items()
        }

        baseline = datasets["xyzrgb"]
        baseline_common = {
            key: value
            for key, value in baseline.items()
            if key not in {
                "augment_point_color",
                "point_feature_mode",
                "polar_feature_normalization",
            }
        }
        for mode, dataset in datasets.items():
            with self.subTest(mode=mode):
                self.assertEqual(dataset["point_feature_mode"], mode)
                common = {
                    key: value
                    for key, value in dataset.items()
                    if key not in {
                        "augment_point_color",
                        "point_feature_mode",
                        "polar_feature_normalization",
                    }
                }
                self.assertEqual(common, baseline_common)

        self.assertTrue(baseline["augment_point_color"])
        self.assertTrue(datasets["xyzrgb_polar"]["augment_point_color"])
        self.assertFalse(datasets["xyz_polar"]["augment_point_color"])
        self.assertEqual(baseline["augment_pc_rot"], 0)
        self.assertEqual(baseline["video_key_ids_for_vlm"], [])
        self.assertEqual(datasets["xyzrgb_polar"]["polar_feature_normalization"], "rgb")
        self.assertEqual(datasets["xyz_polar"]["polar_feature_normalization"], "rgb")


if __name__ == "__main__":
    unittest.main()
