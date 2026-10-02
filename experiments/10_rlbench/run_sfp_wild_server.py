"""Serve an SfP-Wild + PointACT checkpoint for live RLBench evaluation."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import MethodType

import tyro

from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from run_filled9_server import _prepare_filled9
from scripts.run_server import Policy


@dataclasses.dataclass
class Args:
    pretrained_path: str
    seed: int = 7
    host: str = "127.0.0.1"
    port: int = 15570
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = "/tmp/pointact_sfp_wild_server_inputs"


def validate_checkpoint(checkpoint: Path) -> dict:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())
    expected_architecture = ["VLAEncDec3DWithActionRegressionModel"]
    if config.get("architectures") != expected_architecture:
        raise ValueError(
            "SfP RLBench evaluation expects the action-regression checkpoint, "
            f"got {config.get('architectures')}"
        )
    if config.get("ptv3_input_channels") != 9:
        raise ValueError("SfP checkpoint must use incomplete XYZRGB+polar (nine channels)")
    if not config.get("polar_enabled") or config.get("polar_backbone") != "sfp_wild":
        raise ValueError("Checkpoint must enable the SfP-Wild polar branch")
    if not config.get("use_polar_depth_self_supervision"):
        raise ValueError("Checkpoint must contain the trained depth-completion action adapter")
    return config


def main(args: Args) -> None:
    validate_checkpoint(Path(args.pretrained_path))
    set_seed(args.seed)
    policy = Policy(args)
    if policy.model.training:
        raise RuntimeError("Policy must be in evaluation mode")

    # The client already applies the dataset's voxelization and corruption and
    # supplies nine features. Preserve them instead of passing through Open3D,
    # whose generic path keeps only XYZRGB.
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_filled9, policy.processor
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
