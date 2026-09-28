"""Serve a filled9 reconstruction-trained checkpoint in action-only eval mode."""

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
    port: int = 15557
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = "/tmp/pointact_filled9_reconstruction_action_only_inputs"


def _forbid_auxiliary_forward(module, inputs):
    raise RuntimeError(
        f"Action-only evaluation unexpectedly invoked auxiliary module {type(module).__name__}"
    )


def main(args: Args) -> None:
    checkpoint = Path(args.pretrained_path)
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("architectures") != ["VLAEncDec3DWithActionClassificationModel"]:
        raise ValueError(f"Unexpected architecture: {config.get('architectures')}")
    if config.get("ptv3_input_channels") != 9:
        raise ValueError("Checkpoint is not configured for nine-channel filled9 input")
    if not config.get("use_target_reconstruction"):
        raise ValueError("Checkpoint does not enable target reconstruction training")

    set_seed(args.seed)
    policy = Policy(args)
    if policy.model.training:
        raise RuntimeError("Policy must be in eval mode")
    policy.processor._prepare_point_cloud_for_sample = MethodType(
        _prepare_filled9, policy.processor
    )

    # Reconstruction supervision is training-only. These guards prove that an
    # RLBench action request never executes either auxiliary module.
    policy.model.target_reconstruction_head.register_forward_pre_hook(
        _forbid_auxiliary_forward
    )
    policy.model.ptv3_model.ptv3_model.dec.register_forward_pre_hook(
        _forbid_auxiliary_forward
    )
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    main(tyro.cli(Args))
