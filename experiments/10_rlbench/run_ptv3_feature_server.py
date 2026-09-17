"""PointACT inference server with non-invasive PTV3 feature capture."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from scripts.run_server import Policy
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from ptv3_feature_viz import FeatureCapture


@dataclasses.dataclass
class Args:
    seed: int = 7
    pretrained_path: str = ""
    host: str = "127.0.0.1"
    port: int = 5555
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = ""
    capture_dir: str = "ptv3_feature_captures"
    capture_every: int = 1
    max_captures: int = 0
    save_input_points: bool = True


class FeaturePolicy(Policy):
    def __init__(self, args: Args):
        super().__init__(args)
        self.capture = FeatureCapture(
            Path(args.capture_dir),
            capture_every=args.capture_every,
            max_captures=args.max_captures,
            save_input_points=args.save_input_points,
        )
        if not hasattr(self.model, "ptv3_model"):
            raise TypeError(f"{type(self.model).__name__} has no ptv3_model to visualize")
        self._feature_pre_hook = self.model.ptv3_model.register_forward_pre_hook(
            self.capture.ptv3_pre_hook
        )
        self._feature_hook = self.model.ptv3_model.register_forward_hook(
            self.capture.ptv3_hook
        )
        ptv3_backbone = self.model.ptv3_model.ptv3_model
        pooling_modules = sorted(
            (
                (name, module)
                for name, module in ptv3_backbone.named_modules()
                if name.startswith("enc.") and name.endswith(".down")
            ),
            key=lambda item: item[0],
        )
        self._pooling_hooks = [
            module.register_forward_hook(self.capture.make_pooling_hook(stage_index))
            for stage_index, (_name, module) in enumerate(pooling_modules)
        ]
        print(
            "PTV3 pooling hooks:",
            [name for name, _module in pooling_modules],
        )
        if not hasattr(self.model, "action_head"):
            raise TypeError(f"{type(self.model).__name__} has no action_head to visualize")
        self._action_head_hook = self.model.action_head.register_forward_hook(
            self.capture.action_head_hook
        )
        print(f"PTV3 feature capture directory: {self.capture.output_dir}")

    def get_action(self, batch, options):
        self.capture.begin_request(batch)
        return super().get_action(batch, options)


def main(args: Args) -> None:
    set_seed(args.seed)
    policy = FeaturePolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
