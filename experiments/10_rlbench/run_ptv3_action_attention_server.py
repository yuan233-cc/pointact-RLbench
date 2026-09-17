"""PointACT server that captures true action-token to point-token attention."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import tyro

from scripts.run_server import Policy
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from ptv3_action_attention import ActionAttentionCapture


@dataclasses.dataclass
class Args:
    seed: int = 7
    pretrained_path: str = ""
    host: str = "127.0.0.1"
    port: int = 5555
    num_denoise_steps: int = 10
    save_data: bool = False
    save_dir: str = ""
    capture_dir: str = "ptv3_action_attention_captures"
    max_captures: int = 1


class AttentionPolicy(Policy):
    def __init__(self, args: Args):
        super().__init__(args)
        self._latest_scene_center = None
        original_center = self.processor._center_point_cloud_and_state

        def record_scene_center(*center_args, **center_kwargs):
            result = original_center(*center_args, **center_kwargs)
            self._latest_scene_center = result[2].detach().float().cpu().numpy()
            return result

        self.processor._center_point_cloud_and_state = record_scene_center
        self.capture = ActionAttentionCapture(
            Path(args.capture_dir), max_captures=args.max_captures
        )
        outer_ptv3 = self.model.ptv3_model
        outer_ptv3.register_forward_pre_hook(self.capture.ptv3_pre_hook)

        def save_scene_center(_module, _inputs):
            if self.capture.pending_arrays is not None and self._latest_scene_center is not None:
                self.capture.pending_arrays["scene_center"] = self._latest_scene_center.copy()

        outer_ptv3.register_forward_pre_hook(save_scene_center)
        outer_ptv3.register_forward_hook(self.capture.ptv3_hook)

        backbone = outer_ptv3.ptv3_model
        pooling_modules = sorted(
            (
                (name, module)
                for name, module in backbone.named_modules()
                if name.startswith("enc.") and name.endswith(".down")
            ),
            key=lambda item: item[0],
        )
        for stage, (_name, module) in enumerate(pooling_modules):
            module.register_forward_hook(self.capture.make_pooling_hook(stage))

        attention_modules = [
            (name, module)
            for name, module in backbone.named_modules()
            if name.startswith("enc.enc")
            and ".block" in name
            and name.endswith(".attn")
            and module.__class__.__name__ == "SerializedAttentionWithAction"
        ]
        skip_state_token = bool(self.model.config.use_robot_state)
        for name, module in attention_modules:
            module.register_forward_pre_hook(
                self.capture.make_attention_pre_hook(name, skip_state_token)
            )
            module.register_forward_hook(
                self.capture.make_attention_post_hook(name)
            )
        self.model.action_head.register_forward_hook(self.capture.action_head_hook)
        print("Action-to-point attention hooks:", [name for name, _ in attention_modules])
        print("Skip leading robot-state token:", skip_state_token)

    def get_action(self, batch, options):
        self.capture.begin_request(batch)
        return super().get_action(batch, options)


def main(args: Args) -> None:
    set_seed(args.seed)
    policy = AttentionPolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
