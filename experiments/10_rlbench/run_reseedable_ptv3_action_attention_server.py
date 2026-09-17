"""Attention-capture PointACT server with deterministic request resets."""

from __future__ import annotations

import tyro

from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from run_ptv3_action_attention_server import Args, AttentionPolicy


class ReseedableAttentionPolicy(AttentionPolicy):
    def reset(self, options=None):
        options = options or {}
        if "seed" in options:
            set_seed(int(options["seed"]))
        return super().reset(options=options)


def main(args: Args) -> None:
    set_seed(args.seed)
    policy = ReseedableAttentionPolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
