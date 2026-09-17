"""PointACT policy server whose reset endpoint can also reset all RNGs."""

from __future__ import annotations

import tyro

from scripts.run_server import Policy, ServerArgs
from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed


class ReseedablePolicy(Policy):
    def reset(self, options=None):
        options = options or {}
        if "seed" in options:
            set_seed(int(options["seed"]))
        return super().reset(options=options)


def main(args: ServerArgs) -> None:
    set_seed(args.seed)
    policy = ReseedablePolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
