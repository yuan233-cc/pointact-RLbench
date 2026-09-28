"""Episode-consistent incomplete-point server with legacy inference RNG semantics.

The client notifies the server when an episode starts so the structured missing
region can be rebuilt for that episode. Unlike the controlled-RNG evaluation,
this reset deliberately does not reseed Python, NumPy, or PyTorch. Model and
point-subsampling RNG streams are seeded only once when the server starts.
"""

from __future__ import annotations

import tyro

from pointact.utils.server_client import PolicyServer
from pointact.utils.torch_utils import set_seed
from scripts.run_server import Policy
from run_reseedable_incomplete25_server import Args, ReseedableIncompletePolicy


class EpisodeConsistentIncompletePolicy(ReseedableIncompletePolicy):
    def reset(self, options=None):
        options = options or {}
        self.episode_id = int(options.get("episode_id", 0))
        self.hole_field = None
        # Reset bookkeeping only. Do not call the parent implementation because
        # it reseeds every RNG and changes the legacy inference protocol.
        return Policy.reset(self, options=options)


def main(args: Args) -> None:
    set_seed(args.seed)
    policy = EpisodeConsistentIncompletePolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
