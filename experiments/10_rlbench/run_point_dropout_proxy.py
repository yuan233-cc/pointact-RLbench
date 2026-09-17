"""Transparent PointACT policy proxy that drops XYZ pixels before inference."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro

from pointact.utils.server_client import PolicyClient, PolicyServer


@dataclasses.dataclass
class Args:
    backend_host: str = "127.0.0.1"
    backend_port: int = 15495
    host: str = "127.0.0.1"
    port: int = 15496
    missing_rate: float = 0.0
    seed: int = 701
    log_file: str = "point_dropout_proxy.jsonl"


class PointDropoutPolicy:
    def __init__(self, args: Args):
        if not 0 <= args.missing_rate < 1:
            raise ValueError("missing_rate must be in [0, 1)")
        self.args = args
        self.backend = PolicyClient(args.backend_host, args.backend_port)
        if not self.backend.ping():
            raise RuntimeError("Backend policy server is not available")
        self.request_index = 0
        self.log_file = Path(args.log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text("", encoding="utf-8")

    def get_action(self, batch, options):
        modified = dict(batch)
        point_records = {}
        point_keys = sorted(
            key for key in batch if key.startswith("observation.points.")
        )
        for key_index, key in enumerate(point_keys):
            modified_values = []
            records = []
            for sample_index, value in enumerate(batch[key]):
                points = np.asarray(value).copy()
                if points.ndim < 2 or points.shape[-1] < 3:
                    raise ValueError(f"Unexpected point array shape for {key}: {points.shape}")
                spatial_shape = points.shape[:-1]
                rng_seed = (
                    self.args.seed
                    + 1_000_003 * self.request_index
                    + 10_007 * key_index
                    + sample_index
                )
                score = np.random.default_rng(rng_seed).random(spatial_shape)
                keep = score >= self.args.missing_rate
                points[..., :3][~keep] = np.nan
                modified_values.append(points)
                records.append({
                    "shape": list(value.shape),
                    "kept_pixels": int(keep.sum()),
                    "total_pixels": int(keep.size),
                    "kept_fraction": float(keep.mean()),
                    "rng_seed": int(rng_seed),
                })
            modified[key] = modified_values
            point_records[key] = records

        output = self.backend.get_action(modified, options=options)
        action = np.asarray(output.action)
        record = {
            "request_index": self.request_index,
            "missing_rate": self.args.missing_rate,
            "point_inputs": point_records,
            "first_predicted_position_world": (
                action[0, 0, :3].astype(float).tolist() if action.size else None
            ),
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.request_index += 1
        return output

    def reset(self, options=None):
        return self.backend.reset(options=options)


def main(args: Args) -> None:
    policy = PointDropoutPolicy(args)
    PolicyServer.start_server(policy, args.host, args.port)


if __name__ == "__main__":
    tyro.cli(main)
