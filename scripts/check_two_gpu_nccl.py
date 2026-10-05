#!/usr/bin/env python3
"""Bounded two-rank NCCL all-reduce check before DDP training."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two visible GPUs/ranks, got {torch.cuda.device_count()}/{world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        tensor = torch.tensor([float(local_rank + 1)], device=f"cuda:{local_rank}")
        dist.all_reduce(tensor)
        if tensor.item() != 3.0:
            raise RuntimeError(f"bad all-reduce result on rank {local_rank}: {tensor.item()}")
        print(f"rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)} sum={tensor.item()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
