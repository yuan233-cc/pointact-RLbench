#!/usr/bin/env python3
"""Download the authorized DINOv3 weight to a Slurm data node's job scratch.

The Hugging Face token is read from a mode-0600 job-local file, never from an
argument, environment variable, or persistent cache.
"""

import argparse
import hashlib
import os
from pathlib import Path

import requests


MODEL_URL = (
    "https://huggingface.co/facebook/"
    "dinov3-convnext-base-pretrain-lvd1689m/resolve/main/model.safetensors"
)
EXPECTED_SIZE = 350302312
EXPECTED_SHA256 = "ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if not args.job_id.isdecimal():
        raise SystemExit("job id must be numeric")

    credential = Path(f"/tmp/yuan/credentials/job-{args.job_id}/hf_token")
    scratch = Path(f"/tmp/yuan/dinov3-weight-job-{args.job_id}")
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = scratch / "model.safetensors"
    partial = scratch / "model.safetensors.partial"
    if destination.exists() or partial.exists():
        raise SystemExit("destination already exists; inspect before retrying")
    if credential.stat().st_mode & 0o077:
        raise SystemExit("credential file is accessible to other users")
    token = credential.read_text().strip()
    if not token.startswith("hf_"):
        raise SystemExit("invalid credential format")

    response = requests.get(
        MODEL_URL,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=(30, 120),
    )
    response.raise_for_status()
    digest = hashlib.sha256()
    count = 0
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        for block in response.iter_content(chunk_size=8 * 1024 * 1024):
            if not block:
                continue
            output.write(block)
            digest.update(block)
            count += len(block)
            if count // (50 * 1024 * 1024) != (count - len(block)) // (50 * 1024 * 1024):
                print(f"downloaded {count} bytes", flush=True)
        output.flush()
        os.fsync(output.fileno())
    response.close()
    if count != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        raise SystemExit(f"weight failed integrity check: bytes={count}, sha256={digest.hexdigest()}")
    partial.rename(destination)
    print(f"verified {destination}: bytes={count}, sha256={digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
