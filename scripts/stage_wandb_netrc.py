#!/usr/bin/env python3
"""Stage a W&B credential in a job-local NETRC without echoing the key."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import stat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netrc", type=Path, required=True)
    args = parser.parse_args()
    path = args.netrc.resolve()
    if not str(path).startswith("/tmp/yuan/credentials/job-"):
        parser.error("NETRC must be in a job-local /tmp/yuan/credentials/job-* directory")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    os.environ["NETRC"] = str(path)

    import wandb

    key = getpass.getpass("W&B API key (hidden): ")
    try:
        if not wandb.login(key=key, relogin=True, verify=True):
            raise RuntimeError("W&B login did not verify")
    finally:
        key = None
    if not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise RuntimeError("job-local NETRC is missing or not mode 0600")
    viewer = wandb.Api().viewer
    username, entity = viewer.username, viewer.entity
    if not username or not entity or "weihang-li" in (username, entity):
        raise RuntimeError("W&B viewer or entity is missing or prohibited")
    print(json.dumps({"wandb_username": username, "wandb_entity": entity,
                      "netrc_mode": "0600"}), flush=True)


if __name__ == "__main__":
    main()
