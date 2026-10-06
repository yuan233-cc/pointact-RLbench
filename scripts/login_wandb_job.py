"""Hidden prompt into an explicitly selected temporary Job-local NETRC."""
import getpass
import os
from pathlib import Path
import wandb

path = Path(os.environ["NETRC"])
if not str(path).startswith("/tmp/yuan/credentials/job-"):
    raise ValueError("NETRC must be Job-local")
path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
os.umask(0o077)
key = getpass.getpass("W&B API key: ")
try:
    wandb.login(key=key, relogin=True, verify=True)
finally:
    key = None
os.chmod(path, 0o600)
viewer = wandb.Api().viewer
username, entity = viewer.username, viewer.entity
if "weihang-li" in (username, entity):
    raise RuntimeError("Prohibited W&B account")
print(f"Verified W&B viewer={username}, entity={entity}", flush=True)
