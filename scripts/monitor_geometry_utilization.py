"""Monitor active training only; exclude cache/init/validation/checkpoint phases."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True)
    p.add_argument("--supervisor", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if not a.job.isdigit() or a.job != os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Guard may only monitor its own numeric allocation")
    phase_file = a.output_dir / "phase.json"
    metrics_file = a.output_dir / "metrics.jsonl"
    def phase():
        try:
            return json.loads(phase_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    def alive():
        try:
            os.kill(a.supervisor, 0)
            return True
        except ProcessLookupError:
            return False
    window = []
    while alive():
        before = phase()
        time.sleep(5)
        after = phase()
        # A phase changed during sampling, or this is not steady training.
        if before != after or after.get("phase") != "train" or after.get("step", 0) < 10:
            continue
        if not metrics_file.exists():
            continue
        last = json.loads(metrics_file.read_text().splitlines()[-1])
        if last["step"] < 10:
            continue
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True)
        devices = [[float(x) for x in line.split(",")] for line in raw.strip().splitlines()]
        window.append(devices)
        if len(window) < 60:
            continue
        averages = [dict(compute_percent=sum(row[i][0] for row in window)/len(window),
            memory_percent=sum(row[i][1]/row[i][2]*100 for row in window)/len(window))
            for i in range(len(devices))]
        print("STEADY_WINDOW="+json.dumps(dict(job=a.job, active_seconds=300, step=last["step"], devices=averages)), flush=True)
        if any(r["compute_percent"] < 50 or r["memory_percent"] < 15 for r in averages):
            owner = subprocess.check_output(["scontrol", "show", "job", "-o", a.job], text=True)
            if f"({os.getuid()})" not in owner.split("UserId=", 1)[1].split()[0]:
                raise RuntimeError("Job ownership changed")
            print(f"LOW_UTILIZATION_STOP job={a.job}; unsaved progress may be lost; no auto restart", flush=True)
            os.kill(a.supervisor, signal.SIGTERM)
            return
        window = []


if __name__ == "__main__":
    main()
