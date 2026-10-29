"""Unattended dispatcher for the phase-selection sweep."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

GPUS = [0, 1, 2, 3, 4, 5]
MAX_CONCURRENT = 6
MAX_PER_GPU = 1
MAX_ATTEMPTS = 2
TARGET_EPOCHS = 200
MIN_FREE_GB = 60
KEEP_PREDICTIONS = False
POLL = 300
GT = "/path/to/MAMA-MIA_v3/labelsTr"
RUN_RE = re.compile(r"^\s*(\d+)\s+.*-m bcaitu_mri_seg fit --config experiments/(phasesel_\S+)\.yaml")


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def epochs_done(run):
    d = Path("tb_logs", run)
    vs = sorted(d.glob("version_*")) if d.is_dir() else []
    if not vs:
        return 0
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(vs[-1]))
        ea.Reload()
        return len(ea.Scalars("val_dice")) if "val_dice" in ea.Tags()["scalars"] else 0
    except Exception:
        return 0


def active_runs():
    """{run_name: main_pid} for every phasesel training process currently alive."""
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    seen = {}
    for line in out.splitlines():
        m = RUN_RE.search(line)
        if m:
            pid, run = int(m.group(1)), m.group(2)
            seen.setdefault(run, []).append(pid)
    return {r: min(p) for r, p in seen.items()}


def gpu_of(pid):
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().decode("utf8", "ignore")
        kv = dict(x.split("=", 1) for x in env.split("\0") if "=" in x)
        return kv.get("CUDA_VISIBLE_DEVICES")
    except Exception:
        return None


def gpu_load():
    c = Counter({str(g): 0 for g in GPUS})
    for _, pid in active_runs().items():
        g = gpu_of(pid)
        if g is not None:
            c[g] += 1
    return c


def free_gb():
    return shutil.disk_usage(".").free // (1024 ** 3)


def launch(run, gpu):
    log(f"LAUNCH {run} on GPU{gpu}")
    with open(f"logs/{run}.log", "w") as f:
        subprocess.Popen(
            ["python", "-u", "-m", "bcaitu_mri_seg", "fit",
             "--config", f"experiments/{run}.yaml", "--trainer.devices", "[0]"],
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
    time.sleep(30)


def main():
    queue = [l.strip() for l in Path("queue_remaining.txt").read_text().splitlines() if l.strip()]
    attempts = Counter()
    failed = Path("logs/dispatcher_failed.txt")
    log(f"=== dispatcher starting: {len(queue)} runs queued, "
        f"max {MAX_CONCURRENT} concurrent, target {TARGET_EPOCHS} epochs ===")

    while True:
        if Path("STOP_DISPATCHER").exists():
            log("STOP_DISPATCHER present -> leaving training loop (running jobs untouched)")
            break
        fg = free_gb()
        if fg < MIN_FREE_GB:
            log(f"PAUSED: {fg}G free < {MIN_FREE_GB}G. Not launching.")
            time.sleep(900)
            continue

        act = active_runs()
        pending = [r for r in queue
                   if epochs_done(r) < TARGET_EPOCHS
                   and r not in act
                   and attempts[r] < MAX_ATTEMPTS]
        for r in queue:
            if attempts[r] >= MAX_ATTEMPTS and epochs_done(r) < TARGET_EPOCHS:
                prev = failed.read_text().split() if failed.exists() else []
                if r not in prev:
                    with failed.open("a") as f:
                        f.write(r + "\n")
                    log(f"GIVING UP on {r} after {MAX_ATTEMPTS} attempts")

        while pending and len(active_runs()) < MAX_CONCURRENT:
            load = gpu_load()
            gpu, n = min(load.items(), key=lambda kv: kv[1])
            if n >= MAX_PER_GPU:
                break
            run = pending.pop(0)
            attempts[run] += 1
            launch(run, gpu)

        act = active_runs()
        remaining = sum(1 for r in queue if epochs_done(r) < TARGET_EPOCHS)
        log(f"active={len(act)} remaining={remaining} free={fg}G "
            f"gpus={dict(sorted(gpu_load().items()))}")
        if remaining == 0 and not act:
            log("ALL TRAINING COMPLETE")
            break
        time.sleep(POLL)

    log("=== predictions + evaluation ===")
    for cfg in sorted(Path("experiments").glob("phasesel_20260730_*.yaml")):
        run = cfg.stem
        if Path("results", run, "test_eval.json").exists():
            continue
        if epochs_done(run) < TARGET_EPOCHS:
            continue
        pred = Path("results", run, "pred")
        n = len(list(pred.glob("*.nii.gz"))) if pred.is_dir() else 0
        if n < 100:
            log(f"PREDICT {run}")
            r = subprocess.run(["python", "-u", "scripts/phase_sel_predict.py",
                                "--run", run, "--gpu", "0"],
                               env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"},
                               capture_output=True, text=True)
            if r.returncode:
                log(f"PREDICT FAILED {run}")
                continue
        log(f"EVAL {run}")
        r = subprocess.run(["python", "scripts/unified_eval.py",
                            "--pred_dir", str(pred), "--gt_dir", GT, "--hd95",
                            "--out", f"results/{run}/per_case.csv",
                            "--json", f"results/{run}/test_eval.json"],
                           capture_output=True, text=True)
        if r.returncode:
            log(f"EVAL FAILED {run} - keeping predictions for diagnosis")
            continue
        if not KEEP_PREDICTIONS and Path(f"results/{run}/per_case.csv").exists():
            shutil.rmtree(pred, ignore_errors=True)
            log(f"removed predictions for {run} (per_case.csv retained)")

    subprocess.run(["python", "scripts/phase_sel_manifest.py", "--csv"])
    log("=== DISPATCHER FINISHED ===")


if __name__ == "__main__":
    main()
