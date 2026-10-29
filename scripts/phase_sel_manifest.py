"""One index of every phase-selection run: what it is, where it is, what it scored.

    python scripts/phase_sel_manifest.py            # print the table
    python scripts/phase_sel_manifest.py --csv      # also write the CSV

Run names carry the FACT (`s2_p12` = fixed phases 1,2; `s1_k2` = selector, k=2).
The ROLE (learned / clinical / random / exhaustive / curve) is an interpretation
and lives only here, so a run name never has to be re-read as a claim.

Artifacts for run R are always at: experiments/R.yaml, logs/R.log, tb_logs/R/,
checkpoints/R/, results/R/{selection.json,pred/,per_case.csv}.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LEARNED = (1, 2)
ROLES = {(0, 1): "clinical (pre+post1)", (0, 3): "random control",
         (1, 2): "LEARNED (Stage-1 pick)", (0, 2): "exhaustive", (1, 3): "exhaustive",
         (2, 3): "exhaustive (seed44 pick)", (2,): "curve k=1", (1, 2, 3): "curve k=3",
         (0, 1, 2, 3): "reference: all phases"}


def parse(run: str):
    m = re.match(r"phasesel_(\d{8})_(s1|s2|s3)_(k\d|p\d+|sub\d+)(?:_([A-Z0-9]+))?_seed(\d+)$", run)
    if not m:
        return None
    date, stage, spec, cohort, seed = m.groups()
    cohort = cohort or "ISPY2"
    if stage == "s1":
        phases, k, role = None, int(spec[1:]), f"Stage 1 selector, k={spec[1:]}"
    elif spec.startswith("sub"):
        phases = tuple(int(c) for c in spec[3:])
        k, role = len(phases), f"subtraction: phases {spec[3:]} minus pre-contrast"
    else:
        phases = tuple(int(c) for c in spec[1:])
        k, role = len(phases), ROLES.get(phases, "")
    return dict(run=run, date=date, stage=stage, k=k, phases=phases, seed=int(seed),
                cohort=cohort, role=role)


def scalars(run: str):
    d = sorted(Path("tb_logs", run).glob("version_*")) if Path("tb_logs", run).is_dir() else []
    if not d:
        return {}
    ea = EventAccumulator(str(d[-1]))
    ea.Reload()
    t = ea.Tags()["scalars"]
    out = {}
    if "val_dice" in t:
        v = ea.Scalars("val_dice")
        out["epochs"] = len(v)
        out["best_val_dice"] = max(x.value for x in v)
    if "theta/0" in t:
        out["theta"] = [ea.Scalars(f"theta/{i}")[-1].value for i in range(4)]
    return out


def heldout(run: str):
    p = Path("results", run, "heldout_eval.json")
    return json.loads(p.read_text()) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    rows = []
    for cfg in sorted(Path("experiments").glob("phasesel_*.yaml")):
        r = parse(cfg.stem)
        if r is None:
            print(f"WARNING: {cfg.stem} does not match the naming scheme")
            continue
        r |= scalars(r["run"])
        ck = list(Path("checkpoints", r["run"]).glob("*.ckpt")) if Path("checkpoints", r["run"]).is_dir() else []
        npred = len(list(Path("results", r["run"], "pred").glob("*.nii.gz"))) if Path("results", r["run"], "pred").is_dir() else 0
        ep = r.get("epochs", 0)
        r["status"] = "DONE" if ep >= 200 else (f"running {ep}/200" if ep else "queued")
        r["ckpt"] = ck[0].name if ck else ""
        r["n_pred"] = npred
        r |= heldout(r["run"])
        rows.append(r)

    rows.sort(key=lambda r: (r["stage"], r["k"], str(r["phases"]), r["seed"]))
    hdr = f"{'RUN':38s} {'PHASES':10s} {'SEED':>4s} {'STATUS':13s} {'VAL_DICE':>8s} {'DUKE':>6s} {'PRED':>5s}  ROLE"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ph = "learned" if r["phases"] is None else ",".join(map(str, r["phases"]))
        vd = f"{r['best_val_dice']:.4f}" if "best_val_dice" in r else "—"
        ho = f"{r['mean_dice']:.4f}" if "mean_dice" in r else "—"
        print(f"{r['run']:38s} {ph:10s} {r['seed']:4d} {r['status']:13s} {vd:>8s} {ho:>6s} "
              f"{r['n_pred']:5d}  {r['role']}")

    if args.csv:
        out = Path("results/phase_sel_report/manifest.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = ["run", "date", "stage", "cohort", "k", "phases", "seed", "role", "status", "epochs",
                "best_val_dice", "mean_dice", "ci_lo", "ci_hi", "n_pred", "ckpt", "theta"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
