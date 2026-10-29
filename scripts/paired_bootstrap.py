"""Paired case-level bootstrap on two per-case Dice CSVs (day2.md GATE).

Two overlapping per-method CIs are NOT a test of a difference. Both methods are
scored on the SAME cases, so the correct test resamples CASES and recomputes the
mean of the per-case difference — which is also far tighter than the unpaired
comparison because case difficulty cancels.

Inputs are the `--out` CSVs written by scripts/unified_eval.py (case,dice).

    python scripts/paired_bootstrap.py --a results/<runA>/per_case.csv \
                                       --b results/<runB>/per_case.csv \
                                       [--min_diff 1.5]

Reports mean_A - mean_B in Dice POINTS (x100), its 95% CI, and whether the CI
excludes zero and clears --min_diff.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path):
    with Path(path).open() as f:
        return {r["case"]: float(r["dice"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="per-case CSV of method A (the claim)")
    ap.add_argument("--b", required=True, help="per-case CSV of method B (the comparator)")
    ap.add_argument("--label_a", default=None)
    ap.add_argument("--label_b", default=None)
    ap.add_argument("--min_diff", type=float, default=1.5, help="required gap in Dice points")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    na = args.label_a or Path(args.a).parent.name
    nb = args.label_b or Path(args.b).parent.name
    cases = sorted(set(A) & set(B))
    only = (set(A) ^ set(B))
    if only:
        print(f"WARNING: {len(only)} cases not in both files — paired on {len(cases)}")
    if not cases:
        msg = "no shared cases between the two CSVs"
        raise SystemExit(msg)

    a = np.array([A[c] for c in cases])
    b = np.array([B[c] for c in cases])
    d = a - b

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(cases), (args.n_boot, len(cases)))
    boots = d[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = d.mean() * 100.0

    print(f"n paired cases   = {len(cases)}")
    print(f"{na:34s} = {a.mean():.4f}")
    print(f"{nb:34s} = {b.mean():.4f}")
    print(f"paired difference = {mean:+.2f} Dice points  95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"A better on       = {float((d > 0).mean()):.1%} of cases  "
          f"(ties {float((d == 0).mean()):.1%})")
    excl = lo > 0 or hi < 0
    print(f"CI excludes zero  = {excl}")
    print(f"clears {args.min_diff:.1f} points  = {mean >= args.min_diff and lo > 0}")


if __name__ == "__main__":
    main()
