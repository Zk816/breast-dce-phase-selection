"""Seed-average the per-case test Dice of every phase subset.

Each subset was trained with several seeds. A comparison between subsets must not
inherit seed noise from either side, so per-case Dice is averaged over seeds FIRST
and the paired bootstrap then resamples cases only.

    python scripts/phase_sel_pool.py [--cohort ISPY2] [--min_seeds 5]

Writes results/phase_sel_report/pooled/p<subset>.csv (case,dice) plus a coverage
table, and the two class-level pools used by the headline test:
POSTONLY (p12,p13,p23) and PRECONT (p01,p02,p03).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import re
from pathlib import Path

OUT = Path("results/phase_sel_report/pooled")
CLASSES = {"POSTONLY": ["p12", "p13", "p23"], "PRECONT": ["p01", "p02", "p03"]}


def read(path):
    with open(path) as f:
        return {r["case"].split(".")[0]: float(r["dice"]) for r in csv.DictReader(f)}


def write(name, table):
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice"])
        for c in sorted(table):
            w.writerow([c, f"{table[c]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="ISPY2")
    ap.add_argument("--min_seeds", type=int, default=2)
    args = ap.parse_args()
    global OUT
    OUT = OUT.parent / ("pooled" if args.cohort == "ISPY2" else f"pooled_{args.cohort}")
    OUT.mkdir(parents=True, exist_ok=True)

    runs = collections.defaultdict(list)
    for p in glob.glob(f"results/phasesel_*_s2_p*_{args.cohort}_seed*/per_case.csv"):
        runs[re.search(r"_s2_(p\d+)_", p).group(1)].append(p)

    kept, cov = {}, []
    for sub, paths in sorted(runs.items()):
        seeds = sorted(int(re.search(r"seed(\d+)", p).group(1)) for p in paths)
        tables = [read(p) for p in paths]
        cases = set.intersection(*[set(t) for t in tables])
        if len(seeds) < args.min_seeds:
            cov.append((sub, len(seeds), len(cases), "SKIPPED"))
            continue
        kept[sub] = {c: sum(t[c] for t in tables) / len(tables) for c in cases}
        write(sub, kept[sub])
        cov.append((sub, len(seeds), len(cases), ",".join(map(str, seeds))))

    for name, subs in CLASSES.items():
        have = [kept[s] for s in subs if s in kept]
        if len(have) != len(subs):
            continue
        cases = set.intersection(*[set(t) for t in have])
        write(name, {c: sum(t[c] for t in have) / len(have) for c in cases})

    with (OUT / "coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n_seeds", "n_cases", "seeds"])
        w.writerows(cov)
    print(f"{'subset':8s} {'seeds':>5s} {'cases':>6s}  seeds")
    for r in cov:
        print(f"{r[0]:8s} {r[1]:5d} {r[2]:6d}  {r[3]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
