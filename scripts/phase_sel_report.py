"""Stage-1 selection readout (exp_phase_sel.md LOGGING + SECOND FIGURE).

    python scripts/phase_sel_report.py [--glob 'phasesel_*_s1_*']

Reads every ``results/{run}/selection.json`` written at the end of a Stage-1 run
and reports, per k: the selection of each seed, cross-seed agreement (exact match
rate and mean Jaccard), and the acquisition times IN SECONDS of the selected
phases pooled over cases (phase index means different things at different sites;
seconds are physically comparable). Also dumps per-case phase availability.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/path/to/MAMA-MIA_v3"
OUT = Path("results/phase_sel_report")


def load_runs(pattern):
    runs = []
    for p in sorted(Path("results").glob(f"{pattern}/selection.json")):
        d = json.loads(p.read_text())
        d["run"] = p.parent.name
        runs.append(d)
    return runs


def agreement(sels):
    pairs = list(combinations(sels, 2))
    if not pairs:
        return float("nan"), float("nan")
    exact = np.mean([set(a) == set(b) for a, b in pairs])
    jac = np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in pairs])
    return float(exact), float(jac)


def case_table(num_phases, cohorts):
    meta = json.loads(Path(ROOT, "meta.json").read_text())
    rows = []
    for pid, m in sorted(meta.items()):
        t = m["acquisition_times"]
        if pid.split("_")[0] not in cohorts:
            continue
        rows.append({"pid": pid, "cohort": pid.split("_")[0], "K_available": len(t),
                     "eligible": len(t) >= num_phases,
                     "times_s": ";".join(str(int(x)) for x in t)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="phasesel_*_s1_*")
    ap.add_argument("--cohorts", default="DUKE,ISPY1,ISPY2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cohorts = a.cohorts.split(",")

    runs = load_runs(a.glob)
    if not runs:
        print(f"no results/{a.glob}/selection.json found — Stage-1 runs not finished yet")
        return
    num_phases = runs[0]["num_phases"]
    cases = case_table(num_phases, cohorts)
    with (OUT / "case_phases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0]))
        w.writeheader()
        w.writerows(cases)

    lines = []
    times_rows = []
    for k in sorted({r["k"] for r in runs}):
        rk = [r for r in runs if r["k"] == k]
        lines.append(f"\n=== k = {k} ({len(rk)} runs) ===")
        for r in rk:
            theta = " ".join(f"{v:+.3f}" for v in r["theta"])
            lines.append(f"  {r['run']:42s} selected {r['selected_indices']}  theta [{theta}]")
        ex, ja = agreement([r["selected_indices"] for r in rk])
        lines.append(f"  cross-seed agreement: exact match {ex:.2f}, mean Jaccard {ja:.2f}")

        for coh in cohorts:
            sec = [float(c["times_s"].split(";")[i])
                   for c in cases if c["cohort"] == coh and c["eligible"]
                   for r in rk for i in r["selected_indices"]]
            if sec:
                lines.append(f"  {coh:6s} selected acquisition time s: "
                             f"median {np.median(sec):6.1f}  IQR "
                             f"[{np.percentile(sec, 25):.0f}, {np.percentile(sec, 75):.0f}]  n={len(sec)}")
                times_rows += [{"k": k, "cohort": coh, "time_s": s} for s in sec]

    with (OUT / "selected_times.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "cohort", "time_s"])
        w.writeheader()
        w.writerows(times_rows)

    ks = sorted({r["k"] for r in runs})
    fig, axes = plt.subplots(1, len(ks), figsize=(6 * len(ks), 4), squeeze=False)
    for ax, k in zip(axes[0], ks):
        for coh in cohorts:
            s = [r["time_s"] for r in times_rows if r["k"] == k and r["cohort"] == coh]
            if s:
                ax.hist(s, bins=40, alpha=0.5, label=f"{coh} (n={len(s)})")
        ax.set_xlabel("acquisition time of selected phase (s post-injection)")
        ax.set_ylabel("count")
        ax.set_title(f"learned selection, k={k}")
        ax.legend()
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/phase_sel_selected_times.png", dpi=150)

    report = "\n".join(lines)
    (OUT / "summary.txt").write_text(report + "\n")
    print(report)
    print(f"\nwrote {OUT}/summary.txt, selected_times.csv, case_phases.csv "
          f"and results/figures/phase_sel_selected_times.png")


if __name__ == "__main__":
    main()
