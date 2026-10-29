"""Confirmatory statistics for the phase-subset sweep, on the seed-pooled per-case Dice.

Three things the manuscript needs and a single paired bootstrap cannot give:

  1. RANKING with multiplicity. Six pairs admit 15 comparisons. Reporting the largest
     without correction is the classic way to publish a difference that is not there,
     so every non-pre-registered comparison carries a Holm-adjusted p-value.
  2. EQUIVALENCE as a curve. "TOST fails at delta=1.5" invites the reader to guess what
     would have passed. The smallest delta that passes is a number, so report it.
  3. The CLASS-level test (post-contrast-only vs pre-contrast-containing), which is the
     claim the manuscript actually makes.

TOST at alpha uses the (1-2*alpha) interval, so the 90% interval is the one checked
against +-delta, while the reported CI stays 95%.

    python scripts/phase_sel_pool.py && python scripts/phase_sel_stats.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

POOL = Path("results/phase_sel_report/pooled")
OUT = Path("results/phase_sel_report")
PAIRS = ["p01", "p02", "p03", "p12", "p13", "p23"]
PREREG = ("p12", "p01")


def read(name):
    with (POOL / f"{name}.csv").open() as f:
        return {r["case"]: float(r["dice"]) for r in csv.DictReader(f)}


def boot(a, b, n_boot=10000, seed=42):
    cases = sorted(set(a) & set(b))
    d = np.array([a[c] - b[c] for c in cases]) * 100
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(diff=d.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                lo90=np.percentile(bs, 5), hi90=np.percentile(bs, 95),
                p=min(max(p, 1 / n_boot), 1.0), n=len(cases))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    have = {n: read(n) for n in PAIRS + ["p0123", "POSTONLY", "PRECONT"] if (POOL / f"{n}.csv").exists()}

    print("=" * 78)
    print("1. SUBSET RANKING (seed-pooled per-case test Dice)")
    print("=" * 78)
    for n in sorted(have, key=lambda k: -np.mean(list(have[k].values()))):
        v = list(have[n].values())
        print(f"  {n:10s} n={len(v):3d}  Dice {np.mean(v):.4f}")

    print("\n" + "=" * 78)
    print("2. ALL PAIRWISE COMPARISONS AMONG THE SIX PAIRS (Holm-adjusted)")
    print("=" * 78)
    combos = [c for c in itertools.combinations(PAIRS, 2) if c[0] in have and c[1] in have]
    res = [boot(have[a], have[b], args.n_boot) | {"a": a, "b": b} for a, b in combos]
    for r, adj in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = adj
    res.sort(key=lambda r: -abs(r["diff"]))
    print(f"  {'A':6s} {'B':6s} {'diff':>7s} {'95% CI':>18s} {'p':>8s} {'p_Holm':>8s}  sig")
    for r in res:
        tag = "PRE-REGISTERED" if (r["a"], r["b"]) == PREREG else ("*" if r["p_holm"] < 0.05 else "")
        print(f"  {r['a']:6s} {r['b']:6s} {r['diff']:+7.2f} [{r['lo']:+7.2f},{r['hi']:+7.2f}] "
              f"{r['p']:8.4f} {r['p_holm']:8.4f}  {tag}")
    with (OUT / "comparisons_holm.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "n", "diff", "lo", "hi", "p", "p_holm"],
                           extrasaction="ignore")
        w.writeheader(); w.writerows(res)

    print("\n" + "=" * 78)
    print("3. EQUIVALENCE CURVE  k=2 (p12) vs k=4 (p0123)")
    print("=" * 78)
    rows = []
    if "p12" in have and "p0123" in have:
        r = boot(have["p12"], have["p0123"], args.n_boot)
        print(f"  difference {r['diff']:+.2f} pts   95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]"
              f"   90% CI [{r['lo90']:+.2f}, {r['hi90']:+.2f}]")
        print(f"  {'delta':>6s}  TOST equivalence at alpha=0.05")
        smallest = None
        for delta in np.arange(0.5, 3.01, 0.25):
            ok = r["lo90"] > -delta and r["hi90"] < delta
            rows.append(dict(delta=round(float(delta), 2), equivalent=int(ok)))
            if ok and smallest is None:
                smallest = delta
            print(f"  {delta:6.2f}  {'PASS' if ok else 'fail'}")
        print(f"\n  smallest delta that passes: {smallest if smallest else 'none up to 3.0'}")
        with (OUT / "tost_curve.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["delta", "equivalent"])
            w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print("4. CLASS-LEVEL TEST  post-contrast-only vs pre-contrast-containing")
    print("=" * 78)
    if "POSTONLY" in have and "PRECONT" in have:
        r = boot(have["POSTONLY"], have["PRECONT"], args.n_boot)
        print(f"  post-only      {np.mean(list(have['POSTONLY'].values())):.4f}")
        print(f"  pre-containing {np.mean(list(have['PRECONT'].values())):.4f}")
        print(f"  difference {r['diff']:+.2f} pts  95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]  p={r['p']:.4f}")
    print(f"\n-> {OUT}/comparisons_holm.csv, {OUT}/tost_curve.csv")


if __name__ == "__main__":
    main()
