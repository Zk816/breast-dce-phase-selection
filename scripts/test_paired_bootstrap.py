"""Unit test for scripts/paired_bootstrap.py (exp_phase_sel_final.md §12 checklist)."""
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parent / "paired_bootstrap.py"


def write(path, pairs):
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dice", "hd95_mm"])
        for c, d in pairs:
            w.writerow([c, f"{d:.6f}", "nan"])


def run(a, b, extra=()):
    r = subprocess.run([sys.executable, str(SCRIPT), "--a", str(a), "--b", str(b), *extra],
                       capture_output=True, text=True, check=True)
    return r.stdout


def parse(out, key):
    for line in out.splitlines():
        if line.startswith(key):
            return line
    msg = f"{key!r} not in output:\n{out}"
    raise AssertionError(msg)


def main():
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    n = 200
    cases = [f"C_{i:04d}.nii.gz" for i in range(n)]
    difficulty = rng.uniform(0.2, 0.95, n)

    write(tmp / "a1.csv", zip(cases, np.clip(difficulty + 0.05, 0, 1)))
    write(tmp / "b1.csv", zip(cases, difficulty))
    out = run(tmp / "a1.csv", tmp / "b1.csv")
    line = parse(out, "paired difference")
    diff = float(line.split("=")[1].split("Dice")[0])
    assert abs(diff - 5.0) < 0.01, f"constant +5 offset not recovered: {line}"
    assert "CI excludes zero  = True" in out, out
    assert "A better on       = 100.0%" in out, out
    print(f"[OK] constant +5.0-point offset recovered exactly: {diff:+.3f}")

    write(tmp / "a2.csv", zip(cases, difficulty))
    out = run(tmp / "a2.csv", tmp / "b1.csv")
    diff = float(parse(out, "paired difference").split("=")[1].split("Dice")[0])
    assert abs(diff) < 1e-6, f"identical inputs gave {diff}"
    assert "CI excludes zero  = False" in out, out
    print("[OK] identical inputs -> difference 0.000, CI includes zero")

    shuffled = list(zip(cases, np.clip(difficulty + 0.05, 0, 1)))
    rng.shuffle(shuffled)
    write(tmp / "a3.csv", shuffled)
    out = run(tmp / "a3.csv", tmp / "b1.csv")
    diff = float(parse(out, "paired difference").split("=")[1].split("Dice")[0])
    assert abs(diff - 5.0) < 0.01, f"row-order shuffle changed the result: {diff}"
    print(f"[OK] row order irrelevant, pairing is by case name: {diff:+.3f}")

    a = np.clip(difficulty + 0.02 + rng.normal(0, 0.01, n), 0, 1)
    b = np.clip(difficulty + rng.normal(0, 0.01, n), 0, 1)
    write(tmp / "a4.csv", zip(cases, a))
    write(tmp / "b4.csv", zip(cases, b))
    out = run(tmp / "a4.csv", tmp / "b4.csv")
    lo, hi = (float(x) for x in parse(out, "paired difference").split("[")[1].rstrip("]").split(","))
    paired_w = hi - lo
    br = np.random.default_rng(1)
    ua = [a[br.integers(0, n, n)].mean() for _ in range(1000)]
    ub = [b[br.integers(0, n, n)].mean() for _ in range(1000)]
    unpaired_w = (np.percentile(ua, 97.5) - np.percentile(ua, 2.5)) * 100 + \
                 (np.percentile(ub, 97.5) - np.percentile(ub, 2.5)) * 100
    assert paired_w < unpaired_w, f"paired CI {paired_w:.2f} not tighter than unpaired {unpaired_w:.2f}"
    print(f"[OK] paired CI width {paired_w:.2f} pts << summed unpaired widths {unpaired_w:.2f} pts")

    write(tmp / "a5.csv", list(zip(cases, difficulty))[:150])
    out = run(tmp / "a5.csv", tmp / "b1.csv")
    assert "n paired cases   = 150" in out, out
    assert "WARNING" in out, "missing-case warning not emitted"
    print("[OK] disjoint cases dropped with a warning, paired on the 150 shared")

    print("\nall paired-bootstrap tests passed")


if __name__ == "__main__":
    main()
