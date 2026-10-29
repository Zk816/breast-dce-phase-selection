"""Rescore saved k=2 predictions against the uncorrected automatic MAMA-MIA masks."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

RAW = Path("/path/to/MAMA-MIA/segmentations")
LAB = Path("/path/to/MAMA-MIA_v3/labelsTr")
SUBSETS = ["p12", "p13", "p23", "p01", "p02", "p03"]
SEEDS = [42, 43, 44, 45, 46]


def auto_on_label_grid(pid: str) -> np.ndarray:
    lab = nib.load(str(LAB / f"{pid}.nii.gz"))
    a = nib.load(str(RAW / "automatic" / f"{pid}.nii.gz"))
    return np.asarray(resample_from_to(a, (lab.shape, lab.affine), order=0).dataobj) > 0


def dice(p: np.ndarray, g: np.ndarray) -> float:
    s = int(p.sum()) + int(g.sum())
    return 1.0 if s == 0 else 2.0 * float((p & g).sum()) / s


def score_one(args):
    run, cases = args
    ref = {}
    out = []
    for pid in cases:
        if pid not in ref:
            ref[pid] = auto_on_label_grid(pid)
        pr = np.asarray(nib.load(f"results/{run}/pred/{pid}.nii.gz").dataobj) > 0
        out.append((pid, dice(pr, ref[pid])))
    return run, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase_sel_report/auto_masks")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src = "results/phasesel_20260730_s2_p12_ISPY2_seed42/per_case.csv"
    cases = [r["case"].replace(".nii.gz", "") for r in csv.DictReader(open(src))]
    runs = [f"phasesel_20260730_s2_{s}_ISPY2_seed{sd}" for s in SUBSETS for sd in SEEDS]

    with ProcessPoolExecutor(a.workers) as ex:
        for run, rows in ex.map(score_one, [(r, cases) for r in runs]):
            with open(out / f"{run}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["case", "dice"])
                w.writerows(rows)
            print(f"{run}  mean {np.mean([d for _, d in rows]):.4f}", flush=True)


if __name__ == "__main__":
    main()
