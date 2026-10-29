"""Unified held-out evaluation (protocol.md §5) — ONE metric for ALL methods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def case_hd95(pred, gt, label, spacing):
    """95th-percentile symmetric surface distance (mm)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    p = pred == label
    g = gt == label
    if not p.any() or not g.any():
        return float("nan")
    ps = p & ~binary_erosion(p)
    gs = g & ~binary_erosion(g)
    dp = distance_transform_edt(~p, sampling=spacing)
    dg = distance_transform_edt(~g, sampling=spacing)
    d = np.concatenate([dg[ps], dp[gs]])
    return float(np.percentile(d, 95))


def case_dice(pred, gt, label):
    p = pred == label
    g = gt == label
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return 2.0 * float((p & g).sum()) / float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--hd95", action="store_true",
                    help="also compute 95%% Hausdorff distance per case (slower)")
    args = ap.parse_args()

    pred = {p.name: p for p in Path(args.pred_dir).glob("*.nii.gz")}
    gt = {g.name: g for g in Path(args.gt_dir).glob("*.nii.gz")}
    names = sorted(set(pred) & set(gt))
    missing = (set(pred) ^ set(gt))
    if missing:
        print(f"WARNING: {len(missing)} files not in both dirs (using {len(names)} matched)")

    dices, hds = [], []
    for n in names:
        pimg = nib.load(str(pred[n]))
        gimg = nib.load(str(gt[n]))
        p = np.rint(pimg.get_fdata())
        g = np.rint(gimg.get_fdata())
        dices.append(case_dice(p, g, args.label))
        hds.append(case_hd95(p, g, args.label, gimg.header.get_zooms()[:3]) if args.hd95 else float("nan"))
    dices = np.array(dices)
    hds = np.array(hds)

    rng = np.random.default_rng(42)
    boots = [dices[rng.integers(0, len(dices), len(dices))].mean() for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"n_cases       = {len(dices)}")
    print(f"mean Dice     = {dices.mean():.4f}")
    print(f"95% CI (boot) = [{lo:.4f}, {hi:.4f}]")
    print(f"median / min  = {np.median(dices):.4f} / {dices.min():.4f}")
    print(f"frac Dice=0   = {float((dices == 0).mean()):.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write("case,dice,hd95_mm\n")
            for n, d, h in zip(names, dices, hds):
                f.write(f"{n},{d:.6f},{h:.4f}\n")
        print(f"per-case -> {args.out}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"n_cases": len(dices), "mean_dice": float(dices.mean()),
                       "ci_lo": float(lo), "ci_hi": float(hi),
                       "median_dice": float(np.median(dices)),
                       "frac_zero": float((dices == 0).mean()),
                       "hd95_median_mm": (float(np.nanmedian(hds))
                                          if args.hd95 and not np.all(np.isnan(hds)) else None),
                       "pred_dir": args.pred_dir, "gt_dir": args.gt_dir}, f, indent=2)
        print(f"summary  -> {args.json}")


if __name__ == "__main__":
    main()
