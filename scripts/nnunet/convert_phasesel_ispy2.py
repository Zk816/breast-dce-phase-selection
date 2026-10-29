"""Build nnU-Net datasets for the phase-selection ceiling anchor (analysis_lock.md item 2).

Two datasets, identical except the phase subset fed in:

  Dataset520_PhaseSel_p12    channels = phases 1,2      (the learned pair)
  Dataset521_PhaseSel_p0123  channels = phases 0,1,2,3  (all four)

Both use the SAME cases, the SAME within-ISPY2 split as the ResUNet runs
(splits/withinISPY2_K4.json, 688/96/196), and the SAME source voxels: the fp16
magnitude phases already resampled onto the processed grid in .phase_sel_cache/.
So the only difference from our runs is the network and its training recipe —
which is the point of the anchor.

Two nnU-Net defaults must be overridden after preprocessing:
  1. per-channel z-scoring  -> shared normalization across phase channels
     (protocol.md §7; per-channel destroys cross-phase relationships)
  2. nnU-Net's own 5-fold CV -> our fixed split, written to
     nnUNet_preprocessed/<ds>/splits_final.json so `--fold 0` uses train+val,
     and the 196 test cases are held out entirely (they go to imagesTs).

    python scripts/nnunet/convert_phasesel_ispy2.py            # both datasets
    python scripts/nnunet/convert_phasesel_ispy2.py --check    # verify only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = "/path/to/MAMA-MIA_v3"
CACHE = ".phase_sel_cache"
NNRAW = Path("nnunet/nnUNet_raw")
SPLIT = "splits/withinISPY2_K4.json"
DATASETS = {"Dataset520_PhaseSel_p12": [1, 2], "Dataset521_PhaseSel_p0123": [0, 1, 2, 3],
            "Dataset522_PhaseSel_p123": [1, 2, 3],
            "Dataset523_PhaseSel_p012": [0, 1, 2],
            "Dataset524_PhaseSel_p13": [1, 3],
            "Dataset525_PhaseSel_p23": [2, 3],
            "Dataset526_PhaseSel_p01": [0, 1],
            "Dataset527_PhaseSel_p02": [0, 2],
            "Dataset528_PhaseSel_p03": [0, 3]}


def write_case(pid, phases, out_img, out_lab):
    ref = nib.load(f"{ROOT}/labelsTr/{pid}.nii.gz")
    stack = np.load(Path(CACHE, f"{pid}.npy"), mmap_mode="r")
    for ch, ph in enumerate(phases):
        arr = np.asarray(stack[ph]).astype(np.float32)
        nib.save(nib.Nifti1Image(arr, ref.affine, ref.header),
                 out_img / f"{pid}_{ch:04d}.nii.gz")
    lab = (ref.get_fdata() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(lab, ref.affine, ref.header), out_lab / f"{pid}.nii.gz")


def build(name, phases, split):
    ds = NNRAW / name
    tr_i, tr_l, ts_i, ts_l = (ds / "imagesTr", ds / "labelsTr", ds / "imagesTs", ds / "labelsTs")
    for d in (tr_i, tr_l, ts_i, ts_l):
        d.mkdir(parents=True, exist_ok=True)
    train_val = split["train"] + split["val"]
    for i, pid in enumerate(sorted(train_val), 1):
        write_case(pid, phases, tr_i, tr_l)
        if i % 100 == 0:
            print(f"    {name}: {i}/{len(train_val)} train+val", flush=True)
    for pid in sorted(split["test"]):
        write_case(pid, phases, ts_i, ts_l)
    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {str(i): f"phase{p}" for i, p in enumerate(phases)},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": len(train_val),
        "file_ending": ".nii.gz",
        "description": f"Phase-selection ceiling anchor, phases {phases}, within-ISPY2 "
                       f"({len(split['train'])}/{len(split['val'])}/{len(split['test'])})",
    }, indent=2))
    print(f"  {name}: {len(train_val)} train+val, {len(split['test'])} test -> {ds}")
    return ds


def write_split_file(name, split):
    """Pin nnU-Net fold 0 to OUR train/val split (protocol.md §2: one split file for all models)."""
    pre = Path("nnunet/nnUNet_preprocessed") / name
    pre.mkdir(parents=True, exist_ok=True)
    folds = [{"train": sorted(split["train"]), "val": sorted(split["val"])}]
    (pre / "splits_final.json").write_text(json.dumps(folds, indent=2))
    print(f"  {name}: fold 0 pinned to our split -> {pre/'splits_final.json'}")


def check(split):
    print("\nCHECK")
    ok = True
    for name, phases in DATASETS.items():
        ds = NNRAW / name
        n_tr = len(list((ds / "imagesTr").glob("*_0000.nii.gz"))) if (ds / "imagesTr").is_dir() else 0
        n_ts = len(list((ds / "imagesTs").glob("*_0000.nii.gz"))) if (ds / "imagesTs").is_dir() else 0
        n_ch = len(list((ds / "imagesTr").glob(f"{sorted(split['train'])[0]}_*.nii.gz"))) if n_tr else 0
        exp_tr, exp_ts = len(split["train"]) + len(split["val"]), len(split["test"])
        good = n_tr == exp_tr and n_ts == exp_ts and n_ch == len(phases)
        ok &= good
        print(f"  {name}: train+val {n_tr}/{exp_tr}, test {n_ts}/{exp_ts}, "
              f"channels {n_ch}/{len(phases)}  {'OK' if good else 'MISMATCH'}")
        sf = Path("nnunet/nnUNet_preprocessed") / name / "splits_final.json"
        if sf.exists():
            f = json.loads(sf.read_text())[0]
            leak = set(f["train"]) & set(f["val"])
            print(f"    fold0 train {len(f['train'])} val {len(f['val'])} overlap {len(leak)}"
                  f"  {'OK' if not leak else 'LEAK'}")
            ok &= not leak
    print("ALL CHECKS PASS" if ok else "CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    split = json.loads(Path(SPLIT).read_text())
    if a.check:
        raise SystemExit(0 if check(split) else 1)
    print(f"within-ISPY2 split: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test")
    for name, phases in DATASETS.items():
        if (NNRAW / name).exists() and not a.force:
            print(f"  {name} exists, skipping (use --force to rebuild)")
            continue
        if a.force and (NNRAW / name).exists():
            shutil.rmtree(NNRAW / name)
        build(name, phases, split)
        write_split_file(name, split)
    check(split)
    print("\nNEXT: preprocess, then OVERRIDE per-channel normalization to shared before training.")
    print("      see scripts/nnunet/run_phasesel_anchor.sh")


if __name__ == "__main__":
    main()
