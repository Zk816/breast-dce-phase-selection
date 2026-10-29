"""Bake shared-across-phase normalization into the nnU-Net anchor datasets."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = "/path/to/MAMA-MIA_v3"
NNRAW = Path("nnunet/nnUNet_raw")
CACHE = ".phase_sel_cache"
DATASETS = {"Dataset520_PhaseSel_p12": [1, 2], "Dataset521_PhaseSel_p0123": [0, 1, 2, 3],
            "Dataset522_PhaseSel_p123": [1, 2, 3],
            "Dataset523_PhaseSel_p012": [0, 1, 2],
            "Dataset524_PhaseSel_p13": [1, 3],
            "Dataset525_PhaseSel_p23": [2, 3],
            "Dataset526_PhaseSel_p01": [0, 1],
            "Dataset527_PhaseSel_p02": [0, 2],
            "Dataset528_PhaseSel_p03": [0, 3]}
FLAG = ".shared_norm_applied"


def moments(pid, idx):
    st = np.load(Path(CACHE, f"{pid}_stats.npz"))
    n = int(st["count"]) * len(idx)
    if n == 0:
        return 0.0, 1.0
    s = float(st["sums"][list(idx)].sum())
    q = float(st["sqs"][list(idx)].sum())
    mu = s / n
    return mu, float(np.sqrt(max(q / n - mu * mu, 0.0))) + 1e-6


def do_case(args):
    ds_dir, pid, idx, sub = args
    mu, sd = moments(pid, idx)
    for ch in range(len(idx)):
        f = Path(ds_dir, sub, f"{pid}_{ch:04d}.nii.gz")
        img = nib.load(str(f))
        arr = img.get_fdata(dtype=np.float32)
        nib.save(nib.Nifti1Image(((arr - mu) / sd).astype(np.float32), img.affine, img.header), f)
    return pid


def ids_of(ds_dir, sub):
    return sorted({p.name.rsplit("_", 1)[0] for p in Path(ds_dir, sub).glob("*_0000.nii.gz")})


def verify(name, idx):
    ds = NNRAW / name
    rng = np.random.default_rng(0)
    pids = ids_of(ds, "imagesTr")
    sample = rng.choice(pids, min(4, len(pids)), replace=False)
    print(f"\n{name}: shared-normalization check on {len(sample)} cases")
    print(f"  {'case':16s} {'joint mean':>11s} {'joint std':>10s}   per-channel means")
    ok = True
    for pid in sample:
        chans = [nib.load(str(ds / "imagesTr" / f"{pid}_{c:04d}.nii.gz")).get_fdata(dtype=np.float32)
                 for c in range(len(idx))]
        x = np.stack(chans)
        import nibabel.processing as nip
        pre = nib.load(f"{ROOT}/imagesTr/{pid}_0000.nii.gz")
        breast = nip.resample_from_to(nib.load(f"{ROOT}/breast_masks/{pid}.nii.gz"),
                                      pre, order=0, mode="nearest").get_fdata() > 0
        bm = np.broadcast_to(breast, x.shape)
        jm, js = float(x[bm].mean()), float(x[bm].std())
        pcm = [float(c[breast].mean()) for c in chans]
        spread = max(pcm) - min(pcm)
        good = abs(js - 1.0) < 0.05 and abs(jm) < 0.05 and spread > 0.01
        ok &= good
        print(f"  {pid:16s} {jm:11.4f} {js:10.4f}   {[round(v,3) for v in pcm]}"
              f"{'' if good else '   <-- CHECK'}")
    print("  joint std ~1 and per-channel means DIFFER => shared normalization, "
          "phase contrast preserved" if ok else "  PROBLEM")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    for name, idx in DATASETS.items():
        ds = NNRAW / name
        if a.verify:
            verify(name, idx)
            continue
        if (ds / FLAG).exists():
            print(f"{name}: already normalized (flag present), skipping")
            continue
        jobs = [(ds, pid, idx, sub) for sub in ("imagesTr", "imagesTs") for pid in ids_of(ds, sub)]
        print(f"{name}: normalizing {len(jobs)} cases x {len(idx)} channels")
        with ProcessPoolExecutor(a.workers) as ex:
            for i, _ in enumerate(ex.map(do_case, jobs, chunksize=4), 1):
                if i % 200 == 0:
                    print(f"   {i}/{len(jobs)}", flush=True)
        (ds / FLAG).write_text("shared z-norm across phases applied; use NoNormalization in plans\n")
        dj = ds / "dataset.json"
        d = json.loads(dj.read_text())
        d["channel_names"] = {k: "noNorm" for k in d["channel_names"]}
        dj.write_text(json.dumps(d, indent=2))
        print(f"   done; dataset.json channel_names -> 'noNorm' (nnU-Net maps this to NoNormalization)")
        verify(name, idx)


if __name__ == "__main__":
    main()
