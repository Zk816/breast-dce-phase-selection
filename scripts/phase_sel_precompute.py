"""Build the fast training cache for the phase-selection sweep."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np

ROOT = "/path/to/MAMA-MIA_v3"
CACHE = ".phase_sel_cache"
MAG = ".rrm_cache_mag"


def build(pid: str, root: str = ROOT, cache: str = CACHE, mag: str = MAG):
    out = Path(cache, f"{pid}.npy")
    st = Path(cache, f"{pid}_stats.npz")
    if out.exists() and st.exists():
        return pid, "skip", 0
    phases = np.load(Path(mag, f"{pid}.npz"))["mag"].astype(np.float16)
    pre = nib.load(str(Path(root, "imagesTr", f"{pid}_0000.nii.gz")))
    breast = nip.resample_from_to(nib.load(str(Path(root, "breast_masks", f"{pid}.nii.gz"))),
                                  pre, order=0, mode="nearest").get_fdata() > 0
    x = phases.astype(np.float32)
    if breast.any():
        m, fallback = breast, False
    else:
        m, fallback = np.ones(breast.shape, bool), True
    sums = np.array([float(x[k][m].astype(np.float64).sum()) for k in range(x.shape[0])])
    sqs = np.array([float((x[k][m].astype(np.float64) ** 2).sum()) for k in range(x.shape[0])])
    Path(cache).mkdir(parents=True, exist_ok=True)
    np.save(out, phases)
    np.savez(st, sums=sums, sqs=sqs, count=np.int64(int(m.sum())),
             empty_breast_mask=np.bool_(fallback), n_phases=np.int64(x.shape[0]))
    return pid, "built", out.stat().st_size


def norm_stats(stats, idx):
    """Exact (mu, sd) for phase subset ``idx`` from the cached per-phase moments."""
    n = int(stats["count"]) * len(idx)
    if n == 0:
        return 0.0, 1.0
    s = float(stats["sums"][list(idx)].sum())
    q = float(stats["sqs"][list(idx)].sum())
    mu = s / n
    var = max(q / n - mu * mu, 0.0)
    return mu, float(np.sqrt(var)) + 1e-6


def verify(n_cases: int, cohorts: list[str]):
    """Compare derived mu/sd against the ORIGINAL per-item masked computation."""
    meta = json.loads(Path(ROOT, "meta.json").read_text())
    ok = True
    print("VERIFY — derived moments vs original masked mean/std, on the same voxels\n")
    print(f"{'case':16s} {'subset':10s} {'mu_old':>12s} {'mu_new':>12s} {'rel':>10s} "
          f"{'sd_old':>12s} {'sd_new':>12s} {'rel':>10s}")
    for coh in cohorts:
        ids = sorted(p for p in meta if p.startswith(coh)
                     and Path(CACHE, f"{p}.npy").exists())[:n_cases]
        for pid in ids:
            phases = np.load(Path(CACHE, f"{pid}.npy"), mmap_mode="r")
            stats = np.load(Path(CACHE, f"{pid}_stats.npz"))
            pre = nib.load(str(Path(ROOT, "imagesTr", f"{pid}_0000.nii.gz")))
            breast = nip.resample_from_to(nib.load(str(Path(ROOT, "breast_masks", f"{pid}.nii.gz"))),
                                          pre, order=0, mode="nearest").get_fdata() > 0
            for idx in ([0, 1, 2, 3], [1, 2], [2], [0, 3]):
                x = np.asarray(phases[idx]).astype(np.float32)
                region = np.broadcast_to(breast, x.shape) if breast.any() else x != 0
                vals = x[region]
                mu_o = float(vals.mean())
                sd_o = float(vals.std()) + 1e-6
                mu_n, sd_n = norm_stats(stats, idx)
                rm = abs(mu_n - mu_o) / max(abs(mu_o), 1e-12)
                rs = abs(sd_n - sd_o) / max(abs(sd_o), 1e-12)
                flag = "" if (rm < 1e-6 and rs < 1e-6) else "  <-- MISMATCH"
                if flag:
                    ok = False
                print(f"{pid:16s} {str(idx):10s} {mu_o:12.6f} {mu_n:12.6f} {rm:10.2e} "
                      f"{sd_o:12.6f} {sd_n:12.6f} {rs:10.2e}{flag}")
    print("\nEXACT to <1e-6 relative on every case/subset" if ok else "\nMISMATCH FOUND — do not use")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", default="ISPY2,DUKE")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify_cases", type=int, default=3)
    a = ap.parse_args()
    cohorts = a.cohorts.split(",")

    if a.verify:
        raise SystemExit(0 if verify(a.verify_cases, cohorts) else 1)

    meta = json.loads(Path(ROOT, "meta.json").read_text())
    labelled = {p.name.split(".")[0] for p in Path(ROOT, "labelsTr").glob("*.nii.gz")}
    cached = {p.stem for p in Path(MAG).glob("*.npz")}
    ids = sorted(p for p in meta if p.split("_")[0] in cohorts and p in labelled and p in cached)
    print(f"building fast cache for {len(ids)} cases ({a.cohorts}) -> {CACHE}/")
    total = built = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (pid, status, sz) in enumerate(ex.map(build, ids, chunksize=2), 1):
            total += sz
            built += status == "built"
            if i % 100 == 0 or i == len(ids):
                print(f"  {i}/{len(ids)}  built={built}  {total/1e9:.1f} GB", flush=True)
    print(f"done: {built} built, {len(ids)-built} skipped, {total/1e9:.1f} GB")


if __name__ == "__main__":
    main()
