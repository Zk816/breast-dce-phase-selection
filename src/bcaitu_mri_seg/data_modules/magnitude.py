"""Load true-magnitude phases resampled onto the processed (V3) grid.

The processed MAMA-MIA volumes are z-scored (median<0, ~66% negative), so
relative enhancement E=(S(t)-S0)/S0 is meaningless there — it needs a true zero.
The raw volumes are positive magnitude with a true zero. We resample each raw
phase onto the processed pre-contrast grid (linear), so masks/labels/geometry
stay aligned, and cache the result (computed once, reused by precompute + training).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np

RAW_ROOT = "/path/to/MAMA-MIA/images"


def magnitude_phases(pid: str, data_root: str, cache_dir: str = ".rrm_cache_mag") -> np.ndarray:
    """Return (K, D, H, W) float32 magnitude phases on the processed grid for ``pid``."""
    cache = Path(cache_dir) / f"{pid}.npz"
    if cache.exists():
        return np.load(cache)["mag"].astype(np.float32)
    target = nib.load(f"{data_root}/imagesTr/{pid}_0000.nii.gz")
    raw_files = sorted(Path(RAW_ROOT, pid).glob(f"{pid}_0*.nii.gz"))
    mag = np.stack([
        np.clip(nip.resample_from_to(nib.load(str(f)), target, order=1).get_fdata(dtype=np.float32), 0.0, None)
        for f in raw_files
    ], axis=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, mag=mag.astype(np.float16))
    return mag


def relative_enhancement(phases: np.ndarray, breast: np.ndarray) -> np.ndarray:
    """E(t) = (S(t)-S0) / max(S0, 0.05*median(S0 in breast)), clipped to [-1, 5].

    Magnitude-domain enhancement with a positive baseline floor (true zero
    required); E[0] == 0. Shared by precompute and the datamodule so cached
    fits and the training input use an identical definition.
    """
    pre = phases[0]
    floor = max(0.05 * float(np.median(pre[breast])), 1e-3) if breast.any() else 1e-3
    denom = np.maximum(pre, floor)
    E = np.clip((phases - pre[None]) / denom[None], -1.0, 5.0)
    E[0] = 0.0
    return E.astype(np.float32)
