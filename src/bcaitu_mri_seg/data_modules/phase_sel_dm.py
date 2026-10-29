"""Datamodule for learned phase selection (exp_phase_sel.md, protocol.md §1/§2/§7).

Input = RAW MAGNITUDE phases resampled onto the processed grid (protocol §1),
normalized with a SINGLE mean/std shared across all phases the model receives
(protocol §7 — never per-channel, which would destroy cross-phase relations).
Normalization uses only the phases actually fed in, so a Stage-2 k-phase model
never touches the phases it was not given.

Action space: the first ``num_phases`` phases of every case; cases with fewer
are dropped (exp_phase_sel.md VARIABLE K — no padding, no interpolation, since
a padded phase would be a fake phase the selector could choose). Exclusions are
printed per cohort at setup.

Splits (protocol §2): patient-level, seed 42, cohort-stratified, generated once
into ``splits/hold{COHORT}.json`` and reused verbatim afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import nibabel as nib
import nibabel.processing as nip
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bcaitu_mri_seg.data_modules import magnitude as MAG


def build_splits(dataset_root: str, mag_cache_dir: str, heldout_cohort: str,
                 splits_dir: str, val_frac: float, seed: int) -> dict:
    """Load ``splits/hold{COHORT}.json``, creating it once if absent (protocol §2)."""
    path = Path(splits_dir, f"hold{heldout_cohort}.json")
    if path.exists():
        return json.loads(path.read_text())

    root = Path(dataset_root)
    meta = json.loads((root / "meta.json").read_text())
    labelled = {p.name.split(".")[0] for p in (root / "labelsTr").glob("*.nii.gz")}
    cached = {p.stem for p in Path(mag_cache_dir).glob("*.npz")}
    ids = sorted(i for i in meta if i in labelled and i in cached)

    rng = np.random.default_rng(seed)
    train, val = [], []
    for cohort in sorted({i.split("_")[0] for i in ids} - {heldout_cohort}):
        pool = np.array([i for i in ids if i.split("_")[0] == cohort])
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * val_frac)))
        val += sorted(pool[:n_val].tolist())
        train += sorted(pool[n_val:].tolist())
    splits = {
        "heldout_cohort": heldout_cohort, "seed": seed, "val_frac": val_frac,
        "train": sorted(train), "val": sorted(val),
        "heldout": sorted(i for i in ids if i.split("_")[0] == heldout_cohort),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))
    return splits


class PhaseSelDataset(Dataset):
    def __init__(self, ids, data_root, mag_cache_dir, times, num_phases, phase_indices,
                 patch_size, train, seed=0, fast_cache_dir=None, subtract_from=None,
                 per_channel_norm=False):
        self._ids = ids
        self._root = data_root
        self._mag = mag_cache_dir
        self._times = times
        self._np = num_phases
        self._idx = list(range(num_phases)) if phase_indices is None else list(phase_indices)
        self._P = patch_size
        self._train = train
        self._rng = np.random.default_rng(seed)
        self._fast = fast_cache_dir
        self._sub = subtract_from
        self._pcn = per_channel_norm
        self._load_idx = (self._idx if subtract_from is None
                          else sorted(set(self._idx) | {subtract_from}))

    def __len__(self):
        return len(self._ids)

    def _load(self, pid):
        """Fast path: mmap the uncompressed phase stack, read only the phases this run
        needs, and take mu/sd from cached per-phase moments (exact — see
        scripts/phase_sel_precompute.py --verify). Slow path is the original loader."""
        label = nib.load(str(Path(self._root, "labelsTr", f"{pid}.nii.gz"))).get_fdata()[None] > 0
        label = label.astype(np.float32)
        if self._fast is not None:
            stack = np.load(Path(self._fast, f"{pid}.npy"), mmap_mode="r")
            x = np.asarray(stack[self._load_idx]).astype(np.float32)
            stats = np.load(Path(self._fast, f"{pid}_stats.npz"))
            n = int(stats["count"]) * len(self._load_idx)
            if n == 0:
                return x, label, (0.0, 1.0)
            if self._pcn:
                c = int(stats["count"])
                m = stats["sums"][self._load_idx] / c
                v = np.maximum(stats["sqs"][self._load_idx] / c - m * m, 0.0)
                return x, label, (m[:, None, None, None].astype(np.float32),
                                  (np.sqrt(v) + 1e-6)[:, None, None, None].astype(np.float32))
            s = float(stats["sums"][self._load_idx].sum())
            q = float(stats["sqs"][self._load_idx].sum())
            mu = s / n
            sd = float(np.sqrt(max(q / n - mu * mu, 0.0))) + 1e-6
            return x, label, (mu, sd)
        pre = nib.load(str(Path(self._root, "imagesTr", f"{pid}_0000.nii.gz")))
        phases = MAG.magnitude_phases(pid, self._root, self._mag)[: self._np]
        breast = nip.resample_from_to(nib.load(str(Path(self._root, "breast_masks", f"{pid}.nii.gz"))),
                                      pre, order=0, mode="nearest").get_fdata() > 0
        x = phases[self._load_idx]
        region = np.broadcast_to(breast, x.shape) if breast.any() else x != 0
        if self._pcn:
            m = np.empty(len(x), np.float32)
            d = np.empty(len(x), np.float32)
            for c in range(len(x)):
                v = x[c][region[c]]
                m[c] = v.mean() if v.size else 0.0
                d[c] = (v.std() + 1e-6) if v.size else 1.0
            return (x.astype(np.float32), label,
                    (m[:, None, None, None], d[:, None, None, None]))
        vals = x[region]
        mu = float(vals.mean()) if vals.size else 0.0
        sd = float(vals.std()) + 1e-6 if vals.size else 1.0
        return x.astype(np.float32), label, (mu, sd)

    def _pad(self, arr):
        sp = arr.shape[-3:]
        pads = [(0, 0)] * (arr.ndim - 3) + [(0, max(0, self._P - sp[a])) for a in range(3)]
        return np.pad(arr, pads) if any(p[1] for p in pads) else arr

    def _crop(self, shape, label):
        P = self._P
        if label.sum() > 0 and self._rng.random() < 0.5:
            fg = np.argwhere(label[0] > 0)
            ctr = fg[self._rng.integers(len(fg))]
            start = [int(np.clip(ctr[a] - P // 2, 0, max(0, shape[a] - P))) for a in range(3)]
        else:
            start = [int(self._rng.integers(0, max(1, shape[a] - P + 1))) for a in range(3)]
        return tuple(slice(start[a], start[a] + P) for a in range(3))

    def __getitem__(self, i):
        pid = self._ids[i]
        x, label, (mu, sd) = self._load(pid)
        img = ((x - mu) / sd).astype(np.float32)
        if self._sub is not None:
            base = img[self._load_idx.index(self._sub)][None]
            img = img[[self._load_idx.index(j) for j in self._idx]] - base
        if not self._train:
            return {"image": torch.from_numpy(img), "mask": torch.from_numpy(label), "pid": pid}
        img, label = self._pad(img), self._pad(label)
        sl = self._crop(img.shape[1:], label)
        img = img[(slice(None), *sl)]
        label = label[(slice(None), *sl)]
        for axis in range(1, 4):
            if self._rng.random() < 0.5:
                img, label = np.flip(img, axis), np.flip(label, axis)
        return {"image": torch.from_numpy(img.copy()), "mask": torch.from_numpy(label.copy())}


class LitPhaseSelDataset(L.LightningDataModule):
    """Magnitude multi-phase datamodule with a fixed K action space."""

    def __init__(
        self,
        *,
        dataset_root: str,
        mag_cache_dir: str = ".rrm_cache_mag",
        splits_dir: str = "splits",
        heldout_cohort: str = "DUKE",
        num_phases: int = 4,
        phase_indices: list[int] | None = None,
        patch_size: int = 128,
        batch_size: int = 2,
        num_workers: int = 4,
        val_frac: float = 0.1,
        eval_heldout: bool = False,
        heldout_subset: int = 0,
        seed: int = 42,
        split_file: str | None = None,
        aug_seed: int | None = None,
        fast_cache_dir: str | None = None,
        subtract_from: int | None = None,
        per_channel_norm: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        if h.split_file is not None:
            splits = json.loads(Path(h.split_file).read_text())
            splits["heldout"] = splits["test"]
            print(f"[phase_sel] WITHIN-COHORT split {h.split_file} "
                  f"(cohort={splits.get('cohort')}, stratified_by={splits.get('stratified_by')})")
        else:
            splits = build_splits(h.dataset_root, h.mag_cache_dir, h.heldout_cohort,
                                  h.splits_dir, h.val_frac, h.seed)
        meta = json.loads(Path(h.dataset_root, "meta.json").read_text())
        self._times = {i: meta[i]["acquisition_times"] for i in meta}

        def keep(ids):
            return [i for i in ids if len(self._times.get(i, [])) >= h.num_phases]

        for name in ("train", "val", "heldout"):
            kept = keep(splits[name])
            dropped = sorted(set(splits[name]) - set(kept))
            by_cohort = {c: sum(1 for i in dropped if i.split("_")[0] == c)
                         for c in sorted({i.split("_")[0] for i in dropped})}
            print(f"[phase_sel] {name}: {len(kept)} cases with K>={h.num_phases}, "
                  f"{len(dropped)} excluded {by_cohort}")
        self._train_ids, self._val_ids = keep(splits["train"]), keep(splits["val"])
        self._heldout_ids = keep(splits["heldout"])
        if h.heldout_subset and len(self._heldout_ids) > h.heldout_subset:
            self._heldout_ids = sorted(np.random.default_rng(h.seed).choice(
                self._heldout_ids, h.heldout_subset, replace=False).tolist())
        if h.phase_indices is not None:
            print(f"[phase_sel] fixed phase_indices={list(h.phase_indices)} "
                  f"(in_channels={len(h.phase_indices)})")
        in_ch = h.num_phases if h.phase_indices is None else len(h.phase_indices)
        aug = h.seed if h.aug_seed is None else h.aug_seed
        print(f"[phase_sel] EFFECTIVE batch_size={h.batch_size} patch_size={h.patch_size} "
              f"in_channels={in_ch} num_phases={h.num_phases} "
              f"steps_per_epoch={len(self._train_ids) // h.batch_size} "
              f"split_seed={h.seed} aug_seed={aug} "
              f"n_train={len(self._train_ids)} n_val={len(self._val_ids)} n_test={len(self._heldout_ids)} "
              f"num_workers={h.num_workers} fast_cache={h.fast_cache_dir}")

    def _ds(self, ids, train):
        """Augmentation RNG derives from ``aug_seed`` (the TRAINING seed) so replicate
        seeds get independent crops/flips, while the split stays pinned to ``seed``."""
        h = self.hparams
        aug = h.seed if h.aug_seed is None else h.aug_seed
        return PhaseSelDataset(ids, h.dataset_root, h.mag_cache_dir, self._times,
                               h.num_phases, h.phase_indices, h.patch_size, train,
                               aug if train else h.seed, h.fast_cache_dir, h.subtract_from,
                               h.per_channel_norm)

    def train_dataloader(self):
        h = self.hparams
        return DataLoader(self._ds(self._train_ids, True), batch_size=h.batch_size, shuffle=True,
                          num_workers=h.num_workers, drop_last=True,
                          persistent_workers=h.num_workers > 0)

    def val_dataloader(self):
        h = self.hparams
        internal = DataLoader(self._ds(self._val_ids, False), batch_size=1, shuffle=False,
                              num_workers=h.num_workers)
        if not h.eval_heldout:
            return internal
        heldout = DataLoader(self._ds(self._heldout_ids, False), batch_size=1, shuffle=False,
                             num_workers=h.num_workers)
        return [internal, heldout]
