"""Stage-1 selector assertions + protocol pre-flight stats (exp_phase_sel.md Day 1)."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from bcaitu_mri_seg.data_modules.phase_sel_dm import LitPhaseSelDataset
from bcaitu_mri_seg.net_modules.phase_sel.lit_phase_sel import LitPhaseSelNet
from bcaitu_mri_seg.net_modules.phase_sel.selector import PhaseSelector

ROOT = "/path/to/MAMA-MIA_v3"


def check_mask_cardinality(K, draws=200):
    for k in range(1, K + 1):
        sel = PhaseSelector(K, k)
        sel.theta.data = torch.randn(K)
        for train in (True, False):
            sel.train(train)
            for tau in (5.0, 2.0, 0.5):
                sel.set_progress(0.0 if tau == 5.0 else 1.0)
                sel._tau.fill_(tau)
                for _ in range(draws if train else 5):
                    s = float(sel.mask().sum())
                    assert abs(s - k) < 1e-4, f"k={k} train={train} tau={tau}: sum={s}"
    print(f"[OK] mask.sum() == k for k=1..{K}, train and eval, tau in (5.0, 2.0, 0.5)")


def check_theta_grad(K, k=2):
    net = LitPhaseSelNet(in_phases=K, select=True, k=k, depth=3, max_channels=64, patch_size=32)
    x = torch.randn(1, K, 32, 32, 32)
    y = torch.zeros(1, 1, 32, 32, 32)
    y[..., 8:16, 8:16, 8:16] = 1.0
    from bcaitu_mri_seg.net_modules.phase_sel.losses import segmentation_loss
    segmentation_loss(net._predict(x), y).backward()
    g = net._selector.theta.grad
    assert g is not None and float(g.norm()) > 0, f"theta.grad = {g}"
    print(f"[OK] theta.grad non-zero after one step: {g.detach().numpy().round(5)}")


def check_k_equals_K(K):
    torch.manual_seed(0)
    net = LitPhaseSelNet(in_phases=K, select=True, k=K, depth=3, max_channels=64, patch_size=32)
    net.eval()
    x = torch.randn(2, K, 32, 32, 32)
    with torch.no_grad():
        masked = net._predict(x)
        plain = net._net(x)
    d = float((masked - plain).abs().max())
    assert d == 0.0, f"k==K does not reproduce the backbone: max|diff| = {d}"
    net.train()
    with torch.no_grad():
        d_train = float((net._predict(x) - plain).abs().max())
    assert d_train == 0.0, f"k==K differs in train mode: {d_train}"
    print("[OK] k == K reproduces the unmodified backbone exactly (train and eval)")


def check_data(K, cases):
    dm = LitPhaseSelDataset(dataset_root=ROOT, num_phases=K, patch_size=128,
                            batch_size=2, num_workers=0, eval_heldout=True)
    dm.setup()
    ds = dm._ds(dm._train_ids[:cases], train=False)
    stats = []
    for i in range(min(cases, len(ds))):
        img = ds[i]["image"].numpy()
        stats.append([[c.min(), np.percentile(c, 1), np.percentile(c, 50),
                       np.percentile(c, 99), np.percentile(c, 99.9), c.max()] for c in img])
    s = np.mean(stats, 0)
    print(f"\nper-channel input stats over {len(stats)} cases (shared z-norm across phases):")
    print("  ch     min      p1     p50     p99   p99.9     max")
    for c, row in enumerate(s):
        print(f"  {c:2d} " + " ".join(f"{v:7.2f}" for v in row))
    b = next(iter(dm.train_dataloader()))
    print(f"\ntrain batch: image {tuple(b['image'].shape)}, mask {tuple(b['mask'].shape)}, "
          f"fg voxels {int(b['mask'].sum())}")
    print(f"splits: train {len(dm._train_ids)} / val {len(dm._val_ids)} / "
          f"heldout {len(dm._heldout_ids)} ({dm.hparams.heldout_cohort})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_phases", type=int, default=4)
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--skip_data", action="store_true")
    a = ap.parse_args()
    check_mask_cardinality(a.num_phases)
    check_theta_grad(a.num_phases)
    check_k_equals_K(a.num_phases)
    if not a.skip_data:
        check_data(a.num_phases, a.cases)


if __name__ == "__main__":
    main()
