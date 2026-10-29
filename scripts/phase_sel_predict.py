"""Write held-out-cohort predictions for a phase-selection run (protocol.md §4/§5).

Loads the best internal-val checkpoint (the ONLY selection criterion allowed),
runs sliding-window inference over the full held-out cohort, and writes one
uint8 mask per case onto the label grid so `scripts/unified_eval.py` can score
it with the same code as every other method.

    python scripts/phase_sel_predict.py --run phasesel_20260730_learned2_seed42 --gpu 0

Predictions -> results/{run}/pred/{pid}.nii.gz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from bcaitu_mri_seg.data_modules.phase_sel_dm import LitPhaseSelDataset
from bcaitu_mri_seg.net_modules.phase_sel.lit_phase_sel import LitPhaseSelNet


def best_ckpt(run: str) -> Path:
    cks = sorted(Path("checkpoints", run).glob("*.ckpt"))
    if not cks:
        msg = f"no checkpoint in checkpoints/{run}"
        raise FileNotFoundError(msg)
    return max(cks, key=lambda p: float(p.stem.split("val_dice=")[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="heldout", choices=["heldout", "val"])
    ap.add_argument("--ids_from", default=None,
                    help="split JSON whose 'test' ids to predict instead of this run's own "
                         "cohort; the phase indices and checkpoint are unchanged, which is what "
                         "makes it an external application of an ISPY2-trained model")
    ap.add_argument("--out", default=None, help="output dir (default results/{run}/pred)")
    ap.add_argument("--overlap", type=float, default=0.5)
    a = ap.parse_args()

    ck = best_ckpt(a.run)
    print(f"[{a.run}] checkpoint {ck.name}")
    dev = torch.device(f"cuda:{a.gpu}")
    net = LitPhaseSelNet.load_from_checkpoint(str(ck), map_location=dev).to(dev).eval()

    dm = LitPhaseSelDataset.load_from_checkpoint(str(ck))
    dm.setup()
    if a.ids_from:
        ids = json.loads(Path(a.ids_from).read_text())["test"]
        avail = set(dm._times)
        ids = [i for i in ids if i in avail]
    else:
        ids = dm._heldout_ids if a.split == "heldout" else dm._val_ids
    ds = dm._ds(ids, train=False)
    root = Path(dm.hparams.dataset_root)
    out = Path(a.out) if a.out else Path("results", a.run, "pred")
    out.mkdir(parents=True, exist_ok=True)
    if net._selector is not None:
        print(f"  selector active, phases {net._selector.hard_indices()}")
    print(f"  {len(ids)} {a.split} cases -> {out}")

    for i, pid in enumerate(ids):
        x = ds[i]["image"][None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = sliding_window_inference(x, net._patch_size, 4, net._predict,
                                              overlap=a.overlap)
        mask = (torch.sigmoid(logits.float())[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        ref = nib.load(str(root / "labelsTr" / f"{pid}.nii.gz"))
        if mask.shape != ref.shape:
            msg = f"{pid}: pred {mask.shape} != label {ref.shape}"
            raise RuntimeError(msg)
        nib.save(nib.Nifti1Image(mask, ref.affine, ref.header), out / f"{pid}.nii.gz")
        if (i + 1) % 25 == 0 or i + 1 == len(ids):
            print(f"  {i + 1}/{len(ids)}", flush=True)


if __name__ == "__main__":
    main()
