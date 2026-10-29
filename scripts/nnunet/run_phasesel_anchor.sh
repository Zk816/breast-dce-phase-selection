#!/usr/bin/env bash
# nnU-Net ceiling anchor for the phase-selection study (analysis_lock.md item 2).
#
# Datasets 520 (phases 1,2) and 521 (phases 0-3) hold the SAME cases and the SAME
# within-ISPY2 split as the ResUNet sweep, with shared-across-phase normalization
# already baked in (channel_names = noNorm, so nnU-Net does not re-normalize).
# Fold 0 is pinned to our train/val via splits_final.json; the 196 test cases are
# in imagesTs and never seen during training.
#
#   bash scripts/nnunet/run_phasesel_anchor.sh preprocess
#   GPU=0 DATASET=520 bash scripts/nnunet/run_phasesel_anchor.sh train
set -uo pipefail
cd "$(dirname "$0")/../.."
VENV=/path/to/nnunet_venv
NN_ROOT="$PWD/nnunet"
export nnUNet_raw="$NN_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NN_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NN_ROOT/nnUNet_results"
export PATH="$VENV/bin:$PATH"
GPU=${GPU:-0}; DATASET=${DATASET:-520}; CONFIG=${CONFIG:-3d_fullres}

case "${1:-}" in
  preprocess)
    for d in 520 521; do
      echo "=== plan_and_preprocess Dataset$d ==="
      # -np: parallel workers; --verify_dataset_integrity once per dataset
      nnUNetv2_plan_and_preprocess -d $d -c $CONFIG -np 16 --verify_dataset_integrity
    done ;;
  train)
    echo "=== train Dataset$DATASET fold 0 on GPU$GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $DATASET $CONFIG 0 --npz ;;
  *) echo "usage: $0 {preprocess|train}"; exit 1 ;;
esac
