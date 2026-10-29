#!/usr/bin/env bash
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
      nnUNetv2_plan_and_preprocess -d $d -c $CONFIG -np 16 --verify_dataset_integrity
    done ;;
  train)
    echo "=== train Dataset$DATASET fold 0 on GPU$GPU ==="
    CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_train $DATASET $CONFIG 0 --npz ;;
  *) echo "usage: $0 {preprocess|train}"; exit 1 ;;
esac
