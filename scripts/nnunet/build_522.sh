#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
VENV=/path/to/nnunet_venv
NN_ROOT="$PWD/nnunet"
export nnUNet_raw="$NN_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NN_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NN_ROOT/nnUNet_results"
export PATH="$VENV/bin:$PATH"
python scripts/nnunet/convert_phasesel_ispy2.py           || exit 1
python scripts/nnunet/normalize_phasesel_datasets.py      || exit 1
nnUNetv2_plan_and_preprocess -d 522 -c 3d_fullres -np 8 --verify_dataset_integrity
