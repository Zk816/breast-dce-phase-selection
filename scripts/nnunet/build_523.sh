#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
python scripts/nnunet/convert_phasesel_ispy2.py      || exit 1
python scripts/nnunet/normalize_phasesel_datasets.py || exit 1
nnUNetv2_plan_and_preprocess -d 523 -c 3d_fullres -np 8 --verify_dataset_integrity
