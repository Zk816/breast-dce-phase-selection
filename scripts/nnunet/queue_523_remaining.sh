#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
export nnUNet_n_proc_DA=8

running() { ps -u "$USER" -o args= | grep -c "[n]nUNetv2_train $1 3d_fullres $2"; }

for F in 1 2; do
  while [ "$(ps -u "$USER" -o args= | grep -c '[n]nUNetv2_train 52[23] 3d_fullres')" -gt 0 ] \
     && [ "$(ps -u "$USER" -o args= | grep -oE '[n]nUNetv2_train 52[23] 3d_fullres [0-9]' | sort -u | wc -l)" -ge 3 ]; do
    sleep 300
  done
  g=$(( F + 1 ))
  echo "$(date '+%F %T') launching 523 fold $F on GPU$g"
  CUDA_VISIBLE_DEVICES=$g nohup nnUNetv2_train 523 3d_fullres "$F" --npz \
      > "logs/nnunet_523_fold${F}.log" 2>&1 &
  sleep 120
done
wait
echo "$(date '+%F %T') 523 folds 1 and 2 complete"
