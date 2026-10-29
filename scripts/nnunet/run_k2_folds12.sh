#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
export nnUNet_n_proc_DA=8
GPUS=(1 2 3 5 6 7)
DATASETS=(520 524 525 526 527 528)
for F in 1 2; do
  echo "$(date '+%F %T') === round: fold $F ==="
  for i in "${!DATASETS[@]}"; do
    d=${DATASETS[$i]}; g=${GPUS[$i]}
    echo "$(date '+%F %T') launch dataset $d fold $F on GPU$g"
    CUDA_VISIBLE_DEVICES=$g nohup nnUNetv2_train "$d" 3d_fullres "$F" --npz \
        > "logs/nnunet_${d}_fold${F}.log" 2>&1 &
    sleep 20
  done
  wait
  echo "$(date '+%F %T') === fold $F complete ==="
done
echo "$(date '+%F %T') ALL K2 FOLDS 1-2 COMPLETE"
