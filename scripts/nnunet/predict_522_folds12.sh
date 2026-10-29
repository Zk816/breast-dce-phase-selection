#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
GT=/path/to/MAMA-MIA_v3/labelsTr
g=0
for F in 1 2; do
  OUT=results/nnunet_anchor_522_fold$F; mkdir -p "$OUT/pred"
  echo "=== predict 522 fold $F on GPU$g ==="
  CUDA_VISIBLE_DEVICES=$g nnUNetv2_predict -i "$nnUNet_raw/Dataset522_PhaseSel_p123/imagesTs" \
      -o "$OUT/pred" -d 522 -c 3d_fullres -f $F --disable_tta -npp 4 -nps 4 \
      > logs/predict_522_fold$F.log 2>&1 &
  g=$((g+1)); sleep 10
done
wait
for F in 1 2; do
  OUT=results/nnunet_anchor_522_fold$F
  "$PWD/.venv/bin/python" scripts/unified_eval.py --pred_dir "$OUT/pred" --gt_dir "$GT" \
      --out "$OUT/per_case.csv" --json "$OUT/test_eval.json" > logs/score_522_fold$F.log 2>&1
  echo "fold $F scored: $(wc -l < $OUT/per_case.csv) rows"
done
echo "522 FOLD SEED-SPREAD DONE"
