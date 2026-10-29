#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
GT=/path/to/MAMA-MIA_v3/labelsTr
g=0
for spec in 520:Dataset520_PhaseSel_p12 521:Dataset521_PhaseSel_p0123 522:Dataset522_PhaseSel_p123; do
  D=${spec%%:*}; NAME=${spec##*:}; OUT=results/nnunet_anchor_$D
  mkdir -p "$OUT/pred"
  echo "=== predict $NAME on GPU$g ==="
  CUDA_VISIBLE_DEVICES=$g nnUNetv2_predict -i "$nnUNet_raw/$NAME/imagesTs" -o "$OUT/pred" \
      -d "$D" -c 3d_fullres -f 0 --disable_tta -npp 4 -nps 4
  g=$((g+1))
done
for D in 520 521 522; do
  OUT=results/nnunet_anchor_$D
  echo "=== score $D : $(ls $OUT/pred/*.nii.gz 2>/dev/null | wc -l) predictions ==="
  "$PWD/.venv/bin/python" scripts/unified_eval.py --pred_dir "$OUT/pred" --gt_dir "$GT" --hd95 \
      --out "$OUT/per_case.csv" --json "$OUT/test_eval.json"
done
echo "=== ANCHOR SCORING DONE ==="
