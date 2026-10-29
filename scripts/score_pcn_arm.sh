#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
GT=/path/to/MAMA-MIA_v3/labelsTr
for d in results/phasesel_20260825_pcn_*/pred; do
  r=$(basename "$(dirname "$d")")
  [ -f "results/$r/per_case.csv" ] && continue
  n=$(ls "$d"/*.nii.gz 2>/dev/null | wc -l)
  [ "$n" -lt 196 ] && { echo "SKIP $r (only $n preds)"; continue; }
  "$PWD/.venv/bin/python" scripts/unified_eval.py --pred_dir "$d" --gt_dir "$GT" \
      --out "results/$r/per_case.csv" --json "results/$r/test_eval.json" --hd95 \
      > "logs/score_$r.log" 2>&1 && echo "scored $r"
done
echo "SCORING COMPLETE"
