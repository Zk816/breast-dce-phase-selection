#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
GT=/path/to/MAMA-MIA_v3/labelsTr
for d in results/phasesel_20260825_pcn_*/pred; do
  r=$(basename "$(dirname "$d")")
  [ -f "results/$r/per_case.csv" ] && continue
  "$PWD/.venv/bin/python" scripts/unified_eval.py --pred_dir "$d" --gt_dir "$GT" \
      --out "results/$r/per_case.csv" --json "results/$r/test_eval.json" \
      > "logs/score_$r.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge 9 ]; do sleep 5; done
done
wait
echo "FAST SCORING COMPLETE: $(ls results/phasesel_20260825_pcn_*/per_case.csv | wc -l)/18"
