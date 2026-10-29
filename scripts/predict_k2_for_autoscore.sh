#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.venv/bin/python"
GPUS=(1 2 3 5 6 7)
mapfile -t RUNS < "$1"
i=0
for r in "${RUNS[@]}"; do
  g=${GPUS[$(( i % ${#GPUS[@]} ))]}
  while [ "$(jobs -rp | wc -l)" -ge "${#GPUS[@]}" ]; do wait -n; done
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" -u scripts/phase_sel_predict.py --run "$r" --gpu 0 \
    > "logs/predict_${r}.log" 2>&1 &
  echo "launched $r on GPU $g"
  i=$((i+1))
done
wait
echo "ALL PREDICTIONS DONE"
