#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS=$(ls -d checkpoints/phasesel_20260825_pcn_* | xargs -n1 basename)
g=0
for r in $RUNS; do
  [ -f "results/$r/per_case.csv" ] && { echo "skip $r (scored)"; continue; }
  CUDA_VISIBLE_DEVICES=$g nohup "$PWD/.venv/bin/python" -u scripts/phase_sel_predict.py \
      --run "$r" --gpu 0 > "logs/predict_$r.log" 2>&1 &
  echo "$(date '+%T') predict $r on GPU$g"
  g=$(( (g+1) % 8 ))
  [ $g -eq 0 ] && wait
  sleep 5
done
wait
echo "$(date '+%T') ALL PCN PREDICTIONS DONE"
