#!/usr/bin/env bash
# Finish the per-channel-normalisation arm after the shutdown.
# Lightning saved best-val checkpoints only (no save_last), so partial runs restart.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS="p12_ISPY2_seed43 p12_ISPY2_seed44 p13_ISPY2_seed42 p13_ISPY2_seed43 p13_ISPY2_seed44 p23_ISPY2_seed42 p23_ISPY2_seed43 p23_ISPY2_seed44"
g=0
for r in $RUNS; do
  run="phasesel_20260825_pcn_$r"
  [ -f "experiments/$run.yaml" ] || { echo "MISSING CONFIG $run"; continue; }
  echo "$(date '+%F %T') launch $run on GPU$g"
  CUDA_VISIBLE_DEVICES=$g nohup "$PWD/.venv/bin/python" -u -m bcaitu_mri_seg fit \
    --config "experiments/$run.yaml" > "logs/$run.log" 2>&1 &
  g=$(( (g+1) % 6 )); sleep 40
done
wait
echo "$(date '+%F %T') PCN ARM COMPLETE"
