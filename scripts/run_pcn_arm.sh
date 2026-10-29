#!/usr/bin/env bash
# Per-channel-normalisation arm: 6 k=2 subsets x 3 seeds, within-ISPY2.
# Pinned to GPUs that our nnU-Net anchor is not using; one job per GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
GPUS=(1 2 3 6 7)
RUNS=$(ls experiments/phasesel_20260825_pcn_*.yaml | sed 's|experiments/||;s|\.yaml||')
i=0
for r in $RUNS; do
  while :; do
    for g in "${GPUS[@]}"; do
      if [ ! -f "/tmp/pcn_gpu_$g.lock" ] || ! kill -0 "$(cat /tmp/pcn_gpu_$g.lock 2>/dev/null)" 2>/dev/null; then
        echo "$(date '+%F %T') launch $r on GPU$g"
        CUDA_VISIBLE_DEVICES=$g nohup python -u -m bcaitu_mri_seg fit \
          --config "experiments/$r.yaml" > "logs/$r.log" 2>&1 &
        echo $! > "/tmp/pcn_gpu_$g.lock"
        sleep 45
        continue 3
      fi
    done
    sleep 60
  done
done
wait
echo "$(date '+%F %T') PCN ARM COMPLETE"
