#!/usr/bin/env bash
# nnU-Net test of the pre-contrast penalty at a fixed budget of three phases:
# Dataset522 = P1+P2+P3 (no pre) versus Dataset523 = P0+P1+P2 (with pre).
# Three folds per dataset, all sharing the pinned split, so folds act as seeds.
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"

while [ ! -f "$nnUNet_preprocessed/Dataset523_PhaseSel_p012/nnUNetPlans.json" ]; do sleep 120; done
python - <<'PY'
import json, pathlib
p = pathlib.Path("nnunet/nnUNet_preprocessed/Dataset523_PhaseSel_p012/splits_final.json")
s = json.loads(p.read_text())
if len(s) < 3:
    p.write_text(json.dumps([s[0], dict(s[0]), dict(s[0])], indent=1))
c = json.load(open("nnunet/nnUNet_preprocessed/Dataset523_PhaseSel_p012/nnUNetPlans.json"))
n = c["configurations"]["3d_fullres"]["normalization_schemes"]
assert set(n) == {"NoNormalization"}, n
print("523 plans OK:", n)
PY
[ $? -ne 0 ] && { echo "523 plans FAILED the NoNormalization check; aborting"; exit 1; }

i=0
for spec in 522:1 522:2 523:0 523:1 523:2; do
  D=${spec%%:*}; F=${spec##*:}
  g=$(( i % 4 + 4 ))            # GPUs 4-7
  echo "=== train Dataset$D fold $F on GPU$g ==="
  CUDA_VISIBLE_DEVICES=$g nnUNetv2_train "$D" 3d_fullres "$F" --npz \
      > "logs/nnunet_${D}_fold${F}.log" 2>&1 &
  i=$((i+1)); sleep 60
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 300; done
done
wait
echo "=== PENALTY ANCHOR TRAINING COMPLETE ==="
