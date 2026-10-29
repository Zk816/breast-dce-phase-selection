#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
while [ "$(pgrep -cf 'nnUNetv2_train 521')" -gt 0 ]; do sleep 300; done
while [ "$(pgrep -cf 'plan_and_preprocess -d 522')" -gt 0 ]; do sleep 300; done
D=nnunet/nnUNet_preprocessed/Dataset522_PhaseSel_p123
[ -f "$D/nnUNetPlans.json" ] || { echo "522 not preprocessed; aborting"; exit 1; }
python - <<'PY'
import json
c=json.load(open("nnunet/nnUNet_preprocessed/Dataset522_PhaseSel_p123/nnUNetPlans.json"))["configurations"]["3d_fullres"]
assert set(c["normalization_schemes"])=={"NoNormalization"}, c["normalization_schemes"]
print("522 plans OK:", c["normalization_schemes"], c["patch_size"], "batch", c["batch_size"])
PY
[ $? -ne 0 ] && { echo "522 plans FAILED NoNormalization check; aborting"; exit 1; }
GPU=6 DATASET=522 exec bash scripts/nnunet/run_phasesel_anchor.sh train
