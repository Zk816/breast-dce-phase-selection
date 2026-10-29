#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
export nnUNet_n_proc_DA=8

P="$nnUNet_preprocessed/Dataset523_PhaseSel_p012"
python - <<'PY' || exit 1
import json, pathlib, sys
p = pathlib.Path("nnunet/nnUNet_preprocessed/Dataset523_PhaseSel_p012")
plans = json.loads((p / "nnUNetPlans.json").read_text())
n = plans["configurations"]["3d_fullres"]["normalization_schemes"]
assert set(n) == {"NoNormalization"}, n
s = json.loads((p / "splits_final.json").read_text())
need = sorted(set(s[0]["train"]) | set(s[0]["val"]))
d = p / "nnUNetPlans_3d_fullres"
missing = [c for c in need if not (d / f"{c}.b2nd").exists() or not (d / f"{c}_seg.b2nd").exists()]
assert not missing, f"{len(missing)} preprocessed cases missing, first: {missing[:3]}"
assert all(s[0]["train"] == x["train"] and s[0]["val"] == x["val"] for x in s), "folds differ"
print(f"523 ready: {len(need)} cases preprocessed, {len(s)} identical folds, {n[0]}")
PY

i=0
for F in 0 1 2; do
  g=$(( i ))
  echo "=== train Dataset523 fold $F on GPU$g ==="
  CUDA_VISIBLE_DEVICES=$g nohup nnUNetv2_train 523 3d_fullres "$F" --npz \
      > "logs/nnunet_523_fold${F}.log" 2>&1 &
  i=$((i+1)); sleep 90
done
wait
echo "=== 523 TRAINING COMPLETE ==="
