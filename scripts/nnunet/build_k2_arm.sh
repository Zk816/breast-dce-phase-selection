#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export nnUNet_raw="$PWD/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/nnunet/nnUNet_results"
export PATH="/path/to/nnunet_venv/bin:$PATH"
echo "$(date '+%F %T') converting raw volumes"
python scripts/nnunet/convert_phasesel_ispy2.py      || exit 1
echo "$(date '+%F %T') applying shared normalisation"
python scripts/nnunet/normalize_phasesel_datasets.py || exit 1
for D in 524 525 526 527 528; do
  echo "$(date '+%F %T') plan+preprocess $D"
  nnUNetv2_plan_and_preprocess -d $D -c 3d_fullres -np 8 --verify_dataset_integrity || exit 1
  python - <<PY || exit 1
import json,glob,pathlib
p=glob.glob("nnunet/nnUNet_preprocessed/Dataset${D}_*/nnUNetPlans.json")[0]
n=json.load(open(p))["configurations"]["3d_fullres"]["normalization_schemes"]
assert set(n)=={"NoNormalization"}, ("BAD NORM", n)
sp=pathlib.Path(p).with_name("splits_final.json")
s=json.loads(sp.read_text())
if len(s)<3: sp.write_text(json.dumps([s[0],dict(s[0]),dict(s[0])],indent=1))
print("  ${D} plans OK:", n[0], "| folds:", len(json.loads(sp.read_text())))
PY
done
echo "$(date '+%F %T') K2 ARM DATASETS READY"
