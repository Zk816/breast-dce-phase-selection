# Which contrast phases does breast DCE-MRI tumour segmentation need?

Training and analysis code for the exhaustive contrast-phase subset study on breast DCE-MRI:
a learned phase selector, an exhaustive sweep over all fifteen non-empty subsets of four
contrast phases, and the statistical analysis of the results.

The repository contains the model, the run configurations, the data splits, the evaluation and
statistics scripts, the per-case results of every run, and the pre-registered analysis plan.

---

## 1. What is here

```
src/bcaitu_mri_seg/
  net_modules/phase_sel/       selector (straight-through Gumbel top-k), LightningModule, loss
  data_modules/                phase-subset datamodule and the raw-magnitude loader
  net_modules/base.py          shared sliding-window validation
  types/                       batch and metric typed dicts

experiments/                   149 run configurations, one YAML per run
splits/                        the four patient-level splits, generated once with seed 42
scripts/                       training dispatcher, evaluation, statistics
scripts/nnunet/                the nnU-Net anchor arm
preregistration.md             the pre-registered analysis plan
results/                       per-case Dice for every run and the Stage-1 selector outputs,
                               so the reported comparisons can be checked without retraining
```

`results/acq_time_report.json` holds the acquisition-timing summary: ISPY2 is the only
MAMA-MIA cohort whose first post-contrast phase falls inside the contrast wash-in, which is why
the study is within-ISPY2 with DUKE as a smaller confirmation.

## 2. Data

The study uses [MAMA-MIA](https://doi.org/10.1038/s41597-025-04707-4), a public multi-centre
breast DCE-MRI collection with expert-corrected tumour masks. The data is not redistributed
here. Configs and scripts use placeholder paths (`/path/to/MAMA-MIA_v3` for the preprocessed
copy, `/path/to/MAMA-MIA` for the original download, `/path/to/nnunet_venv` for the nnU-Net
environment); replace them with your own locations.

Model input is **raw magnitude** phases resampled onto the processed grid. Intensity
normalisation uses a **single mean and standard deviation shared across all phases the model
receives**, computed over breast-mask voxels. Per-channel normalisation is a confound rather
than a neutral choice, so it is run as a separate declared arm (`pcn` configs) rather than as
the default.

## 3. Installation

```bash
uv sync
```

Python 3.12 or newer. `uv.lock` pins the exact environment the reported runs used.

## 4. Reproducing one run

Training reads all hyperparameters from YAML; nothing is hardcoded.

```bash
mkdir -p logs
nohup python -u -m bcaitu_mri_seg fit \
  --config experiments/phasesel_20260730_s2_p12_ISPY2_seed42.yaml \
  > logs/phasesel_20260730_s2_p12_ISPY2_seed42.log 2>&1 &
tensorboard --logdir tb_logs/phasesel_20260730_s2_p12_ISPY2_seed42 --port 6006
```

Before the first run, build the fast training cache. Without it the loader, not the GPU, is the
bottleneck:

```bash
python scripts/phase_sel_precompute.py            # writes .phase_sel_cache/
python scripts/phase_sel_precompute.py --verify   # checks the cached moments are exact
```

`scripts/phasesel_dispatcher.py` runs the full sweep unattended: it recomputes state from the
process table and the TensorBoard logs on every tick, so restarting it is always safe.

### Configuration naming

`phasesel_{date}_{arm}_{spec}_{cohort}_seed{seed}`

| Arm | Count | Meaning |
|---|---|---|
| `s1_k{k}` | 5 | Stage 1: the selector learns which `k` phases to keep |
| `s2_p{phases}` | 110 | Stage 2: retrain from scratch on a fixed subset (75 ISPY2, 35 DUKE) |
| `s3_sub{phases}` | 10 | subtraction inputs, phases minus pre-contrast |
| `pcn` / `pcnsub` | 24 | per-channel normalisation arm |

149 configurations in total. Stage 2 retrains from scratch on only the selected phases, so the
claim concerns **acquisition**, not channel masking: a learned subset, a fixed subset and a
random subset differ only in which phases enter the network.

## 5. Evaluation

Every method is scored by the same code on the same held-out cases, so no comparison can be an
artefact of differing evaluation.

```bash
# per-case foreground Dice for one run's saved predictions
python scripts/phase_sel_predict.py --run <run_name> --gpu 0
python scripts/unified_eval.py --pred_dir results/<run_name>/pred \
                               --gt_dir <labels> --out results/<run_name>/per_case.csv

# average over seeds first, then compare
python scripts/phase_sel_pool.py --cohort ISPY2 --min_seeds 5
python scripts/paired_bootstrap.py --a results/.../per_case.csv \
                                   --b results/.../per_case.csv
```

Comparisons resample **cases**, not runs, and per-case Dice is averaged over seeds before any
comparison so that seed noise is not inherited from either side. The confirmatory analysis in
`scripts/phase_sel_stats.py` additionally resamples training seeds jointly with cases and
applies Holm correction within each declared family of comparisons.

| Script | Produces |
|---|---|
| `phase_sel_stats.py` | ranking with multiplicity, equivalence curves, the class-level test |
| `phase_sel_stratified.py` | paired differences within lesion-volume, subtype, scanner strata |
| `phase_sel_report.py` | Stage-1 selection readout and cross-seed agreement |
| `score_vs_automatic.py` | rescoring against uncorrected automatic masks |
| `phase_sel_check.py` | selector assertions: exactly `k` selected, non-zero theta gradient |

All bootstrap intervals use the same resample count and seed 42. The per-case scores in
`results/` are included, so the statistics run on a clone with no GPU and no access to the
imaging data.

## 6. Pre-registration

`preregistration.md` is the analysis plan, reproduced unchanged. It was committed to the
authors' version control on 30 July 2026 at 15:24 (UTC+5), together with the launch of the
within-ISPY2 sweep, and amended once at 18:06 the same day, before any test result existed; the
amendment concerns only the backbone fallback rule (§7.1–7.2). A PDF of the plan is deposited at
[Zenodo record 22305161](https://zenodo.org/records/22305161). The plan separates the
confirmatory analysis from the exploratory ones, and the manuscript keeps that separation.
References inside the plan to internal working notes (`exp_phase_sel_final.md`,
`paper_docs/`) point to documents that are not part of this release.

## 7. Citation

If you use this code, please cite the manuscript and the MAMA-MIA benchmark.

## 8. Licence

Apache License 2.0, see `LICENSE`. The MAMA-MIA data is not included and remains under its own
licence.
