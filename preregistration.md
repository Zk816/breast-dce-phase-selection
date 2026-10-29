# PRE-REGISTRATION — Phase Selection in Breast DCE-MRI

**Written 2026-07-30, before the sweep launched. Git-tagged. Frozen.**
Protocol: `exp_phase_sel_final.md`. Evidence base: `paper_docs/README.md`.

---

## 1. Claim under test

> Within a cohort that acquires the contrast wash-in, breast tumour segmentation needs
> approximately two post-contrast phases; the pre-contrast phase contributes little once a
> post-contrast phase is available; and most multi-centre collections do not acquire the phases
> needed to test this.

## 2. Confirmatory test — ONE test, protected

**`p12` (learned pair) vs `p01` (clinical pair: pre + first post), within-ISPY2 test set,
paired case-level bootstrap over the same 196 test cases, n = 1000.**

- Metric: per-case foreground Dice, mean over cases, via `scripts/unified_eval.py`.
- Comparison: `scripts/paired_bootstrap.py` (unit-tested, `scripts/test_paired_bootstrap.py`).
- Reported: mean paired difference in Dice points, 95% CI, % of cases improved.
- **Reported regardless of direction, magnitude, or significance.**
- Two seeds per configuration (42, 43); the confirmatory number is computed per seed and pooled,
  with both reported.

`p12` was chosen by Stage 1 on the *hold-DUKE cross-cohort* fold before this protocol existed
(3 seeds: {1,2}, {1,2}, {2,3}). It is therefore a genuine a-priori prediction with respect to the
within-ISPY2 experiment, not a post-hoc pick from the ISPY2 sweep.

## 3. Exploratory — descriptive only, no p-values

- The full 6-pair ranking (15 possible pairwise comparisons).
- The Dice-vs-k curve (k = 1, 2, 3, 4).
- The DUKE within-cohort confirmation (60 test cases).
- Selector stability across 5 seeds.

These are reported as rankings with confidence intervals. **Selecting the largest of 15 gaps post
hoc is not a result and will not be presented as one.**

## 4. Decision branches — framing only; the protocol does not change

| Outcome | Framing |
|---|---|
| Post-only pairs beat pre-containing, `p12` best | Which two phases matter; clinical default is not optimal |
| Post-only pairs win, all roughly equal | Pre-contrast redundant; any two post-contrast phases suffice |
| No pair differs, k=2 ≈ k=4 | Two phases suffice on the cohort that acquires them |
| k=2 below k=4 | Dice-vs-k curve: temporal information requirement, quantified |

**We will not retroactively claim to have predicted whichever branch we land in.**

## 5. Fixed before launch — deviations invalidate a run

- Backbone ResUNet (MONAI UNet, depth 4, 4.7 M params); loss Tversky(0.3, 0.7) + Focal(0.75, γ=2);
  Adam lr 0.01, PolynomialLR power 1.0; 200 epochs; bf16-mixed; grad clip 1.0.
- batch 2, patch 128³, hard-coded and printed in every log.
- Raw magnitude input, single shared normalization across the phases the model receives.
- Split files `splits/withinISPY2_K4.json` (688/96/196) and `splits/withinDUKE_K3.json` (203/28/60),
  seed 42, patient-level, lesion-volume-quartile stratified, verified zero overlap.
- Checkpoint selected on `val` Dice only. The `test` split is scored **once**, at the end.
- Action space = first 4 phases (indices 0–3).
- No post-processing, no TTA, no ensembling, in any run.
- **What varies between runs: the phase list and the training seed. Nothing else.**

## 6. Convergence rule, declared in advance

`best_epoch` is logged for every run. **If any run peaks at epoch ≥ 190, the budget is extended for
ALL runs, not just that one.** (One earlier cross-cohort run already peaked at 198.)

## 7. Backbone contingency, declared in advance

The C2 baseline (`p0123`, within-ISPY2, ResUNet) simultaneously serves as the paper's baseline row
and as the answer to the backbone question. Prior evidence: our ResUNet scored **3.77 Dice points
below nnU-Net** cross-cohort (paired CI [−5.42, −2.18]) while using more input phases, at 1/5 the
training epochs.

**The plan is: keep the 42 ResUNet runs and add 2 nnU-Net runs as a ceiling anchor. A full
re-sweep on nnU-Net is a FALLBACK for a catastrophic C2, not the expected path.**

| C2 result (ResUNet, within-ISPY2 test) | the 42 runs | nnU-Net runs |
|---|---|---|
| **reasonable** — test Dice > 0.70 and within a few points of nnU-Net | **kept as the main result** | **+2** (`p0123`, `p12`) as a ceiling anchor |
| **catastrophically low** — ≥10 Dice points below nnU-Net | sweep re-run on nnU-Net | full sweep |

The 2 nnU-Net runs are required in **both** branches, so the nnU-Net build starts in parallel and
does not wait for the sweep.

### 7.1 Two ambiguities in the rule above, resolved here rather than after seeing the number

1. **The middle band.** "Within a few points" and "≥10 points below" leave 5–8 points undefined, and
   the prior cross-cohort gap was 3.77, so landing there is plausible. Declared resolution: the
   **42 runs are kept as the main result whenever C2 > 0.70 absolute**, and the nnU-Net anchor is
   reported alongside as measured headroom. A full re-sweep is triggered **only** by the ≥10-point
   catastrophic case, **or** by the 2 nnU-Net runs disagreeing with ResUNet **in the direction** of
   the `p12` vs `p01` effect (not merely in magnitude). Direction disagreement, not size, is what
   would make the ranking backbone-dependent and therefore unpublishable as stated.
2. **Circularity.** "Within a few points of nnU-Net" cannot be evaluated without the nnU-Net
   within-ISPY2 number, which is itself one of the 2 added runs. The **absolute > 0.70 criterion is
   self-sufficient and is the operative test**; the relative comparison is reported descriptively
   once nnU-Net `p0123` exists. This makes nnU-Net `p0123` the higher priority of the two added runs.

### 7.2 What the anchor is for

Absolute parity with nnU-Net is **not** the bar. The two claims have different exposure:

- *"learned pair beats clinical pair"* is a **relative** comparison inside one backbone and survives
  the backbone sitting below ceiling, provided it can express the difference at all.
- *"k=2 matches k=4"* is **vulnerable**: below ceiling, phases 0 and 3 may carry information the
  model cannot extract, which would appear as false saturation. The nnU-Net `p0123` vs `p12` pair is
  what tests this, and it is the reason the anchor is needed in every branch.

## 8. Scope limitation, declared as a finding rather than a defect

The primary result is single-cohort (ISPY2) because ISPY2 is the **only** MAMA-MIA cohort that
acquires the wash-in. Measured fraction of cases with any phase in 60–180 s:
**ISPY2 82%, DUKE 21%, ISPY1 10%, NACT 0%** (`results/acq_time_report.json`). Phase-index labels
cross over between cohorts (DUKE idx2 = 327 s is later than ISPY2 idx3 = 313 s), so cross-cohort
phase comparison is impossible in principle, not merely underpowered.
