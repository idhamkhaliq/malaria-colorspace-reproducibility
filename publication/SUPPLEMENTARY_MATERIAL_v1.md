# Supplementary Material

## S1. Experimental Design and Reproducibility Scope

The primary experiment used a 2 × 2 × 2 factorial design:
- representation: RGB vs grayscale;
- training: frozen vs fine-tuned;
- preprocessing: `[0,1]` vs canonical MobileNetV2;
- evaluation: five repeated group-aware partitions.

This produced 40 main CNN runs. Five global color-statistics Logistic Regression controls and five Gaussian-blurred CNN controls produced 10 additional runs, for 50 training/evaluation runs in total.

The five repeated partitions originate from the same dataset and are not treated as independent external cohorts.

## S2. Dataset and Partition Integrity

The logical dataset contains 27,558 images: 13,779 parasitized and 13,779 uninfected. Group parsing yielded 420 inferred groups: 148 patient-like identifiers and 272 fallback image/acquisition identifiers.

| Split | Seed | Train | Validation | Test |
|---:|---:|---:|---:|---:|
| 0 | 42 | 17,576 | 4,402 | 5,580 |
| 1 | 123 | 21,097 | 3,240 | 3,221 |
| 2 | 7 | 20,536 | 3,648 | 3,374 |
| 3 | 2024 | 19,400 | 4,900 | 3,258 |
| 4 | 99 | 19,214 | 3,616 | 4,728 |

Exact group and filepath overlap checks passed for all five partitions.

## S3. Main Factorial Results

The full aggregate numerical table is supplied in `Malaria_Paper_Final_Tables.xlsx` (Table2_MainFactorial). The principal structural pattern is:
1. RGB > grayscale on mean accuracy in all four matched preprocessing/training comparisons.
2. Fine-tuning increases mean accuracy in all four representation/preprocessing cells.
3. The RGB–grayscale gap is substantially larger under a frozen backbone than under fine-tuning.
4. Switching from `[0,1]` to canonical preprocessing does not uniformly improve absolute performance.

## S4. Paired Statistical Contrasts and Holm Correction

### S4.1 RGB minus grayscale

| Preprocessing   | Training   | Metric   |   Mean_diff |    CI_low |   CI_high |    p_raw |   p_Holm | Reject_Holm   |
|:----------------|:-----------|:---------|------------:|----------:|----------:|---------:|---------:|:--------------|
| [0,1]           | Frozen     | Accuracy |    0.025317 |  0.019255 |  0.03138  | 0.000316 | 0.003161 | True          |
| [0,1]           | Frozen     | Recall   |    0.024417 | -0.004201 |  0.053036 | 0.076918 | 0.153837 | False         |
| [0,1]           | Frozen     | AUC      |    0.008871 |  0.006112 |  0.01163  | 0.00087  | 0.007834 | True          |
| [0,1]           | Fine-tuned | Accuracy |    0.00441  |  0.002501 |  0.006319 | 0.003037 | 0.024298 | True          |
| [0,1]           | Fine-tuned | Recall   |    0.009272 |  0.004688 |  0.013856 | 0.00494  | 0.034581 | True          |
| [0,1]           | Fine-tuned | AUC      |    0.001686 |  0.000674 |  0.002699 | 0.00985  | 0.04925  | True          |
| Canonical       | Frozen     | Accuracy |    0.029626 |  0.02405  |  0.035202 | 0.000123 | 0.001475 | True          |
| Canonical       | Frozen     | Recall   |    0.03132  |  0.014422 |  0.048218 | 0.006763 | 0.040578 | True          |
| Canonical       | Frozen     | AUC      |    0.013054 |  0.010343 |  0.015765 | 0.000181 | 0.001991 | True          |
| Canonical       | Fine-tuned | Accuracy |    0.005559 |  0.001813 |  0.009305 | 0.014615 | 0.05846  | False         |
| Canonical       | Fine-tuned | Recall   |    0.006222 | -0.002171 |  0.014616 | 0.108666 | 0.153837 | False         |
| Canonical       | Fine-tuned | AUC      |    0.002988 |  0.000805 |  0.005172 | 0.0191   | 0.05846  | False         |

### S4.2 Fine-tuned minus frozen

| Preprocessing   | Representation   | Metric   |   Mean_diff |    CI_low |   CI_high | p_raw   |   p_Holm | Reject_Holm   |
|:----------------|:-----------------|:---------|------------:|----------:|----------:|:--------|---------:|:--------------|
| [0,1]           | Grayscale        | Accuracy |    0.035638 |  0.027427 |  0.043849 |         | 0.002719 | True          |
| [0,1]           | Grayscale        | Recall   |    0.016789 | -0.022582 |  0.056159 |         | 0.603958 | False         |
| [0,1]           | Grayscale        | AUC      |    0.012409 |  0.008208 |  0.01661  |         | 0.008429 | True          |
| [0,1]           | RGB              | Accuracy |    0.01473  |  0.012993 |  0.016468 |         | 0.000232 | True          |
| [0,1]           | RGB              | Recall   |    0.001643 | -0.01092  |  0.014206 |         | 0.734869 | False         |
| [0,1]           | RGB              | AUC      |    0.005225 |  0.002847 |  0.007602 |         | 0.021893 | True          |
| Canonical       | Grayscale        | Accuracy |    0.041696 |  0.034839 |  0.048552 |         | 0.000793 | True          |
| Canonical       | Grayscale        | Recall   |    0.035295 |  0.012676 |  0.057913 |         | 0.049307 | True          |
| Canonical       | Grayscale        | AUC      |    0.015147 |  0.011618 |  0.018677 |         | 0.002719 | True          |
| Canonical       | RGB              | Accuracy |    0.017629 |  0.012624 |  0.022633 |         | 0.0049   | True          |
| Canonical       | RGB              | Recall   |    0.010197 |  0.001541 |  0.018853 |         | 0.092303 | False         |
| Canonical       | RGB              | AUC      |    0.005082 |  0.002048 |  0.008116 |         | 0.048305 | True          |

### S4.3 Canonical minus `[0,1]`

| Representation   | Training   | Metric   |   Mean_diff_canonical_minus_0_1 |    CI_low |   CI_high |      p_raw |   p_Holm | Reject_Holm   |
|:-----------------|:-----------|:---------|--------------------------------:|----------:|----------:|-----------:|---------:|:--------------|
| Grayscale        | Frozen     | Accuracy |                       -0.004859 | -0.010299 |  0.000581 | nan        | 0.613931 | False         |
| Grayscale        | Frozen     | Recall   |                       -0.010342 | -0.029081 |  0.008396 | nan        | 0.932328 | False         |
| Grayscale        | Frozen     | AUC      |                       -0.003553 | -0.004942 | -0.002164 |   0.002076 | 0.024908 | True          |
| Grayscale        | Fine-tuned | Accuracy |                        0.001199 | -0.002846 |  0.005244 | nan        | 1        | False         |
| Grayscale        | Fine-tuned | Recall   |                        0.008164 | -0.006068 |  0.022395 | nan        | 0.932328 | False         |
| Grayscale        | Fine-tuned | AUC      |                       -0.000815 | -0.001793 |  0.000163 | nan        | 0.613931 | False         |
| RGB              | Frozen     | Accuracy |                       -0.00055  | -0.005006 |  0.003905 | nan        | 1        | False         |
| RGB              | Frozen     | Recall   |                       -0.00344  | -0.012273 |  0.005394 | nan        | 1        | False         |
| RGB              | Frozen     | AUC      |                        0.00063  | -3.2e-05  |  0.001292 | nan        | 0.574258 | False         |
| RGB              | Fine-tuned | Accuracy |                        0.002348 | -0.000316 |  0.005013 | nan        | 0.613931 | False         |
| RGB              | Fine-tuned | Recall   |                        0.005114 |  0.000603 |  0.009625 |   0.034589 | 0.380475 | False         |
| RGB              | Fine-tuned | AUC      |                        0.000487 | -0.00019  |  0.001165 | nan        | 0.698937 | False         |

These paired comparisons use the same five repeated partitions. Confidence intervals and p values should therefore be interpreted as repeated-partition evidence on one dataset rather than external-population validation.

## S5. Calibration and Probability Quality

| Preprocessing   | Representation   | Training   |   ECE_mean |   ECE_SD |   MCE_mean |   MCE_SD |   Brier_mean |   Brier_SD |
|:----------------|:-----------------|:-----------|-----------:|---------:|-----------:|---------:|-------------:|-----------:|
| Canonical       | Grayscale        | Frozen     |   0.021237 | 0.006049 |   0.16565  | 0.074533 |     0.061997 |   0.009832 |
| Canonical       | Grayscale        | Fine-tuned |   0.018178 | 0.007747 |   0.231988 | 0.065324 |     0.033344 |   0.010094 |
| Canonical       | RGB              | Frozen     |   0.015922 | 0.007838 |   0.162265 | 0.057533 |     0.040759 |   0.010232 |
| Canonical       | RGB              | Fine-tuned |   0.014617 | 0.006098 |   0.220459 | 0.050684 |     0.029125 |   0.009037 |
| [0,1]           | Grayscale        | Frozen     |   0.030463 | 0.011881 |   0.200594 | 0.07126  |     0.058129 |   0.009994 |
| [0,1]           | Grayscale        | Fine-tuned |   0.018563 | 0.011098 |   0.296794 | 0.069748 |     0.033596 |   0.01031  |
| [0,1]           | RGB              | Frozen     |   0.018107 | 0.006722 |   0.166307 | 0.074873 |     0.0413   |   0.009952 |
| [0,1]           | RGB              | Fine-tuned |   0.016774 | 0.008322 |   0.232066 | 0.083637 |     0.030491 |   0.009957 |

ECE and MCE used 15 equal-width probability bins. Brier, ECE, and MCE are descriptive probability-quality measures; they do not establish clinical calibration.

## S6. Bootstrap Analysis

A stratified 2,000-replicate bootstrap was completed for all 40 main runs to characterize per-run uncertainty without retraining. Exact numeric per-run bootstrap interval files were not independently re-verified in the current manuscript assembly environment, so this supplement does not invent or reproduce unavailable interval values. If the original bootstrap CSV is recovered, it should be added as a machine-readable supplementary table without changing the main paired-contrast analysis.

## S7. Anti-Shortcut Controls

| Control | Accuracy | ROC-AUC | Interpretation |
|---|---:|---:|---|
| Global color-statistics Logistic Regression | ~0.814 | ~0.8668 | Global color summary alone remains substantially predictive. |
| Gaussian-blur CNN | ~0.9497 | ~0.9862 | High performance persists after strong spatial-detail degradation. |

These controls are stress tests. They do not causally identify stain, background, illumination, device, or acquisition pipeline as the source of a shortcut.

## S8. Exact-Duplicate and Near-Duplicate Audit

| Audit_component                                              |   Result |
|:-------------------------------------------------------------|---------:|
| Logical images                                               |    27558 |
| Raw extracted image files                                    |    55116 |
| Extraction-copy pixel mismatches                             |        0 |
| Global exact logical duplicate pairs                         |        0 |
| Exact cross-partition duplicate pairs                        |        0 |
| Global pHash candidates (Hamming ≤4)                         |     2962 |
| Cross-partition pHash occurrences across 5 splits            |     6875 |
| Unique pHash candidate pairs crossing ≥1 partition           |     2813 |
| Same inferred group among cross-partition candidates         |        0 |
| Similarity threshold ≥0.98 among cross-partition candidates  |        0 |
| Similarity threshold ≥0.995 among cross-partition candidates |        0 |

The extraction tree contained two decoded-pixel-identical copies of each logical basename. Analyses therefore use 27,558 logical images, not 55,116 independent observations. No exact logical duplicates were detected. Perceptual-hash candidates were subsequently characterized; none of the cross-partition candidates shared the same inferred group or reached the operational 0.98 similarity threshold.

**Important pre-submission item:** verify the exact implementation behind the quantity recorded as SSIM before describing it as standard local-window SSIM in the final journal submission.

## S9. Threshold, Prevalence, and Computational Characterization

For the selected RGB fine-tuned configuration, a validation-selected threshold reduced aggregate false negatives from 529 to 470, a net reduction of approximately 11.15%. This improvement was not uniform across splits and is therefore reported as operating-point characterization rather than a universal sensitivity improvement.

Prevalence analyses are hypothetical simulations and are not external clinical validation.

Computational profiling was performed on a Google Colab T4 GPU. The architecture has 2,619,074 parameters. GPU latency must not be presented as a mobile-CPU or point-of-care benchmark.

## S10. Reproducibility and Persistence

- original prediction recomputation: PASS;
- five frozen split manifests: PASS;
- CNN artifact structural checks: PASS;
- canonical 20/20 prediction archives: PASS;
- fresh-runtime recovery: PASS;
- canonical full-model deserialization: 20/20 PASS;
- no retraining was performed during post-hoc calibration, bootstrap, Holm, or near-duplicate analyses.

Exact bitwise continuation across physically different GPU runtimes is not claimed.

## S11. Files Recommended for Submission

Main manuscript:
- `MASTER_MANUSCRIPT_Malaria_Bahasa_Indonesia_v5_SCIENTIFIC_AUDITED.md`

Main numerical tables:
- `Malaria_Paper_Final_Tables.xlsx`

Main figure files:
- Figure 1–7 PNG files in this package.

Potential additional supplementary files when available:
- per-run bootstrap interval CSV;
- per-bin reliability/calibration data;
- representative Grad-CAM panels;
- frozen split manifests/hashes;
- reproducibility artifact manifest;
- final near-duplicate audit machine-readable outputs.
