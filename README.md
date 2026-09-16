# Malaria color-representation study — data, code, and reproducibility package

**Release:** 2.0.0 (compact public-deposit candidate)  
**Authors:** Idham Khaliq; Uturestantix  
**Associated manuscript:** *Color Representation, Fine-Tuning, and Shortcut-Sensitive Cues in Deep Learning-Based Malaria Cell Classification: A Controlled Factorial Study*

## Study design represented by this release

The manuscript uses one unified factorial design:

- image representation: RGB vs grayscale;
- training strategy: frozen vs fine-tuned MobileNetV2;
- preprocessing: `[0,1]` vs canonical MobileNetV2 preprocessing;
- five repeated group-aware data partitions.

This yields **2 × 2 × 2 × 5 = 40 main CNN runs**. Five color-only Logistic Regression controls and five Gaussian-blur CNN controls bring the total to **50 training/evaluation runs**.

The five repeated partitions are partitions of the same dataset, not independent cohorts. The training seed was fixed at 42 for the main runs; 42, 123, 7, 2024, and 99 are split seeds.

## Quick start

### A. Verify split integrity

```bash
python code/verify_split_integrity.py
```

Expected: 5/5 split files pass group-overlap and filepath-overlap checks, with 293/64/63 inferred groups in train/validation/test for every split.

### B. Rebuild publication-facing summary tables

```bash
python code/reproduce_publication_tables.py
```

This reads the included machine-readable result tables and writes regenerated summaries to `reproduced_outputs/`.

### C. Inspect full experimental code

- `code/01_original_unit01_experiment.ipynb`
- `code/02_canonical_preprocessing_experiment.ipynb`
- `code/03_nontraining_posthoc_analysis.ipynb`
- `code/03_nontraining_posthoc_analysis.py`
- `code/04_reproducibility_audit.ipynb`

## Directory layout

```text
code/          notebooks, analysis scripts, verification utilities
data_splits/   exact frozen split manifests for all five partitions
results/       original 30-run table + verified publication-facing summaries
audit/         integrity, model-load, canonical, and near-duplicate summaries
environment/   verified environment provenance and version pins
publication/   manuscript/supplement snapshot files useful for traceability
docs/          availability statements, scope, and artifact exclusions
provenance/    SHA-256 inventory and known external hashes
```

## Important reproducibility boundary

This is a **compact public-deposit candidate**, not a claim that every large model/prediction artifact is contained in the archive. See `docs/REPRODUCIBILITY_SCOPE.md` and `docs/LARGE_ARTIFACTS_NOT_INCLUDED.md`.

The raw dataset is not redistributed. Exact split manifests are included. The full model/prediction archives can be linked as a separate large-artifact deposit if desired.

## Integrity highlights

- original stored predictions: 30/30 metric-recomputation PASS;
- original CNN full deserialization: 25/25 PASS;
- canonical runs: 20/20 complete, 20/20 prediction-integrity PASS;
- fresh-runtime canonical model deserialization: 20/20 PASS;
- split leakage: 5/5 PASS;
- exact cross-partition decoded-pixel duplicate pairs: 0;
- high-similarity cross-partition pairs at the operational >=0.98 threshold: 0.

These results establish internal computational integrity for the archived experiment. They do not establish external clinical validity or transportability to another laboratory, microscope, staining protocol, or patient population.

## Public release checklist

Before uploading to Zenodo/OSF/GitHub:

1. fill the exact dataset URL in `docs/DATA_AVAILABILITY.md`;
2. choose explicit licenses (`Confirm repository licensing and third-party data terms before release.` currently blocks silent license assumptions);
3. optionally add the full prediction/model artifacts as a separately versioned large deposit;
4. upload this frozen release;
5. obtain the repository URL/DOI;
6. replace `<REPOSITORY_URL_OR_DOI>` in the manuscript/code-availability statement;
7. do not modify the deposited release after DOI assignment—create a new version instead.
