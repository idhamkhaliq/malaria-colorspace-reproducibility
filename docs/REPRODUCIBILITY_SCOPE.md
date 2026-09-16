# Reproducibility scope

## What this compact public package supports

1. Inspection of the original `[0,1]` experiment notebook and canonical MobileNetV2 preprocessing notebook.
2. Exact verification of the five frozen group-aware split manifests.
3. Reproduction of the 30-run `[0,1]`/control aggregate results from the archived result table.
4. Reproduction of paper-facing 2×2×2 aggregate tables, Holm-corrected contrasts, calibration summaries, control summaries, and split summaries.
5. Inspection of recorded audit summaries for prediction integrity, model loading, split leakage, canonical persistence, and near-duplicate screening.

## What is not independently re-computable from this compact release alone

- the 40 individual main-run prediction archives are not all present in the working container used to assemble this package;
- the 25 original and 20 canonical Keras model archives are not duplicated here;
- the raw 27,558-image dataset is not redistributed;
- exact numeric per-run 2,000-replicate bootstrap interval files were not independently recovered during manuscript assembly;
- the exact implementation of the secondary similarity score recorded as “SSIM” in the final near-duplicate audit has not been recovered and must not be represented as canonical local-window SSIM without verification.

The included aggregate canonical tables are publication-facing verified outputs, not a substitute for the absent per-run canonical prediction archives.
