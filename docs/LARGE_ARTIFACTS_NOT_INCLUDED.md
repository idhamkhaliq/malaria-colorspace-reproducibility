# Large or externally hosted artifacts not included

The compact repository package intentionally excludes:

- raw malaria image archive (`cell_images.zip`; recorded SHA-256 below);
- 25 original CNN `best_model.keras` archives and associated checkpoints;
- 20 canonical CNN `best_model.keras` archives and associated checkpoints;
- the complete 40 main-run prediction-archive set, which was verified in the persistent experiment storage but is not locally available in this assembly environment;
- private credentials such as `kaggle.json` (must never be distributed).

Known external/persistent hashes:
- dataset ZIP: `b751d27bae4bda39c656052f07b6bff26e3815414434f330dc4cfbbb823ad008`
- canonical MobileNetV2 no-top pretrained weights: `f8aff69536bd77a692c594f559c798c19bf7f3f36668fc9fa00b21c6aab4797c`

For a maximal archival release, the model and prediction artifacts may be deposited separately if repository quota and licensing permit, and then cross-linked by DOI.
