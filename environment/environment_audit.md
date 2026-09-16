# Environment Reconstruction Audit — Malaria Color-Space Experiment

## Verdict

**Environment provenance: STRONG / publication-usable, with one explicit limitation.**

The original notebook contains executed output that records the core software versions used during the experiment. Colab metadata also records a GPU accelerator of type **NVIDIA T4**. Therefore, the core Python/ML environment can be reconstructed without guessing package versions.

## 1. Directly verified from executed notebook output

| Component | Verified version / value | Evidence class |
|---|---|---|
| Python | 3.13.15 | EXACT — printed by executed notebook |
| TensorFlow | 2.20.0 | EXACT — printed by executed notebook |
| NumPy | 2.1.3 | EXACT — printed by executed notebook |
| pandas | 2.2.3 | EXACT — printed by executed notebook |
| scikit-learn | 1.6.1 | EXACT — printed by executed notebook |
| SciPy | 1.16.3 | EXACT — printed by executed notebook |
| Matplotlib | 3.10.0 | EXACT — printed by executed notebook |
| GPU availability | TensorFlow detected 1 GPU | EXACT — printed by executed notebook |
| Colab GPU type | NVIDIA T4 | EXACT — notebook metadata |

## 2. Directly verified from saved model archive

A saved `best_model.keras` archive was opened successfully. Its internal `metadata.json` records:

- `keras_version`: **3.13.2**
- `date_saved`: **2026-08-29 15:17:27**

The archive also contains `config.json` and `model.weights.h5` and has a valid Keras v3 archive structure.

## 3. Not recorded — must NOT be invented

The following exact values were not found in the saved notebook/runtime evidence:

- CUDA toolkit version
- cuDNN version
- NVIDIA driver version
- exact Colab base-image identifier
- exact Linux kernel / OS image build
- all transitive `pip freeze` dependencies

These should be reported as **not archived in the original experiment**, not reconstructed by assumption.

## 4. Important observation

No `pip install`, `%pip`, CUDA installation, or cuDNN installation cell was found in the notebook. This indicates the experiment used the libraries already available in the Colab runtime rather than a custom package-installation step.

## 5. Recommended reproduction modes

### Mode A — Metric/result reproduction (highest confidence)
No GPU or original TensorFlow runtime is required. Use the saved `preds.npz` files and recompute metrics. This path has already passed 30/30 binary-integrity checks.

### Mode B — Model loading / inference reproduction
Use Python 3.13 with the exact pinned TensorFlow/Keras versions in `requirements_verified.txt`. Validate model loading and inference from the saved `.keras` artifacts.

### Mode C — Full retraining reproduction
Use the exact package pins, the saved 5 group-aware split CSVs, the original notebook code and configuration. Because the exact CUDA/cuDNN/driver stack was not archived, small numerical/non-deterministic differences may occur even with identical seeds.

## 6. Hashes for immutable provenance

SHA-256 values computed during the audit:

- Notebook: `d4c50c77367b861232b12ebae589a387765e7638ef3f149c047cca249916410f`
- split_0_seed42.csv: `b18ed348f559c2f90f5cc71806c5302296bea7af6490f08dbe3532c0ca3241a6`
- split_1_seed123.csv: `9698fae7fa62a191846dfe6196af6a745c8dc6af5001d9dbca6e5a30efeaec41`
- split_2_seed7.csv: `0719562d2623fd3602cd7f80ca174bd2bb4962eaf314caabcbb6ceb2cdd3352f`
- split_3_seed2024.csv: `1fbe3b529d8359e240d4fb2294cc7dccb8152c529469fdcf8823350a49321f86`
- split_4_seed99.csv: `9dee1e6abb0e17b20b8c0e12861584a3b9ca9ba080aa5c44ec229d7fab028673`
- all_results.csv: `0d99e0ae039efaba05a775333d4e7ea937e346097fd520fdde072a6c24acc358`
- audited RGB-FT split-0 best_model.keras: `5828b0a3b8837bc452692ae80d5659970b9dfa790355211edbcfe7c753762914`

Hashes should be retained in the reproducibility package to detect accidental modification of the archived experiment sources.

## 7. Publication-ready wording

> Experiments were executed in Google Colab using Python 3.13.15 and TensorFlow 2.20.0. The recorded runtime included NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.16.3, and Matplotlib 3.10.0. Colab notebook metadata indicates an NVIDIA T4 GPU, and TensorFlow detected one GPU device during execution. Saved Keras model archives report Keras 3.13.2. Exact CUDA, cuDNN, and NVIDIA driver versions were not archived and are therefore not claimed.

## 8. Reproducibility status after this audit

- Package versions needed for analysis/training code: **verified**
- Python version: **verified**
- GPU family: **verified**
- Keras model-save version: **verified**
- CUDA/cuDNN/driver versions: **not archived**
- Metric reproduction from raw predictions: **30/30 PASS**
- Group/data leakage audit: **5/5 PASS**

This is sufficient to construct a transparent publication reproducibility package, provided the missing low-level GPU stack is disclosed rather than inferred.
