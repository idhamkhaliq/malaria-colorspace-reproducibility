# Runtime limitations

The archived experiment directly records Python 3.13.15, TensorFlow 2.20.0, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.16.3, Matplotlib 3.10.0, Keras 3.13.2 in saved-model metadata, and an NVIDIA T4 GPU.

Exact CUDA toolkit, cuDNN, NVIDIA driver, Colab base-image, Linux kernel, and complete transitive dependency versions were not archived and must not be inferred. Full GPU retraining across physical runtimes is therefore not claimed to be bitwise deterministic.
