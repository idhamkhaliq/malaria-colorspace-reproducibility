# Final Figure Legends

**Figure 1. Unified experimental design.** The main factorial design crossed image representation (RGB vs grayscale), training strategy (frozen vs fine-tuned), and preprocessing regime (`[0,1]` vs canonical MobileNetV2) across five repeated group-aware partitions, yielding 40 main CNN runs. Five global color-statistics controls and five Gaussian-blur CNN controls brought the total to 50 training/evaluation runs.

**Figure 2. Fine-tuning narrows the RGB–grayscale accuracy gap.** Mean accuracy ± SD across five repeated partitions is shown for RGB and grayscale under both preprocessing regimes. RGB has a larger advantage under a frozen backbone, whereas fine-tuning substantially reduces the difference.

**Figure 3. Paired RGB–grayscale accuracy contrasts.** Mean paired RGB-minus-grayscale accuracy differences with 95% confidence intervals across the five repeated partitions. Positive values favor RGB. Family-wise multiplicity was handled using Holm correction in the reported inferential analysis.

**Figure 4. Preprocessing effects on accuracy.** Paired canonical-minus-`[0,1]` accuracy differences with 95% confidence intervals. Effects are small and condition-dependent, supporting the interpretation that preprocessing changes magnitude more than the qualitative RGB/fine-tuning pattern.

**Figure 5. Anti-shortcut controls.** Mean performance of the global color-statistics Logistic Regression and Gaussian-blur CNN control. These controls show that substantial predictive signal survives strong information restriction; they do not identify a shortcut source causally.

**Figure 6. Expected calibration error across factorial conditions.** Mean ECE ± SD across five repeated partitions for all eight main factorial conditions. Calibration metrics are descriptive probability-quality measures and are not evidence of clinical calibration.

**Figure 7. Duplicate and near-duplicate audit summary.** Counts from the final exhaustive integrity audit. Perceptual-hash candidates are screening candidates rather than confirmed duplicates; no cross-partition candidate shared the same inferred group or reached the operational similarity threshold of 0.98 used in the audit.
