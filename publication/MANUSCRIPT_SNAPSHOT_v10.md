# Color Representation, Fine-Tuning, and Shortcut-Sensitive Cues in Deep Learning-Based Malaria Cell Classification: A Controlled Factorial Study

**Authors:** [INSERT FULL AUTHOR NAMES IN FINAL ORDER]\
**Affiliations:** [INSERT DEPARTMENT/UNIT, INSTITUTION, CITY, POSTAL CODE, COUNTRY; map each author with superscripts if multiple affiliations]\
**Corresponding author:** [FULL NAME, FULL POSTAL ADDRESS, EMAIL]\
**ORCID:** [INSERT ORCID iDs WHERE AVAILABLE]

## Abstract

High predictive accuracy in malaria image classification does not establish which visual information a model exploits or whether the observed behavior is stable across transfer-learning choices. We evaluated color representation, fine-tuning, preprocessing, and information-restricted controls using MobileNetV2 on 27,558 balanced thin-blood-smear cell images. A 2×2×2 factorial design crossed RGB versus grayscale input, frozen versus fine-tuned backbones, and [0,1] versus canonical MobileNetV2 preprocessing across five repeated group-aware partitions (40 CNN runs); five color-statistics classifiers and five Gaussian-blur CNNs provided additional controls. RGB improved mean accuracy most clearly with frozen backbones: the paired RGB-minus-grayscale difference was 0.0253 under [0,1] preprocessing (Holm-adjusted p=0.0032) and 0.0296 under canonical preprocessing (p=0.0015). After fine-tuning, the corresponding gaps narrowed to 0.0044 (p=0.0243) and 0.0056 (p=0.0585). Color-only features retained approximately 81.4% accuracy, while blurred-image CNNs retained approximately 94.97%, showing that substantial label-related signal persisted under strong information restriction. Changing preprocessing modified some absolute metrics but did not overturn the main representation-by-fine-tuning pattern. These results support color-sensitive model behavior and motivate explicit shortcut-sensitive evaluation; external validation is required before clinical generalization.

**Keywords:** Malaria; deep learning; medical image classification; color representation; transfer learning; shortcut learning; model robustness

# 1. Introduction

Microscopic examination of stained blood smears remains an established approach to malaria diagnosis, but it is labor intensive and sensitive to specimen preparation, staining, equipment, and operator expertise. These constraints have motivated computer-assisted microscopy and deep-learning approaches to malaria detection (Poostchi et al., 2018; Rajaraman et al., 2018; Zhao et al., 2020). Transfer learning is attractive in this setting because pretrained convolutional networks can provide useful visual representations when task-specific datasets are limited relative to natural-image corpora (Pan and Yang, 2010; Raghu et al., 2019).

Predictive accuracy alone, however, does not reveal which image cues a classifier uses. Medical-image models can exploit signals that correlate with labels without corresponding to the intended biological evidence, including acquisition conditions, preprocessing, texture, color, background, or other dataset-specific regularities (Geirhos et al., 2020; Zech et al., 2018; Banerjee et al., 2023). Such shortcut-sensitive behavior can preserve internal performance while reducing reliability under distribution shift, making controlled model-behavior analysis important alongside conventional performance reporting.

Color is particularly relevant in stained microscopy. RGB inputs retain chromatic variation introduced by staining and acquisition, whereas grayscale conversion preserves luminance structure while discarding most chromatic information. Prior work in pathology and medical-image analysis shows that color representation, stain variation, and normalization can affect CNN behavior (Clarke and Treanor, 2017; Tellez et al., 2019; Velastegui and Pedersen, 2021). An RGB advantage, however, cannot by itself identify a causal biomarker or shortcut; it instead motivates experiments that isolate representation while controlling architecture and data partitions.

Transfer-learning strategy provides a second source of representation sensitivity. A frozen ImageNet-pretrained backbone must reuse a representation learned from RGB natural images, whereas fine-tuning allows task-specific adaptation. The RGB-grayscale gap may therefore differ systematically between frozen and fine-tuned models. Preprocessing is also relevant because canonical MobileNetV2 preprocessing maps 0-255 inputs to approximately -1 to 1, while unit-range scaling presents a different input distribution to the pretrained network. Testing both regimes provides a direct sensitivity analysis rather than assuming that one preprocessing choice is neutral.

The contribution of this study is a controlled characterization of model behavior rather than a new neural architecture. We jointly evaluate (i) RGB versus grayscale representation, (ii) frozen versus fine-tuned transfer learning, and (iii) [0,1] versus canonical MobileNetV2 preprocessing in a unified factorial design, then add global color-statistics and Gaussian-blur controls to test how much predictive signal survives information restriction. Group-aware partitioning, multiplicity-adjusted paired comparisons, probability-quality analysis, duplicate screening, and artifact-recovery checks provide complementary safeguards against overly performance-centric interpretation.

Accordingly, we ask four questions: **RQ1**, how much does RGB representation improve classification relative to grayscale; **RQ2**, how does fine-tuning modify that difference; **RQ3**, does substantial predictive performance persist when only global color statistics are available or when fine spatial detail is degraded; and **RQ4**, are the principal RGB, grayscale, and fine-tuning patterns stable under [0,1] and canonical MobileNetV2 preprocessing?

# 2. Materials and methods

## 2.1 Study design

The primary experiment crossed image representation (RGB or grayscale), transfer-learning strategy (frozen or fine-tuned backbone), and preprocessing ([0,1] scaling or canonical MobileNetV2 preprocessing). Each of the eight factorial conditions was evaluated on the same five repeated group-aware partitions, yielding 40 main CNN runs. Five global color-statistics classifiers and five Gaussian-blur CNNs added 10 information-restricted controls, for 50 training/evaluation runs in total (Figure 1).

## 2.2 Dataset and logical image set

The analysis used the public malaria cell-image dataset associated with Rajaraman et al. (2018), containing segmented thin-blood-smear cell images labeled as parasitized or uninfected. The analysis set comprised 27,558 unique logical images, balanced between 13,779 parasitized and 13,779 uninfected images. The extracted archive contained two physical copies of each logical basename (55,116 files in total); decoded-pixel comparison showed zero mismatches between paired copies, so duplicated extraction files were not treated as independent observations.

## 2.3 Group inference and repeated group-aware partitioning

A group identifier was inferred from each filename to reduce the risk that closely related images crossed training, validation, and test subsets. Patient-like identifiers matching `C\d+P\d+` were used when present; otherwise, a fallback image/acquisition identifier was used. This produced 420 inferred groups: 148 patient-like groups and 272 fallback groups. Because grouping was inferred from filenames, these identifiers should not be interpreted as universally verified patient identities.

Five repeated partitions used split seeds 42, 123, 7, 2024, and 99. Each partition contained 293 training groups, 64 validation groups, and 63 test groups. Exact group overlap and filepath overlap between training, validation, and test subsets were zero in all five partitions. The partitions are repeated resamples of one dataset rather than independent external cohorts.

## 2.4 Image representations and preprocessing

RGB images were decoded in three channels. Grayscale representation was obtained using the luminance transformation

**Y = 0.2989R + 0.5870G + 0.1140B**, 

The resulting single channel was replicated to three channels to preserve the MobileNetV2 input shape, changing chromatic information while holding architecture and tensor dimensionality constant.

Two preprocessing regimes were evaluated. In the **[0,1] regime**, decoded intensities were scaled to the unit interval. In the **canonical regime**, decoded and augmented intensities remained on the 0-255 scale before `tf.keras.applications.mobilenet_v2.preprocess_input` mapped inputs approximately to -1 to 1. Preprocessing was analyzed as a factorial component rather than as a separate experiment.

Training augmentation comprised horizontal flipping, brightness perturbation, and padding/cropping. Identical augmentation logic was applied within matched representation/training conditions, with representation-specific operations occurring before model input.

## 2.5 CNN architecture and transfer-learning strategies

MobileNetV2 pretrained on ImageNet was used as the common backbone (Sandler et al., 2018; Russakovsky et al., 2015). The classification head comprised global average pooling, Dense(256, ReLU), dropout 0.5, Dense(128, ReLU), dropout 0.3, and a two-unit softmax output. The complete model contained 2,619,074 parameters. Holding architecture constant isolated representation, fine-tuning, and preprocessing effects.

In the **frozen** condition, the pretrained backbone was non-trainable and only the classification head was optimized. In the **fine-tuned** condition, backbone layers from index 100 onward were trainable. In both conditions the backbone was called with `training=False`, so Batch Normalization layers remained in inference mode during fine-tuning; this implementation detail is therefore part of the operational definition of the fine-tuned condition.

Models were optimized with Adam (Kingma and Ba, 2015), using learning rates of 1×10⁻⁴ for frozen models and 1×10⁻⁵ for fine-tuned models. Training used batch size 32, a maximum of 50 epochs, and early stopping with patience 5. The training seed was fixed at 42 for all main runs; split seeds were 42, 123, 7, 2024, and 99. Model selection used validation performance only.

## 2.6 Information-restricted controls

Two controls tested whether substantial predictive information remained after strong information restriction. The first used global color statistics only: per-channel means, standard deviations, and eight-bin histograms, standardized before logistic-regression classification. Because these features contain no explicit spatial arrangement, the control quantifies the predictive value of coarse global color summaries without claiming to represent all possible color information.

The second control applied a Gaussian blur (15×15 kernel, σ=7) before CNN classification. This intervention reduces high-frequency spatial detail while retaining broad intensity, color, and low-frequency morphology. It is therefore treated as a **spatial-detail-degraded baseline**, not a morphology-free condition. Neither control is interpreted as causal proof of a specific shortcut source.

## 2.7 Evaluation metrics and statistical analysis

Performance was summarized by accuracy, precision, recall, F1 score, ROC-AUC, and Brier score. Means and standard deviations were computed across the five repeated partitions. Matched conditions were compared within the same split using two-sided paired t-tests, and multiplicity within each contrast family was controlled using the Holm procedure (Holm, 1979). Because the five partitions are repeated resamples of one dataset, these tests characterize repeated-partition evidence rather than independent-cohort population inference.

Probability-quality analysis additionally included expected calibration error (ECE) and maximum calibration error (MCE) using 15 equal-width probability bins, together with Brier score (Brier, 1950; Guo et al., 2017). These measures describe probability behavior rather than clinical calibration. A stratified 2,000-replicate bootstrap characterized per-run uncertainty without retraining; matched repeated-partition contrasts remained the primary comparative analysis.

## 2.8 Data-integrity and near-duplicate audit

Integrity checks were performed at several levels. Frozen split manifests were checked for exact group and filepath overlap, and decoded pixels were hashed with SHA-256 to identify exact duplicates independently of file metadata. Near-duplicate screening used a 64-bit DCT-based perceptual hash on grayscale 32×32 representations, enumerating all candidate pairs within Hamming distance ≤4. Cross-partition candidates were then screened with an additional grayscale 128×128 similarity score using operational thresholds of 0.98 and 0.995. The audit output labeled this quantity “SSIM,” but its implementation has not been independently verified as canonical local-window SSIM; accordingly, the manuscript reports it only as an operational similarity score. Perceptual-hash similarity alone was never classified as leakage.

## 2.9 Secondary characterization and reproducibility

For the selected RGB fine-tuned model family, validation-based threshold analysis characterized the trade-off between sensitivity and false negatives. Prevalence-based positive-predictive-value simulations were exploratory and hypothetical. Computational profiling was performed on a Google Colab NVIDIA T4 GPU and is reported only as GPU-environment characterization, not as a mobile-device benchmark.

Prediction archives, model files, histories, split manifests, and provenance artifacts were persisted. Recomputed metrics from all 30 [0,1]/control prediction archives matched stored values to numerical precision, and all 20 canonical-preprocessing prediction archives passed the corresponding integrity audit. Twenty canonical Keras models were fully deserialized in a fresh runtime without retraining. These checks establish artifact persistence and computational recoverability; bitwise-equivalent continuation across different GPU runtimes is not claimed.

# 3. Results

## 3.1 Dataset and partition integrity

All five frozen split manifests passed group-overlap and filepath-overlap checks. Each partition contained 293 training groups, 64 validation groups, and 63 test groups; image counts varied because inferred groups contained different numbers of cells. Test sets ranged from 3,221 to 5,580 images. This structure motivates matched split-level comparisons and should not be interpreted as five independent cohorts.

## 3.2 RQ1: RGB versus grayscale representation

RGB representation produced higher mean accuracy than grayscale in every matched preprocessing/training combination. With frozen backbones, the paired RGB-minus-grayscale accuracy difference was 0.0253 under [0,1] preprocessing (95% CI 0.0193-0.0314; Holm-adjusted p=0.0032) and 0.0296 under canonical preprocessing (95% CI 0.0241-0.0352; p=0.0015).

After fine-tuning, the RGB-grayscale gap was much smaller: 0.0044 under [0,1] preprocessing (95% CI 0.0025-0.0063; Holm-adjusted p=0.0243) and 0.0056 under canonical preprocessing (95% CI 0.0018-0.0093; p=0.0585). ROC-AUC showed the same qualitative pattern, with larger RGB advantages in frozen than fine-tuned models.

Thus, RGB provided the clearest advantage when the backbone was frozen. The effect remained positive after fine-tuning but was smaller, and the canonical fine-tuned accuracy contrast did not remain below 0.05 after Holm correction.

## 3.3 RQ2: Fine-tuning and representation sensitivity

Fine-tuning improved mean accuracy in all four representation/preprocessing cells. Accuracy gains were +0.0417 for canonical grayscale (Holm-adjusted p=0.0008), +0.0356 for [0,1] grayscale (p=0.0027), +0.0176 for canonical RGB (p=0.0049), and +0.0147 for [0,1] RGB (p=0.0002). ROC-AUC gains also survived Holm correction in all four cells, whereas recall effects were less uniform; only canonical grayscale remained below 0.05 after adjustment.

Fine-tuning reduced the RGB-grayscale accuracy gap from approximately 2.5-3.0 percentage points in frozen models to approximately 0.4-0.6 percentage points in fine-tuned models under both preprocessing regimes. This reproducible interaction is consistent with reduced representation sensitivity after task-specific adaptation.

## 3.4 RQ3: Information-restricted controls

The global color-statistics logistic-regression control achieved approximately 0.814 mean accuracy and 0.8668 ROC-AUC. Because it contains no spatial arrangement, this result shows that coarse global color summaries alone carry substantial label-related information in this dataset; it does not imply that the CNN relies only on color or that the signal is necessarily spurious.

The Gaussian-blur CNN retained approximately 0.9497 mean accuracy and 0.9862 ROC-AUC after strong degradation of fine spatial detail. Predictive information therefore persisted despite high-frequency information loss, but the surviving signal may reflect broad morphology, color, low-frequency structure, dataset-specific regularities, or a combination of these factors. The control does not causally identify the source.

Together, these controls show why high classification accuracy should be complemented by direct tests of which information remains predictive under controlled restriction.

## 3.5 RQ4: Sensitivity to MobileNetV2 preprocessing

Changing from [0,1] scaling to canonical MobileNetV2 preprocessing did not uniformly improve accuracy. Mean paired changes were -0.0049 for grayscale-frozen, +0.0012 for grayscale-fine-tuned, -0.0006 for RGB-frozen, and +0.0023 for RGB-fine-tuned models; none remained below 0.05 after Holm correction. For ROC-AUC, only the frozen-grayscale contrast remained below 0.05 after correction (mean difference -0.0036; adjusted p=0.0249).

Canonical preprocessing therefore changed some absolute metrics without overturning the main structural pattern: the RGB advantage was largest with a frozen backbone and narrowed after fine-tuning. The result supports stability of the interpretation across the two tested preprocessing regimes, not universal superiority of canonical preprocessing.

## 3.6 Probability-quality characterization

Mean ECE ranged from 0.0146 to 0.0305 across the eight main conditions. Canonical RGB fine-tuning had the lowest mean ECE (0.0146) and Brier score (0.0291), whereas MCE varied more across splits and is sensitive to sparse calibration bins.

These metrics characterize probability quality only. They do not establish clinical calibration, which depends on the target population, prevalence, case mix, operating threshold, and external validation.

## 3.7 Data-integrity and near-duplicate audit

Across 27,558 logical images, the exhaustive audit found zero global exact logical duplicate pairs and zero exact cross-partition decoded-pixel duplicate pairs. Perceptual-hash screening at Hamming radius ≤4 identified 2,962 global candidate pairs; 2,813 unique pairs crossed at least one train/validation/test boundary across the five partitions. None shared the same inferred group, and none reached the operational 0.98 or 0.995 threshold of the secondary similarity score.

The audit therefore found no evidence of exact or high-similarity near-duplicate leakage under the implemented procedure, while remaining unable to exclude slide-, session-, or acquisition-level relationships absent from the available metadata.

## 3.8 Secondary operating-point and computational characterization

For the selected RGB fine-tuned model family, validation-based threshold selection reduced aggregate false negatives from 529 to 470 (11.15%). Because recall worsened in some individual partitions, this is reported as an operating-point trade-off rather than a uniform sensitivity improvement.

All CNN configurations used the same 2,619,074-parameter architecture. Differences in serialized file size reflect training/serialization state rather than parameter count. Latency was measured on a Colab T4 GPU and is not interpreted as mobile or point-of-care performance.

# 4. Discussion

## 4.1 Color contributes predictive information, particularly with a frozen backbone

The central representation result is that RGB improved performance most clearly when the ImageNet-pretrained backbone was frozen. This pattern is plausible because stained blood-smear images contain both morphological and chromatic information, while the pretrained backbone was learned from three-channel natural images. Related pathology studies likewise show that color representation and stain variation can alter CNN behavior (Clarke and Treanor, 2017; Tellez et al., 2019; Velastegui and Pedersen, 2021).

The experiment nevertheless identifies representation sensitivity rather than a causal color biomarker. RGB-versus-grayscale conversion changes chromatic content while retaining luminance structure, so the observed advantage may reflect parasite/stain appearance, broader erythrocyte characteristics, acquisition variation, background statistics, interactions with ImageNet pretraining, or combinations of these factors.

## 4.2 Fine-tuning reduces the RGB–grayscale gap

Fine-tuning improved performance and narrowed the RGB-grayscale gap under both preprocessing regimes. This pattern is consistent with task-specific adaptation compensating for part of the representation mismatch introduced by removing chromatic information, whereas a frozen backbone remains constrained by pretrained features (Pan and Yang, 2010; Raghu et al., 2019).

The result does not imply color independence after fine-tuning. RGB means remained higher in both preprocessing regimes; the [0,1] fine-tuned contrast survived Holm correction, whereas the canonical fine-tuned contrast did not. The supported conclusion is therefore that fine-tuning **reduces**, rather than eliminates, representation sensitivity.

Architectural generalizability remains unresolved because MobileNetV2 was intentionally held constant for factorial control. Replication with other convolutional or transformer backbones would determine whether the same interaction generalizes beyond this architecture.

## 4.3 High performance after information restriction motivates shortcut-sensitive evaluation

The information-restricted controls provide the most direct evidence that strong predictive performance is not confined to the full fine-resolution representation. Global color statistics alone achieved approximately 81.4% accuracy, and the blurred CNN retained approximately 95% accuracy after substantial loss of fine spatial detail. Together, these findings challenge a purely fine-morphology-based interpretation of the main CNN performance.

They do not, however, prove shortcut learning. The controls establish that alternative information channels remain predictive, but they do not determine whether those channels are biologically legitimate, dataset-specific, or fragile under distribution shift (Geirhos et al., 2020). Because blur preserves broad morphology and low-frequency structure, these experiments are best interpreted as **shortcut-sensitive stress tests** rather than causal shortcut identification.

A stronger causal design would vary stain, illumination, background, microscope, and acquisition site independently across external cohorts, then test whether normalization or augmentation reduces dependence on those factors without degrading clinically relevant signal.

## 4.4 The preprocessing sensitivity analysis strengthens the main interpretation

The preprocessing analysis addresses a specific transfer-learning concern: unit-range scaling presents a different input distribution from canonical MobileNetV2 preprocessing. If the RGB-grayscale result were primarily an artifact of that mismatch, changing the input transformation could have reversed the pattern.

Instead, canonical preprocessing altered absolute metrics in a condition-dependent manner while preserving the dominant representation-by-fine-tuning pattern. Frozen RGB models retained a clear advantage over frozen grayscale models, whereas fine-tuned differences remained small. This cross-preprocessing consistency strengthens the qualitative interpretation.

At the same time, canonical preprocessing was not uniformly better. Most canonical-versus-[0,1] differences were small and did not survive Holm correction. The result therefore supports **stability of interpretation**, not superiority of one preprocessing pipeline.

## 4.5 Statistical and calibration analyses support cautious interpretation

Paired split-level inference with Holm correction separates stable effects from small positive differences that are less secure across repeated partitions. The RGB advantage was strongest in frozen models, and fine-tuning improved accuracy and ROC-AUC across all four factorial cells after multiplicity correction; recall effects were more heterogeneous.

Probability-quality metrics add a different perspective because discrimination and probability reliability are not equivalent (Guo et al., 2017). Canonical RGB fine-tuning had the lowest mean ECE and Brier score, but neither metric alone establishes clinical calibration. Calibration requires evaluation in the intended population and operating context.

The threshold analysis reinforces this caution. Validation-selected thresholding reduced aggregate false negatives by 11.15%, but the effect was not uniform across partitions. It therefore characterizes a possible operating-point trade-off rather than a deployment recommendation.

## 4.6 Data integrity and reproducibility are part of model evaluation

Data integrity was treated as part of model evaluation rather than as a post hoc check. Group-aware partitioning prevented exact inferred-group overlap, filepath overlap was zero, and decoded-pixel hashing found no exact cross-partition duplicates. Exhaustive perceptual-hash screening produced visually similar candidates, but no cross-partition candidate shared an inferred group or reached the operational high-similarity threshold used in the secondary audit (Rouzrokh et al., 2022).

These checks reduce important leakage risks without reconstructing metadata that were never available. Slide-, session-, or acquisition-level relationships may remain unobserved if they are not encoded in filenames. The appropriate claim is therefore that no evidence of exact or high-similarity near-duplicate leakage was detected under the implemented audit, not that all dependence was excluded.

Artifact checks further showed that predictions, split manifests, model structures, and canonical model files could be recovered in a fresh runtime without retraining. This does not replace external validation, but it strengthens the traceability of the reported computational results.

## 4.7 Limitations and implications

The study has four main limitations affecting generalizability. First, all repeated partitions derive from one public dataset, so they assess internal stability rather than cross-site performance across laboratories, stains, microscopes, patient populations, or geographic settings. Second, group identifiers were inferred from filenames and are not universally verified patient identities. Third, the training seed was fixed across main runs, so optimization variability across seeds was not quantified. Fourth, MobileNetV2 was the only main backbone, leaving architectural generalizability unresolved.

Additional limitations concern interpretation of the controls and secondary analyses. RGB versus grayscale compares chromatic with luminance representation rather than directly separating color from morphology. Global color features capture only coarse statistics, and Gaussian blur retains broad morphology and low-frequency information. The secondary similarity score used in near-duplicate screening requires implementation-level verification before being described as canonical SSIM. Calibration, prevalence simulation, threshold analysis, and T4-GPU latency are secondary characterizations rather than evidence of clinical calibration, external validity, or mobile deployment performance.

Within these bounds, the study provides a controlled internal characterization of how representation, transfer-learning strategy, preprocessing, and information restriction affect malaria image classification. The most informative next steps are external multi-site validation, cross-device/stain testing, and replication across model families.

# 5. Conclusions

RGB representation improved malaria-cell classification most clearly when the ImageNet-pretrained MobileNetV2 backbone was frozen, while fine-tuning narrowed the RGB-grayscale gap under both preprocessing regimes. Color-only features and strongly blurred images retained substantial predictive performance, showing that label-related signal persists after major restrictions on spatial information and motivating explicit shortcut-sensitive evaluation. Canonical MobileNetV2 preprocessing changed some absolute metrics but did not overturn the principal representation-by-fine-tuning pattern. Group-aware splitting, duplicate screening, multiplicity-adjusted paired comparisons, and artifact-recovery checks strengthen the internal validity and reproducibility of these findings. The conclusions remain specific to the evaluated dataset and architecture; external multi-site validation is required before clinical generalization or deployment claims.

# Declarations

**Funding:** [AUTHOR CONFIRMATION REQUIRED. If there was no research funding, use: “This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.” If funding existed, insert funder name(s) and grant number(s).]

**Competing interests:** [AUTHOR CONFIRMATION REQUIRED. If none, use: “The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.” Otherwise, disclose the relevant interests.]

**Data availability:** The malaria cell-image dataset analyzed in this study is publicly available and is described by Rajaraman et al. (2018). The final submission should provide the exact dataset access URL used in this study. Frozen split manifests, prediction outputs, analysis tables, and reproducibility artifacts supporting the reported results will be deposited at **[REPOSITORY NAME, DOI/URL]** before submission.

**Code availability:** The code used for preprocessing, model training, statistical analysis, calibration analysis, and data-integrity audits will be made available at **[REPOSITORY NAME, DOI/URL]** before submission.

**Author contributions (CRediT):** [COMPLETE AFTER FINAL AUTHOR LIST. Assign only roles actually performed: Conceptualization; Data curation; Formal analysis; Funding acquisition; Investigation; Methodology; Project administration; Resources; Software; Supervision; Validation; Visualization; Writing – original draft; Writing – review & editing.]

**Acknowledgements:** [INSERT IF APPLICABLE; otherwise remove this line.]

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used **ChatGPT (OpenAI)** to support language editing, manuscript organization, and drafting/revision assistance. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

# References

Banerjee, I., Bhattacharjee, K., Burns, J. L., Trivedi, H., Purkayastha, S., Seyyed-Kalantari, L., Patel, B. N., Shiradkar, R., & Gichoya, J. (2023). “Shortcuts” causing bias in radiology artificial intelligence: Causes, evaluation, and mitigation. Journal of the American College of Radiology, 20(9), 842–851. https://doi.org/10.1016/j.jacr.2023.06.025

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

Clarke, E. L., & Treanor, D. (2017). Colour in digital pathology: A review. Histopathology, 70(2), 153–163. https://doi.org/10.1111/his.13079

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence, 2, 665–673. https://doi.org/10.1038/s42256-020-00257-z

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, 70, 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70. https://doi.org/10.2307/4615733

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1412.6980

Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22(10), 1345–1359. https://doi.org/10.1109/TKDE.2009.191

Poostchi, M., Silamut, K., Maude, R. J., Jaeger, S., & Thoma, G. (2018). Image analysis and machine learning for detecting malaria. Translational Research, 194, 36–55. https://doi.org/10.1016/j.trsl.2017.12.004

Raghu, M., Zhang, C., Kleinberg, J., & Bengio, S. (2019). Transfusion: Understanding transfer learning for medical imaging. Advances in Neural Information Processing Systems, 32, 3347–3357.

Rajaraman, S., Antani, S. K., Poostchi, M., Silamut, K., Hossain, M. A., Maude, R. J., Jaeger, S., & Thoma, G. R. (2018). Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images. PeerJ, 6, e4568. https://doi.org/10.7717/peerj.4568

Rouzrokh, P., Khosravi, B., Faghani, S., Moassefi, M., Vera Garcia, D. V., Singh, Y., Zhang, K., Conte, G. M., & Erickson, B. J. (2022). Mitigating bias in radiology machine learning: 1. Data handling. Radiology: Artificial Intelligence, 4(5), e210290. https://doi.org/10.1148/ryai.210290

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. International Journal of Computer Vision, 115(3), 211–252. https://doi.org/10.1007/s11263-015-0816-y

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 4510–4520. https://doi.org/10.1109/CVPR.2018.00474

Tellez, D., Litjens, G., Bándi, P., Bulten, W., Bokhorst, J.-M., Ciompi, F., & van der Laak, J. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis, 58, 101544. https://doi.org/10.1016/j.media.2019.101544

Velastegui, R., & Pedersen, M. (2021). The impact of using different color spaces in histological image classification using convolutional neural networks. In 2021 9th European Workshop on Visual Information Processing (EUVIP) (pp. 1–6). IEEE. https://doi.org/10.1109/EUVIP50544.2021.9484035

Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine, 15(11), e1002683. https://doi.org/10.1371/journal.pmed.1002683

Zhao, O. S., Kolluri, N., Anand, A., Chu, N., Bhavaraju, R., Ojha, A., Tiku, S., Nguyen, D., Chen, R., Morales, A., Valliappan, D., Patel, J. P., & Nguyen, K. (2020). Convolutional neural networks to automate the screening of malaria in low-resource countries. PeerJ, 8, e9674. https://doi.org/10.7717/peerj.9674
