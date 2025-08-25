# Beyond 85%: Improving Javanese Hate Speech Detection with Targeted Training and Balanced Evaluation

## Abstract
Hate speech in local languages such as Javanese demands classifiers that are both accurate and fair across classes. We present a four-class Javanese hate speech detector (non-hate, light, medium, heavy) that surpasses the 85% target by combining a targeted training strategy and a balanced evaluation protocol. Starting from an IndoBERT Large v1.2 baseline (Accuracy 65.80%, F1-Macro 60.75%), our improved training pipeline integrates focal loss, class weighting, cosine learning-rate scheduling with warmup, early stopping on F1-Macro, and mixed-precision (FP16). On a balanced test set of 4,993 samples (~20% of the balanced dataset), the best model achieves 86.98% Accuracy and 86.88% F1-Macro. Threshold tuning on the baseline improves its performance to 80.37% Accuracy and 80.36% F1-Macro but remains below the improved model on the full test set. Homogeneous stacking ensembles provide marginal gains, suggesting the need for architectural diversity. Results highlight that training strategy and balanced evaluation matter more than changing architectures alone.

Keywords: Javanese hate speech, F1-Macro, Transformer, focal loss, balanced evaluation, threshold tuning

## 1. Introduction
Detecting hate speech in Javanese is challenging due to class imbalance, orthographic variety, dialectal differences, and code-mixing. We target high performance on a four-class task, prioritizing F1-Macro to ensure class-level fairness. Our goals are: (i) exceed 85% Accuracy and F1-Macro, (ii) document a reproducible pipeline, and (iii) chart a realistic path toward 90%+.

Contributions:
- A targeted training strategy that achieves 86.98% Accuracy and 86.88% F1-Macro on a balanced test set.
- Evidence that optimization of training and evaluation protocol yields larger gains than architecture swapping.
- A documented evaluation pipeline and a roadmap to 90%+ via augmentation, hyperparameter optimization, and multi-architecture ensembling.

## 2. Data and Preprocessing
We use a balanced evaluation setup with a test split of 4,993 samples (~20% of the balanced dataset). The original dataset contains ~24,964 samples; augmentation increases it to ~32,452 samples (+7,488), with near-uniform class distribution (≈8,113 per class). Preprocessing includes light normalization, model tokenization, and class rebalancing during training via class weights.

## 3. Methodology
### 3.1 Model Families
Preliminary explorations considered IndoBERT, mBERT, and XLM-RoBERTa variants. Final gains primarily stem from the training strategy rather than switching backbones.

### 3.2 Targeted Training Strategy
- Loss and imbalance handling: focal loss (γ=2, α=1) + class weighting.
- Optimization: AdamW, cosine scheduler, warmup ratio 0.1, weight decay 0.01.
- Efficiency and stability: mixed precision (FP16), gradient accumulation; batch size 16 (train) / 32 (eval).
- Early stopping: patience=3 based on F1-Macro on validation.

### 3.3 Threshold Tuning
We optimize per-class decision thresholds on the baseline model using a validation subset. Optimal thresholds found: non-hate ≈ 0.7128; light ≈ 0.2332; medium ≈ 0.2023; heavy ≈ 0.3395. This improves the baseline but does not surpass the improved model on the full test set.

### 3.4 Ensembling
Homogeneous stacking (similar models) yields marginal test gains. We hypothesize that diversity (IndoBERT + XLM-R + ELECTRA), richer meta-features (logits/confidence), and calibration will be needed for meaningful improvements.

## 4. Evaluation Protocol
Primary metric: F1-Macro. Secondary: Accuracy, F1-Weighted, macro Precision/Recall, and per-class scores. Evaluation uses a balanced test set (4,993 samples) to mitigate class bias. We also report confusion matrix and classification report to analyze error patterns.

## 5. Results
### 5.1 Final Performance (Improved Training)
- Accuracy: 86.98% (target: 85% achieved)
- F1-Macro: 86.88%
- F1-Weighted: 86.88%
- Per-class (example standardized report):
  - non-hate: F1=0.811, Precision=0.866, Recall=0.762
  - light: F1=0.875, Precision=0.868, Recall=0.883
  - medium: F1=0.864, Precision=0.834, Recall=0.896
  - heavy: F1=0.925, Precision=0.913, Recall=0.938

### 5.2 Baseline and Threshold Tuning
- Baseline (threshold 0.5): Accuracy 73.75%, F1-Macro 73.72% (on subset protocol).
- Tuned thresholds: Accuracy 80.37%, F1-Macro 80.36% (+6.6% over baseline).

### 5.3 Compared to Initial Baseline
- IndoBERT Large v1.2 (initial): Accuracy 65.80%, F1-Macro 60.75%.
- Total improvement: +21.18% Accuracy, +26.13% F1-Macro.

### 5.4 Ensembling
Homogeneous ensembling shows marginal improvements on validation and does not outperform the improved model on the full test set.

## 6. Analysis and Discussion
Key drivers of success are the combination of focal loss, class weighting, and learning-rate scheduling, supported by FP16 efficiency. Threshold tuning helps under-optimized baselines but cannot compensate for representation quality across the full test distribution. The heavy class attains the highest F1, whereas non-hate exhibits lower recall, likely due to ambiguous and context-dependent usage, indicating room for better contextual modeling and label quality control.

## 7. Error Analysis
Common confusions occur between light and medium severity; non-hate with coarse words in non-aggressive contexts; and dialect/orthography variations. Future work should include a contextual lexicon for Javanese profanity, orthography normalization, and pragmatic cues.

## 8. Limitations
- Minor inconsistencies across per-class reports in intermediate documents; we recommend one canonical evaluation JSON as ground truth.
- No explicit cross-domain evaluation yet (domain/platform shift).
- No ablation and statistical significance tests reported; these are needed to isolate component contributions and quantify reliability.

## 9. Future Work
Toward 90%+: quality-aware augmentation, multi-objective hyperparameter optimization (Accuracy + F1-Macro), diverse multi-architecture ensembles with richer meta-features, calibration (ECE/Brier), robustness to noise/adversarial typos, and fairness analyses across dialects/orthographies.

## 10. Ethics and Bias
Hate speech detection systems risk biased moderation. We advocate auditing label quality, periodic fairness checks across subgroups, transparent moderation policies, and avenues for appeal.

## 11. Reproducibility
We release code, requirements, and evaluation scripts. For strict reproducibility, fix random seeds, record package versions and GPU specs, publish the exact data split indices, and store a single official evaluation file (e.g., results/improved_model_evaluation.json).

## 12. Conclusion
A targeted training strategy with balanced evaluation pushes Javanese hate speech detection to 86.98% Accuracy and 86.88% F1-Macro, surpassing the 85% target. The approach provides a practical, reproducible path, and a clear roadmap toward 90%+ via HPO and diverse ensembling.

## Acknowledgments
We thank annotators and contributors who supported data preparation and system development.

## References
[1] Lin et al., Focal Loss for Dense Object Detection.  
[2] Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.  
[3] Conneau et al., Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).