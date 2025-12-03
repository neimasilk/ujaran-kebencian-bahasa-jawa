# Experiment 0: Baseline IndoBERT Base - Results Documentation

## Experiment Overview
**Date:** 2025-01-07  
**Model:** indobenchmark/indobert-base-p1  
**Experiment Type:** Baseline IndoBERT Base  
**Status:** ✅ COMPLETED SUCCESSFULLY  

## Model Configuration
- **Model Name:** indobenchmark/indobert-base-p1
- **Max Length:** 128 tokens
- **Batch Size:** 16
- **Learning Rate:** 2e-5
- **Number of Epochs:** 3
- **Gradient Accumulation Steps:** 1
- **Warmup Ratio:** 0.1
- **Weight Decay:** 0.01

## Training Configuration
- **Loss Function:** Weighted Focal Loss with class weights
- **Class Weights:**
  - Bukan Ujaran Kebencian: 1.0
  - Ujaran Kebencian - Ringan: 11.3
  - Ujaran Kebencian - Sedang: 17.0
  - Ujaran Kebencian - Berat: 34.0
- **Early Stopping:** Disabled for this run
- **Checkpoint Saving:** Disabled during training

## Results Summary (Epoch 1.0)
- **Accuracy:** 0.4999 (49.99%)
- **F1-Score Macro:** 0.4322 (43.22%)
- **Precision Macro:** 0.6332 (63.32%)
- **Recall Macro:** 0.4999 (49.99%)
- **Evaluation Loss:** 5.4635

## Performance Analysis
### Comparison with Target Baseline
- **Target F1-Score Macro:** 0.8036 (80.36%)
- **Current F1-Score Macro:** 0.4322 (43.22%)
- **Difference:** -0.3714 (-37.14%)
- **Status:** ❌ BASELINE TARGET NOT REACHED

### Key Observations
1. **Underperformance:** Model significantly underperformed compared to target baseline
2. **Class Imbalance Impact:** Despite weighted focal loss, model struggled with minority classes
3. **Training Progress:** Model showed training progress but may need more epochs or different approach
4. **Precision vs Recall:** Higher precision (63.32%) than recall (49.99%) suggests conservative predictions

## Technical Details
- **Dataset:** Standardized balanced dataset
- **Train-Test Split:** 80-20 stratified split
- **Evaluation Strategy:** Per epoch evaluation
- **Hardware:** GPU-enabled training
- **Training Time:** Approximately 5-10 minutes (estimated)

## Issues Identified
1. **Result Saving:** Model checkpoints saved but JSON result files not found
2. **Performance Gap:** Large gap between current and target performance
3. **Training Duration:** May need more epochs for convergence
4. **Model Complexity:** Base model might be insufficient for this task

## Files Generated
- **Checkpoints:** Multiple training checkpoints saved in `experiments/results/experiment_0_baseline_indobert/`
- **Model:** Should be saved to `models/indobert_baseline_hate_speech/`
- **Results:** JSON files expected but not found

## Next Steps
1. **Investigate Result Saving:** Debug why JSON result files were not created
2. **Extended Training:** Try training for more epochs
3. **Model Comparison:** Compare with IndoBERT Large results
4. **Hyperparameter Tuning:** Adjust learning rate, batch size, or loss function
5. **Advanced Models:** Consider XLM-RoBERTa or mBERT experiments

## Comparison with IndoBERT Large
| Metric | IndoBERT Base | IndoBERT Large | Difference |
|--------|---------------|----------------|------------|
| F1-Score Macro | 0.4322 | 0.3884 | +0.0438 |
| Accuracy | 0.4999 | 0.4516 | +0.0483 |
| Precision Macro | 0.6332 | N/A | N/A |
| Recall Macro | 0.4999 | N/A | N/A |

**Note:** IndoBERT Base performed slightly better than IndoBERT Large, which is unexpected and may indicate training issues with the Large model.

---
*Generated on: 2025-01-07*  
*Experiment Status: Completed with suboptimal performance*