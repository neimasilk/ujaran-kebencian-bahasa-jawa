# Experiment 1.2: XLM-RoBERTa Analysis Report
**Javanese Hate Speech Detection - XLM-RoBERTa Baseline**

## Experiment Overview
**Date:** 2025-01-07  
**Model:** xlm-roberta-base  
**Status:** ❌ FAILED/INCOMPLETE  
**Execution Time:** <1 minute (abnormally fast)  
**Script:** `experiments/experiment_1_2_xlm_roberta.py`  

## Configuration Details

### Model Configuration
- **Model Name:** xlm-roberta-base
- **Max Sequence Length:** 256 tokens
- **Number of Labels:** 4 classes
- **Architecture:** XLM-RoBERTa with classification head

### Training Configuration
- **Batch Size:** 8
- **Number of Epochs:** 5
- **Learning Rate:** 1e-5
- **Optimizer:** AdamW
- **Loss Function:** WeightedFocalLoss
- **Evaluation Strategy:** epoch
- **Save Strategy:** epoch

### Data Configuration
- **Dataset:** balanced_dataset.csv
- **Total Samples:** 24,964
- **Train Set:** 19,971 samples
- **Test Set:** 4,993 samples
- **Class Distribution:** Balanced (25% each class)

## Execution Log Analysis

### Successful Initialization
✅ **Device Detection:** CUDA successfully detected  
✅ **Data Loading:** Dataset loaded correctly (24,964 samples)  
✅ **Class Distribution:** Confirmed balanced distribution  
✅ **Train-Test Split:** Stratified split completed  
✅ **Tokenizer Loading:** XLM-RoBERTa tokenizer loaded  
✅ **Model Loading:** XLM-RoBERTa base model loaded  
✅ **Classifier Initialization:** New classifier weights initialized  

### Training Initiation
✅ **Training Started:** Training process began  
✅ **Progress Tracking:** Reached 1% (100/12,485 steps)  
✅ **Loss Monitoring:** Initial loss: 0.8037  
✅ **Learning Rate:** 1.96e-06 (as expected)  

### Premature Termination
❌ **Unexpected Exit:** Process terminated with exit code 0  
❌ **Incomplete Training:** Only 1% of training completed  
❌ **No Results Generated:** No evaluation metrics or saved models  
❌ **Missing Output Files:** No result JSON files created  

## Issue Analysis

### Possible Causes

#### 1. Memory Issues
- **GPU Memory:** Potential CUDA out of memory error
- **System Memory:** Insufficient RAM for large model
- **Batch Size:** May be too large for available resources

#### 2. Configuration Errors
- **Training Arguments:** Incorrect TrainingArguments configuration
- **Model Setup:** Issues with model initialization
- **Data Pipeline:** Problems with dataset processing

#### 3. Dependency Issues
- **Library Conflicts:** Incompatible package versions
- **CUDA Compatibility:** GPU driver or CUDA version issues
- **Transformers Version:** Outdated or incompatible transformers library

#### 4. Script Logic Errors
- **Exception Handling:** Unhandled exceptions causing silent failures
- **Early Exit Conditions:** Incorrect conditional statements
- **Resource Management:** Improper cleanup causing premature termination

## Comparison with Other Experiments

### vs. IndoBERT Base (Successful)
- **Model Size:** XLM-RoBERTa (~270M) vs IndoBERT Base (~110M)
- **Memory Usage:** Higher memory requirements
- **Batch Size:** Same (8) but different model complexity
- **Max Length:** 256 vs 128 tokens

### vs. IndoBERT Large (Successful with Issues)
- **Model Size:** XLM-RoBERTa (~270M) vs IndoBERT Large (~340M)
- **Memory Management:** IndoBERT Large required batch size reduction
- **Training Duration:** IndoBERT Large completed in ~20 minutes
- **Resource Usage:** Both models resource-intensive

## Diagnostic Recommendations

### Immediate Actions
1. **Check GPU Memory Usage:** Monitor NVIDIA-SMI during execution
2. **Reduce Batch Size:** Try batch size 4 or 2
3. **Reduce Max Length:** Use 128 tokens instead of 256
4. **Add Debug Logging:** Implement more verbose error reporting

### Configuration Adjustments
```python
# Recommended configuration changes
class Config:
    BATCH_SIZE = 4  # Reduced from 8
    MAX_LENGTH = 128  # Reduced from 256
    GRADIENT_ACCUMULATION_STEPS = 4  # Maintain effective batch size
    FP16 = True  # Enable mixed precision
    DATALOADER_NUM_WORKERS = 0  # Reduce multiprocessing overhead
```

### Error Handling Improvements
```python
# Add comprehensive error handling
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    logger.error("CUDA out of memory. Reduce batch size.")
except Exception as e:
    logger.error(f"Training failed: {str(e)}")
    raise
```

## Next Steps

### Short-term Actions
1. **Debug Current Script:** Add error handling and logging
2. **Resource Optimization:** Reduce memory requirements
3. **Retry Experiment:** Run with adjusted configuration
4. **Monitor Resources:** Track GPU/CPU/Memory usage

### Alternative Approaches
1. **Smaller Model:** Try distilbert-base-multilingual-cased
2. **Different Framework:** Consider PyTorch Lightning for better error handling
3. **Incremental Training:** Start with smaller dataset subset
4. **Cloud Resources:** Use higher-memory GPU instances

## Impact on Overall Study

### Missing Baseline
- **Multilingual Comparison:** Cannot compare XLM-RoBERTa performance
- **Cross-lingual Capabilities:** Missing evaluation of multilingual model
- **Benchmark Completeness:** Incomplete model comparison matrix

### Research Implications
- **Model Selection:** Cannot determine if multilingual models are beneficial
- **Resource Requirements:** Unknown computational costs for XLM-RoBERTa
- **Performance Ceiling:** Missing potential high-performing baseline

## Conclusion

The XLM-RoBERTa experiment failed due to premature termination, likely caused by resource constraints or configuration issues. The model showed successful initialization and began training, indicating that the setup was partially correct. Immediate debugging and resource optimization are required to complete this important baseline comparison.

**Priority:** HIGH - This experiment is crucial for comprehensive model comparison and should be resolved before proceeding with advanced techniques.

---
*Status: Failed - Requires debugging and retry*  
*Next Action: Resource optimization and error handling improvements*  
*Expected Resolution Time: 1-2 hours with proper debugging*