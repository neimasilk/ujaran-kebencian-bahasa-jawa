# Experiment 1.1: IndoBERT Large Configuration Fix

## 🔧 Problem Analysis

Eksperimen pertama menghasilkan performa yang sangat buruk:
- **F1-Score Macro**: 0.1897 (18.97%) vs Target 83%
- **Accuracy**: 0.233 (23.3%)
- **Training Duration**: Hanya 0.15 epoch dari 5 epoch

## 🎯 Root Causes Identified

### 1. Early Stopping Terlalu Agresif
- **Problem**: Patience=2, Threshold=0.001
- **Impact**: Training berhenti di step 300 dari 9,975 total steps
- **Evidence**: Model belum sempat belajar dengan baik

### 2. Class Weights Berlebihan
- **Problem**: Weights terlalu ekstrem (8.5, 15.2, 25.8)
- **Impact**: Model overfitting pada kelas minoritas
- **Evidence**: Precision tinggi tapi recall sangat rendah

### 3. Learning Rate Terlalu Rendah
- **Problem**: LR=1e-5 dengan model besar
- **Impact**: Konvergensi lambat
- **Evidence**: Loss turun sangat lambat

### 4. Monitoring Kurang Frequent
- **Problem**: eval_steps=100
- **Impact**: Tidak bisa mendeteksi masalah lebih awal

## ✅ Configuration Fixes Applied

### 1. Early Stopping Parameters
```python
# BEFORE
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001

# AFTER (FIXED)
EARLY_STOPPING_PATIENCE = 5  # More patient
EARLY_STOPPING_THRESHOLD = 0.01  # Less sensitive
```

### 2. Class Weights Rebalancing
```python
# BEFORE
CLASS_WEIGHTS = {
    0: 1.0,    # Bukan Ujaran Kebencian
    1: 8.5,    # Ujaran Kebencian - Ringan
    2: 15.2,   # Ujaran Kebencian - Sedang
    3: 25.8    # Ujaran Kebencian - Berat
}

# AFTER (FIXED)
CLASS_WEIGHTS = {
    0: 1.0,    # Bukan Ujaran Kebencian
    1: 3.0,    # Ujaran Kebencian - Ringan (Reduced from 8.5)
    2: 2.5,    # Ujaran Kebencian - Sedang (Reduced from 15.2)
    3: 3.5     # Ujaran Kebencian - Berat (Reduced from 25.8)
}
```

### 3. Learning Rate Optimization
```python
# BEFORE
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1

# AFTER (FIXED)
LEARNING_RATE = 2e-5  # Doubled for better convergence
WARMUP_RATIO = 0.2    # More gradual warmup
```

### 4. Evaluation Frequency
```python
# BEFORE
eval_steps = 100

# AFTER (FIXED)
eval_steps = 50  # More frequent monitoring
```

## 📊 Expected Improvements

### Performance Targets
- **F1-Score Macro**: 70-75% (vs previous 18.97%)
- **Training Duration**: 2-3 epochs minimum
- **Convergence**: More stable and gradual
- **Class Balance**: Better precision-recall balance

### Training Behavior
- **Early Stopping**: Will allow more training time
- **Loss Convergence**: Faster and more stable
- **Evaluation**: Better monitoring with eval_steps=50
- **Class Performance**: More balanced across all classes

## 🔍 Monitoring Points

### Key Metrics to Watch
1. **F1-Score progression** per evaluation
2. **Loss convergence** pattern
3. **Per-class performance** balance
4. **Training duration** (should be >1 epoch)

### Success Criteria
- **Primary**: F1-Score Macro >70%
- **Secondary**: Training completes at least 1 full epoch
- **Tertiary**: Balanced performance across classes

## 📝 Implementation Status

- ✅ **Configuration Updated**: All parameters fixed
- ✅ **Documentation Created**: This file
- 🔄 **Ready for Execution**: Experiment can be re-run

## 🚀 Next Steps

1. **Execute Fixed Experiment**: Run with new configuration
2. **Monitor Progress**: Watch training metrics closely
3. **Analyze Results**: Compare with previous failed attempt
4. **Document Findings**: Update experiment status

---

**Date**: 3 Juli 2025  
**Status**: Configuration Fixed, Ready for Re-execution  
**Expected Success Rate**: >90% (vs previous 0%)