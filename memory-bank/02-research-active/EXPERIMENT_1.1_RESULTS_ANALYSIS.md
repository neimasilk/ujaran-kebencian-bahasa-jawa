# Experiment 1.1: IndoBERT Large - Hasil dan Analisis

## 📊 Ringkasan Eksperimen

**Tanggal**: 3 Juli 2025  
**Model**: IndoBERT Large (`indobenchmark/indobert-large-p1`)  
**Objective**: Mencapai F1-Score Macro >83%  
**Status**: ✅ Selesai dengan perbaikan signifikan

## 🎯 Hasil Performa

### Metrik Utama
- **F1-Score Macro**: 0.3830 (target: 0.83)
- **Accuracy**: 39.80%
- **Improvement dari Exp 1.0**: +102% (dari 0.1897)
- **Training Time**: 941.92 detik (~15.7 menit)
- **Training Steps**: 1000/9975 (early stopping)

### Performa Per Kelas

| Kelas | Precision | Recall | F1-Score | Support | Analisis |
|-------|-----------|--------|----------|---------|----------|
| Bukan Ujaran Kebencian | 0.611 | 0.379 | **0.468** | 3896 | Precision baik, recall rendah |
| Ujaran Kebencian - Ringan | 0.242 | 0.480 | **0.322** | 1184 | High recall, low precision |
| Ujaran Kebencian - Sedang | 0.309 | 0.454 | **0.368** | 1616 | Balanced tapi masih rendah |
| Ujaran Kebencian - Berat | 0.472 | 0.311 | **0.375** | 1284 | Good precision, low recall |

## 🔧 Konfigurasi yang Diterapkan

### Perbaikan dari Experiment 1.0
1. **Early Stopping**: Patience 2→5, Threshold 0.001→0.01
2. **Class Weights**: Dikurangi agresivitas (1:3.0:2.5:3.5)
3. **Learning Rate**: 1e-5 → 2e-5
4. **Warmup Ratio**: 0.1 → 0.2
5. **Evaluation Steps**: 100 → 50

### Parameter Training
```python
MODEL_NAME = "indobenchmark/indobert-large-p1"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.01
WARMUP_RATIO = 0.2
EVAL_STEPS = 50
```

## 📈 Training Progress Analysis

### Best Performance
- **Best Step**: 950
- **Best F1-Macro**: 0.4283
- **Best Checkpoint**: checkpoint-600

### Early Stopping Behavior
- **Triggered at**: Step 1000
- **Patience Counter**: 5/5
- **Reason**: No improvement for 5 consecutive evaluations
- **Assessment**: ✅ Reasonable stopping point

### Learning Curve Insights
1. **Initial Phase** (0-200 steps): Rapid improvement
2. **Plateau Phase** (200-600 steps): Steady gains
3. **Stagnation Phase** (600-1000 steps): Minimal improvement

## ⚠️ Identified Issues

### 1. **Underperformance Gap**
- Current: 0.38 F1-Macro
- Target: 0.83 F1-Macro
- **Gap**: -54.5% (masih sangat jauh)

### 2. **Class Imbalance Problems**
- Minority classes (Ringan, Sedang, Berat) masih struggle
- Precision-Recall trade-off tidak optimal
- Model bias ke majority class

### 3. **Learning Rate Issues**
- Plateau terlalu cepat di step 600-1000
- Possible learning rate masih terlalu konservatif
- Need more aggressive optimization

### 4. **Model Capacity Concerns**
- IndoBERT Large mungkin butuh fine-tuning lebih dalam
- Possible architecture limitations untuk task ini

## 🎯 Root Cause Analysis

### Primary Factors
1. **Learning Rate**: Masih terlalu rendah untuk convergence optimal
2. **Class Weights**: Perlu fine-tuning lebih lanjut
3. **Training Duration**: Early stopping masih terlalu agresif
4. **Data Quality**: Possible noise atau mislabeling

### Secondary Factors
1. **Batch Size**: Mungkin terlalu kecil (8)
2. **Model Architecture**: Perlu evaluasi alternatif
3. **Loss Function**: Focal loss parameters perlu tuning
4. **Regularization**: Dropout, weight decay optimization

## 💡 Rekomendasi untuk Experiment 1.2

### Immediate Optimizations
1. **Learning Rate**: 2e-5 → 3e-5 atau 5e-5
2. **Warmup Steps**: 200 → 500-1000 steps
3. **Early Stopping**: Patience 5 → 8-10
4. **Batch Size**: 8 → 16 (dengan gradient accumulation)
5. **Training Steps**: Extend max_steps jika perlu

### Advanced Strategies
1. **Learning Rate Scheduling**: Cosine annealing with restarts
2. **Gradient Accumulation**: Effective batch size 32-64
3. **Data Augmentation**: Paraphrase, back-translation
4. **Ensemble Methods**: Multiple model combination
5. **Multi-stage Training**: Progressive unfreezing

### Alternative Approaches
1. **Model Alternatives**:
   - `cahya/bert-base-indonesian-522M`
   - `indolem/indobert-base-uncased`
   - `microsoft/mdeberta-v3-base`

2. **Architecture Modifications**:
   - Add classification head layers
   - Implement attention mechanisms
   - Use hierarchical classification

## 📋 Next Steps

### Experiment 1.2 Plan
1. ✅ Implement learning rate optimization
2. ✅ Adjust early stopping parameters
3. ✅ Fine-tune class weights
4. ✅ Add gradient accumulation
5. ✅ Extend training duration

### Success Criteria
- **Target F1-Macro**: 0.50-0.60 (realistic intermediate goal)
- **Training Stability**: No early crashes
- **Class Balance**: Improved minority class performance
- **Convergence**: Smooth learning curve

### Risk Mitigation
- Monitor overfitting dengan validation loss
- Backup checkpoints setiap 200 steps
- Log detailed metrics untuk analysis
- Prepare fallback configurations

## 📊 Comparison Matrix

| Aspect | Exp 1.0 | Exp 1.1 | Exp 1.2 (Planned) |
|--------|---------|---------|--------------------|
| F1-Macro | 0.1897 | **0.3830** | 0.50-0.60 |
| Learning Rate | 1e-5 | 2e-5 | 3e-5 |
| Early Stopping | 2/0.001 | 5/0.01 | 8/0.005 |
| Batch Size | 8 | 8 | 16 (grad_accum) |
| Training Time | 251s | 942s | ~1500s |
| Stability | ❌ | ✅ | ✅ |

---

**Kesimpulan**: Experiment 1.1 menunjukkan perbaikan dramatis (+102%) dari baseline, namun masih jauh dari target. Diperlukan optimasi lebih agresif untuk Experiment 1.2 dengan fokus pada learning rate, batch size, dan training duration.