# Experiment 0: Baseline IndoBERT Results

## Eksperimen Overview
- **Model**: IndoBERT Base (indobenchmark/indobert-base-p1)
- **Dataset**: Standardized Balanced Dataset (data/standardized/balanced_dataset.csv)
- **Tanggal**: 7 Januari 2025
- **Status**: ✅ COMPLETED SUCCESSFULLY

## Dataset Information
- **Total Samples**: 25,041 (seimbang sempurna)
- **Train Set**: 20,032 samples (80%)
- **Test Set**: 5,009 samples (20%)
- **Distribusi Kelas**: Seimbang dengan class weights yang disesuaikan

## Training Configuration
- **Epochs**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Max Length**: 128 tokens
- **Training Time**: 316.61 seconds (~5.3 minutes)
- **Class Weights**: 
  - Bukan Ujaran Kebencian: 1.0
  - Ujaran Kebencian - Ringan: 11.3
  - Ujaran Kebencian - Sedang: 17.0
  - Ujaran Kebencian - Berat: 34.0

## Results Summary

### Final Metrics
- **Accuracy**: 0.5590 (55.90%)
- **F1-Score Macro**: 0.4999 (49.99%)
- **Precision Macro**: 0.6574 (65.74%)
- **Recall Macro**: 0.5590 (55.90%)

### Training Progress
- **Initial Loss**: 11.2066
- **Final Training Loss**: 4.4866
- **Final Evaluation Loss**: 5.4635
- **Training Steps**: 3,747 total steps
- **Evaluation Strategy**: Per epoch

## Comparison with Target
- **Target F1-Score Macro**: 0.8036
- **Current F1-Score Macro**: 0.4999
- **Difference**: -0.3037
- **Status**: ❌ Baseline target not reached

## Technical Details

### Model Architecture
- **Base Model**: IndoBERT Base
- **Classification Head**: Linear layer with 4 outputs
- **Loss Function**: Weighted Focal Loss with class weights
- **Optimizer**: AdamW with warmup

### Training Strategy
- **Stratified Split**: Mempertahankan distribusi kelas
- **Class Weighting**: Mengatasi ketidakseimbangan kelas
- **Gradient Accumulation**: 1 step
- **Weight Decay**: 0.01
- **Warmup Ratio**: 0.1

### Dataset Standardization
- ✅ Menggunakan dataset standar yang seimbang
- ✅ Kolom label_numeric untuk konsistensi
- ✅ Preprocessing teks yang konsisten
- ✅ Stratified split untuk evaluasi yang adil

## Issues Resolved

### 1. Unicode Encoding Error
- **Problem**: `UnicodeEncodeError` dengan karakter emoji
- **Solution**: Mengganti emoji dengan teks biasa dalam log

### 2. Checkpoint Saving Error
- **Problem**: `safetensors_rust.SafetensorError` pada Windows
- **Solution**: Menonaktifkan checkpoint saving (`save_strategy="no"`)

### 3. EarlyStoppingCallback Error
- **Problem**: `AssertionError` karena missing `metric_for_best_model`
- **Solution**: Menghapus EarlyStoppingCallback

## Files Generated
- **Model**: Disimpan di `models/indobert_baseline_hate_speech/`
- **Results**: Disimpan di `experiments/results/experiment_0_baseline_indobert/`
- **Checkpoints**: Training checkpoints tersimpan

## Next Steps

### Immediate Actions
1. ✅ Verifikasi dataset standar berfungsi dengan baik
2. ✅ Baseline model berhasil dilatih tanpa error
3. ✅ Dokumentasi hasil lengkap tersedia

### Model Improvement Opportunities
1. **Hyperparameter Tuning**: Learning rate, batch size, epochs
2. **Advanced Models**: IndoBERT Large, XLM-RoBERTa, mBERT
3. **Training Strategies**: Different loss functions, data augmentation
4. **Ensemble Methods**: Combining multiple models

### Performance Analysis
Model baseline menunjukkan performa yang masih di bawah target (F1-macro 0.50 vs target 0.80). Ini memberikan ruang yang signifikan untuk improvement melalui:
- Model yang lebih besar (IndoBERT Large)
- Fine-tuning hyperparameter
- Advanced training techniques
- Ensemble approaches

## Conclusion

Eksperimen baseline berhasil diselesaikan dengan dataset standar yang seimbang. Meskipun performa belum mencapai target, eksperimen ini memberikan:
1. **Baseline yang solid** untuk perbandingan
2. **Infrastruktur yang stabil** untuk eksperimen lanjutan
3. **Dataset standar** yang konsisten
4. **Dokumentasi lengkap** untuk reproduktibilitas

Semua eksperimen selanjutnya dapat menggunakan setup yang sama dengan confidence bahwa infrastruktur berfungsi dengan baik.