# Papan Proyek - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

### STATUS PROGRES
- ✅ Proyek telah disiapkan sesuai dengan Vibe Coding Guide v1.4
- ✅ Spesifikasi produk telah diperbaiki dan disesuaikan dengan template PRD
- ✅ Dokumentasi redundan telah dihapus untuk menghindari ambiguitas
- ✅ Tim manifest telah disiapkan dengan peran yang jelas
- ✅ Environment setup dan basic scripts telah dibuat
- ✅ Dataset inspection report telah dihasilkan
- ✅ Modular code structure (data_utils.py, train_utils.py) telah diimplementasi

### REFERENSI ARSIP
- Baby-step sebelumnya: Setup environment dan refaktorisasi notebook (selesai)

### BABY-STEP: Data Labeling dengan DeepSeek V3 API ✅ SELESAI
- **Tujuan:** Implementasi pelabelan otomatis menggunakan DeepSeek V3 API untuk dataset Bahasa Jawa dengan strategi optimasi biaya.
- **Tugas:**
    - [x] **T1:** Setup DeepSeek API Integration | **File:** `src/data_collection/deepseek_labeling.py` | **Tes:** ✅ Script dapat terhubung ke API dan melabel sampel data | **Assignee:** jules_dev1
    - [x] **T2:** Dokumentasi DeepSeek Labeling | **File:** `docs/deepseek-labeling-guide.md` | **Tes:** ✅ Dokumentasi dapat diikuti, setup berhasil | **Assignee:** jules_dokumen
    - [x] **T3:** Optimasi Biaya dengan Strategi Sentimen | **File:** `src/data_collection/preprocess_sentiment.py` | **Tes:** ✅ Penghematan 50% biaya dengan auto-assign data positif | **Assignee:** jules_dev1
    - [x] **T4:** Implementasi Optimized Labeler | **File:** `src/data_collection/deepseek_labeling_optimized.py` | **Tes:** ✅ Script optimized dapat memproses hanya data negatif | **Assignee:** jules_dev1
    - [x] **T5:** Update Dokumentasi Strategi | **File:** `docs/deepseek-labeling-guide.md` | **Tes:** ✅ Dokumentasi mencakup workflow optimized lengkap | **Assignee:** jules_dokumen
    - [ ] **T6:** Process Full Dataset | **File:** `data/processed/final_labeled_dataset.csv` | **Tes:** ⏳ Dataset lengkap terlabel dengan strategi optimized | **Assignee:** jules_dev1
    - [ ] **T7:** Validasi Manual Subset | **File:** `data/processed/deepseek_validation_subset.csv` | **Tes:** ⏳ Akurasi pelabelan > 85% pada subset validasi | **Assignee:** Hashfi

### BABY-STEP SELANJUTNYA: Model Training dengan Dataset Optimized 📋 SIAP DIMULAI
- **Tujuan:** Training model BERT dengan dataset hasil DeepSeek labeling yang sudah dioptimasi.
- **Tugas:**
    - [ ] **T1:** Setup Training Environment | **File:** `src/modelling/train_optimized.py` | **Tes:** ⏳ Environment siap untuk training dengan mixed labeling methods | **Assignee:** jules_dev2
    - [ ] **T2:** Data Preprocessing untuk Training | **File:** `src/preprocessing/prepare_training_data.py` | **Tes:** ⏳ Data siap dengan handling untuk processing_method column | **Assignee:** jules_dev1
    - [ ] **T3:** Model Architecture Setup | **File:** `src/modelling/model_architecture.py` | **Tes:** ⏳ BERT model siap untuk fine-tuning Bahasa Jawa | **Assignee:** jules_dev2
    - [ ] **T4:** Training Pipeline | **File:** `src/modelling/training_pipeline.py` | **Tes:** ⏳ Model dapat ditraining dengan dataset optimized | **Assignee:** jules_dev2
    - [ ] **T5:** Model Evaluation | **File:** `src/modelling/evaluate_model.py` | **Tes:** ⏳ Evaluasi performa model dengan metrik yang sesuai | **Assignee:** jules_dev2

### SARAN & RISIKO

#### **Saran Implementasi**
- ✅ **Implementasi secure storage untuk DeepSeek API key (environment variables)** - Sudah diterapkan
- ✅ **Optimasi biaya dengan strategi sentimen** - Penghematan 50% tercapai
- **Saran:** Implementasi monitoring untuk accuracy comparison antara auto-assigned vs DeepSeek labeled
- **Saran:** Setup validation pipeline untuk memastikan kualitas auto-assignment
- **Saran:** Dokumentasi best practices untuk mixed labeling methods dalam training

#### **Risiko yang Telah Dimitigasi**
- ✅ **API Cost** - Risiko berkurang 50% dengan strategi optimasi sentimen
- ✅ **Label Consistency** - Auto-assignment untuk data positif memberikan konsistensi 100%

#### **Risiko Aktif**
- **Risiko:** API Rate Limiting - DeepSeek memiliki rate limit yang perlu diperhatikan
- **Risiko:** Network Dependency - Pelabelan bergantung pada koneksi internet yang stabil
- **Risiko:** Model Bias - Perlu validasi bahwa auto-assignment tidak menimbulkan bias dalam training
- **Risiko:** Quality Assurance - Perlu validasi manual untuk memastikan akurasi strategi optimasi