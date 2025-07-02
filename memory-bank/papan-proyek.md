# Papan Proyek - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

### STATUS [Update: 2024-12-29]
- ✅ Proyek telah disiapkan sesuai dengan Vibe Coding Guide v1.4
- ✅ Spesifikasi produk telah diperbaiki dan disesuaikan dengan template PRD
- ✅ Tim manifest telah disiapkan dengan peran yang jelas
- ✅ Environment setup dan modular code structure telah diimplementasi
- ✅ Dataset inspection dan basic testing infrastructure telah selesai
- 🔄 Dokumentasi sedang disesuaikan dengan panduan terbaru

### REFERENSI ARSIP
- Baby-step sebelumnya: Implementasi Testing dan Dokumentasi API (selesai)
- Arsip lengkap tersedia di: `baby-steps-archive/`

### BABY-STEP SAAT INI

**"Model Training & Evaluation"** 🚀 SIAP DIMULAI
- **Tujuan:** Melatih model deteksi ujaran kebencian menggunakan dataset yang telah dilabeli secara otomatis dan mengevaluasi performanya secara menyeluruh.
- **Tugas:**
     - [ ] **T1: Implementasi `train_model.py`** | **File:** `src/modelling/train_model.py` | **Tes:** Skrip dapat memuat data dari `hasil-labeling.csv`, melakukan fine-tuning pada model, dan menyimpan model terlatih. | **Assignee:** Developer Backend
     - [ ] **T2: Implementasi `evaluate_model.py`** | **File:** `src/modelling/evaluate_model.py` | **Tes:** Skrip dapat memuat model terlatih, melakukan prediksi pada data uji, dan menghasilkan laporan metrik (akurasi, presisi, recall, F1-score) serta confusion matrix. | **Assignee:** Developer Backend
     - [ ] **T3: Integrasi Utilitas Pelatihan** | **File:** `src/modelling/train_utils.py` | **Tes:** Fungsi-fungsi bantuan untuk pelatihan (misalnya, data loader, optimizer setup) terintegrasi dengan baik. | **Assignee:** Developer Backend
     - [ ] **T4: Pembuatan Unit Test** | **File:** `src/tests/test_training.py`, `src/tests/test_evaluation.py` | **Tes:** Unit test untuk memverifikasi fungsionalitas skrip pelatihan dan evaluasi. | **Assignee:** Developer Backend

### BABY-STEP SELANJUTNYA

**"API Development & Prototyping"** ⏳ MENUNGGU
- **Tujuan:** Membangun API untuk menyajikan model dan membuat prototipe antarmuka pengguna sederhana.

### REFERENSI ARSIP
- **Arsip 1:** Production Deployment & Real Data Labeling (selesai)
- **Arsip 2:** Implementasi Testing dan Dokumentasi API (selesai)
- Arsip lengkap tersedia di: `baby-steps-archive/`

## ✅ Selesai Dikerjakan

### T5: Infrastructure Hardening & Data Labeling Preparation - SELESAI ✅
- T5.1: Architecture Review - SELESAI ✅
- T5.2: Dependencies Management - SELESAI ✅
- T5.3: Configuration Management - SELESAI ✅
- T5.4: Logging Infrastructure - SELESAI ✅
- T5.5: Documentation Consolidation - SELESAI ✅
- T4.3: Project Structure Reorganization - SELESAI ✅
- T5.6: Documentation Enhancement & Team Onboarding - SELESAI ✅
- T5.7: Parallel DeepSeek API Labeling Implementation - SELESAI ✅
- T5.8: Comprehensive Testing & Tutorial Documentation - SELESAI ✅

### SARAN & RISIKO (Review Arsitek)

**🎯 Saran Teknis:**
- **Prioritas 1:** Fokus pada kualitas data labeling - ini akan menentukan 80% dari performa model
- **Prioritas 2:** Setup environment yang konsisten untuk semua developer menggunakan virtual environment
- **Prioritas 3:** Implementasi logging yang komprehensif sejak awal untuk debugging dan monitoring
- **Best Practice:** Gunakan configuration management untuk semua parameter model dan API

**⚠️ Risiko Teknis:**
- **TINGGI:** Kualitas dataset - inconsistent labeling dapat merusak model performance
- **SEDANG:** IndoBERT compatibility dengan Bahasa Jawa - perlu extensive testing
- **SEDANG:** Dependencies conflicts - beberapa library ML memiliki version requirements yang strict
- **RENDAH:** API performance - model inference time perlu dioptimasi untuk production

**🔧 Mitigasi:**
- Buat clear labeling guidelines dan quality control process
- Prepare fallback model strategy jika IndoBERT tidak optimal
- Use pinned versions di requirements.txt dan virtual environment
- Implement model caching dan async processing untuk API