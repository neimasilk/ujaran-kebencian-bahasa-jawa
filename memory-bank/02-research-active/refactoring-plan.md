# Rencana Refactoring Proyek Ujaran Kebencian Bahasa Jawa

## Status Saat Ini

### Masalah yang Diidentifikasi
1. **File tes dan demo tersebar** - Ada banyak file demo dan tes di root `src/` yang membuat struktur tidak rapi
2. **Duplikasi environment check** - Ada `check_env.py` dan `environment_check.py` yang fungsinya mirip
3. **File tes di luar direktori tests** - `test_deepseek_negative_10.py` dan `test_google_drive_integration.py` ada di root `src/`
4. **Banyak file demo** - Multiple demo files yang bisa dikonsolidasi
5. **Entry point tidak konsisten** - Ada `labeling.py` di root dan `google_drive_labeling.py` di `src/`

### Struktur Saat Ini
```
src/
├── api/
├── config/
├── data_collection/
├── modelling/
├── preprocessing/
├── tests/
├── utils/
├── notebooks/
├── checkpoints/
├── demo_*.py (5 files)
├── test_*.py (2 files)
├── check_env.py
├── environment_check.py
├── google_drive_labeling.py
└── count_dataset.py
```

## Rencana Refactoring

### 1. Reorganisasi File Tes

#### Aksi:
- **Pindahkan** `test_deepseek_negative_10.py` → `tests/integration/test_deepseek_api.py`
- **Pindahkan** `test_google_drive_integration.py` → `tests/integration/test_google_drive.py`
- **Gabungkan** `check_env.py` dan `environment_check.py` → `tests/test_environment.py`
- **Buat** direktori `tests/integration/` untuk integration tests

#### Justifikasi:
- Semua tes berada di satu tempat (`tests/`)
- Pemisahan unit tests dan integration tests
- Mengurangi clutter di root `src/`

### 2. Konsolidasi File Demo

#### Aksi:
- **Buat** direktori `examples/` di dalam `src/`
- **Pindahkan** semua `demo_*.py` → `examples/`
- **Buat** `examples/README.md` dengan penjelasan setiap demo
- **Pertimbangkan** menggabungkan demo yang serupa

#### File Demo yang Ada:
1. `demo_cloud_checkpoint.py` - Demo cloud checkpoint operations
2. `demo_cost_efficient_labeling.py` - Demo cost-efficient strategies
3. `demo_cost_optimization.py` - Demo cost optimization
4. `demo_persistent_labeling.py` - Demo persistent labeling

#### Konsolidasi yang Diusulkan:
- **Gabungkan** `demo_cost_efficient_labeling.py` + `demo_cost_optimization.py` → `examples/cost_optimization_demo.py`
- **Pertahankan** `demo_cloud_checkpoint.py` → `examples/cloud_checkpoint_demo.py`
- **Pertahankan** `demo_persistent_labeling.py` → `examples/persistent_labeling_demo.py`

### 3. Perbaikan Entry Point

#### Aksi:
- **Pertahankan** `labeling.py` di root sebagai main entry point
- **Pastikan** `google_drive_labeling.py` tetap di `src/` sebagai core module
- **Tambahkan** dokumentasi yang jelas tentang cara penggunaan

### 4. Cleanup Utility Files

#### Aksi:
- **Hapus** `environment_check.py` (simple version)
- **Perbaiki** `check_env.py` dan pindahkan ke `tests/`
- **Review** `count_dataset.py` - apakah perlu dipindahkan ke `utils/` atau `examples/`

### 5. Struktur Akhir yang Diinginkan

```
src/
├── api/
├── config/
├── data_collection/
├── modelling/
├── preprocessing/
├── utils/
├── examples/
│   ├── README.md
│   ├── cloud_checkpoint_demo.py
│   ├── cost_optimization_demo.py
│   ├── persistent_labeling_demo.py
│   └── dataset_analysis.py (dari count_dataset.py)
├── tests/
│   ├── integration/
│   │   ├── test_deepseek_api.py
│   │   └── test_google_drive.py
│   ├── test_environment.py
│   ├── test_data_loading.py
│   ├── test_evaluation.py
│   └── test_preprocessing.py
├── notebooks/
├── checkpoints/
└── google_drive_labeling.py
```

## Implementasi Bertahap

### Fase 1: Reorganisasi Tes
1. Buat direktori `tests/integration/`
2. Pindahkan dan refactor integration tests
3. Gabungkan environment checks
4. Update import paths

### Fase 2: Konsolidasi Demo
1. Buat direktori `examples/`
2. Pindahkan dan konsolidasi demo files
3. Buat dokumentasi examples

### Fase 3: Cleanup dan Dokumentasi
1. Hapus file yang tidak diperlukan
2. Update dokumentasi
3. Test semua functionality

### Fase 4: Validasi
1. Jalankan semua tes
2. Verifikasi labeling pipeline masih berfungsi
3. Update panduan penggunaan

## Kriteria Sukses

1. ✅ **Struktur Rapi**: Semua file berada di direktori yang tepat
2. ✅ **Tes Terorganisir**: Unit tests dan integration tests terpisah
3. ✅ **Demo Terdokumentasi**: Examples mudah dipahami dan dijalankan
4. ✅ **Functionality Utuh**: Labeling pipeline tetap berfungsi sempurna
5. ✅ **Dokumentasi Lengkap**: Tim dapat memahami struktur dengan mudah

## Risiko dan Mitigasi

### Risiko:
- Import paths berubah bisa break existing code
- Demo scripts mungkin tidak berfungsi setelah dipindah

### Mitigasi:
- Test setiap perubahan secara bertahap
- Update import paths dengan hati-hati
- Backup sebelum melakukan perubahan besar
- Dokumentasikan setiap perubahan

## 📋 Implementation Status

### Phase 1: Test Organization ✅ COMPLETED
- [x] Create `tests/integration/` directory
- [x] Move and refactor `test_deepseek_negative_10.py` → `tests/integration/test_deepseek_api.py`
- [x] Move and refactor `test_google_drive_integration.py` → `tests/integration/test_google_drive.py`
- [x] Consolidate `check_env.py` + `environment_check.py` → `tests/test_environment.py`
- [x] Delete original scattered test files
- [x] Fix integration test compatibility with CloudCheckpointManager API
- [x] All integration tests passing (13/13)

### Phase 2: Demo Organization ✅ COMPLETED
- [x] Create `examples/` directory
- [x] Move demo files to `examples/`:
  - `demo_cloud_checkpoint.py`
  - `demo_cost_efficient_labeling.py`
  - `demo_cost_optimization.py`
  - `demo_persistent_labeling.py`
- [x] Create `examples/README.md` with comprehensive usage instructions
- [x] All demo files successfully moved and organized

## Timeline

Estimasi: 2-3 jam untuk implementasi lengkap
- Fase 1: 45 menit
- Fase 2: 45 menit  
- Fase 3: 30 menit
- Fase 4: 30 menit