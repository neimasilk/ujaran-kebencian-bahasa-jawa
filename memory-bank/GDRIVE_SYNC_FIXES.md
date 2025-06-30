# Google Drive Sync Fixes

## Masalah yang Diperbaiki

### 1. Ctrl+C Tidak Keluar dari Program
**Masalah**: Ketika menekan Ctrl+C, program hanya pause dan kemudian lanjut lagi, bukan keluar dan menyimpan progress.

**Solusi**:
- Diperbaiki signal handling di `labeling.py` dan `google_drive_labeling.py`
- Ctrl+C sekarang akan:
  - Menghentikan proses labeling
  - Menyimpan checkpoint emergency
  - Melakukan sync paksa ke Google Drive
  - Keluar dari program dengan pesan instruksi untuk melanjutkan

### 2. Google Drive Kosong
**Masalah**: File tidak ter-upload ke Google Drive meskipun tidak ada error yang terlihat.

**Solusi**:
- Diperbaiki method `sync_to_cloud()` di `google_drive_labeling.py`:
  - Menambahkan pengecekan authentication status
  - Menambahkan logging yang lebih detail
  - Menggunakan method yang tepat untuk upload checkpoint dan dataset
  - Menambahkan error handling yang lebih baik

- Menambahkan method `upload_dataset()` di `cloud_checkpoint_manager.py`:
  - Method ini sebelumnya tidak ada
  - Sekarang bisa upload file CSV ke folder datasets di Google Drive

- Memperbaiki temporary file handling di Windows:
  - Mengatasi error "file being used by another process"
  - Menggunakan proper cleanup di finally block

### 3. Sync Otomatis Selama Proses
**Peningkatan**: Menambahkan periodic sync setiap 5 menit selama proses labeling berjalan untuk memastikan progress tidak hilang.

## File yang Dimodifikasi

1. **`src/labeling.py`**
   - Mengubah behavior Ctrl+C dari pause menjadi stop dan save
   - Menambahkan pesan yang jelas tentang sync ke Google Drive

2. **`src/google_drive_labeling.py`**
   - Memperbaiki method `sync_to_cloud()`
   - Menambahkan periodic sync thread
   - Memperbaiki signal handling untuk emergency save
   - Menambahkan import json yang hilang

3. **`src/utils/cloud_checkpoint_manager.py`**
   - Menambahkan method `upload_dataset()` untuk upload file CSV
   - Memperbaiki temporary file handling di Windows
   - Menambahkan error handling yang lebih robust

## Testing

Dibuat script test `test_gdrive_sync.py` untuk memverifikasi:
- ✅ Authentication ke Google Drive
- ✅ Upload dan download checkpoint
- ✅ Upload dataset file
- ✅ Status checking

## Cara Menggunakan

1. **Menjalankan Labeling**:
   ```bash
   python src/labeling.py --model deepseek --output hasil_labeling --cloud
   ```

2. **Menghentikan dengan Ctrl+C**:
   - Tekan Ctrl+C untuk menghentikan proses
   - Program akan otomatis menyimpan progress dan sync ke Google Drive
   - Jalankan ulang command yang sama untuk melanjutkan dari checkpoint terakhir

3. **Test Google Drive Sync**:
   ```bash
   python test_gdrive_sync.py
   ```

## Struktur Folder di Google Drive

```
ujaran-kebencian-datasets/
├── checkpoints/
│   ├── checkpoint_hasil_labeling_20250701_031520.json
│   └── ...
└── datasets/
    ├── results_hasil_labeling_20250701_031524.csv
    └── ...
```

## Catatan Penting

- Pastikan file `credentials.json` ada di root directory
- Pada first run, akan diminta untuk authenticate ke Google Drive
- File akan disimpan dengan timestamp untuk menghindari overwrite
- Sync berjalan otomatis setiap 5 menit selama proses labeling
- Emergency sync akan dilakukan saat Ctrl+C atau error