# Panduan Sinkronisasi Data Lokal ke Google Drive

## Situasi: Cloud (Google Drive) Sudah Dibersihkan

Jika Anda telah menghapus semua data di Google Drive tetapi masih memiliki data lokal, Anda dapat dengan mudah melakukan sinkronisasi ulang menggunakan script yang telah disediakan.

## 🔄 Cara Sinkronisasi Data Lokal ke Cloud

### 1. Sinkronisasi Lengkap (Checkpoint + Hasil Labeling)

```bash
python sync_local_to_cloud.py
```

Script ini akan:
- ✅ Mencari semua checkpoint lokal di `src/checkpoints/`
- ✅ Mencari semua file hasil labeling (`hasil-labeling*.csv`)
- ✅ Upload semua data ke Google Drive
- ✅ Memberikan laporan lengkap

### 2. Sinkronisasi Hanya Checkpoint

```bash
python sync_local_to_cloud.py --checkpoint-only
```

### 3. Sinkronisasi Hanya File Hasil

```bash
python sync_local_to_cloud.py --results-only
```

## 📊 Contoh Output Sinkronisasi

```
🔄 SINKRONISASI DATA LOKAL KE GOOGLE DRIVE
==================================================
🔧 Inisialisasi...
🔐 Autentikasi Google Drive...
✅ Berhasil terhubung ke Google Drive

📁 Ditemukan 1 checkpoint lokal:
   - labeling_raw-dataset_hasil-labeling.json

🔄 Mengupload checkpoints ke Google Drive...
   ✅ labeling_raw-dataset_hasil-labeling.json -> labeling_raw-dataset_hasil-labeling

📤 Berhasil upload 1/1 checkpoints

📊 Ditemukan 1 file hasil labeling:
   - hasil-labeling.csv

🔄 Mengupload hasil labeling ke Google Drive...
   ✅ hasil-labeling.csv -> hasil-labeling_20250701_092343.csv

📤 Berhasil upload 1/1 file hasil

==================================================
🎉 SINKRONISASI SELESAI
📤 Total file yang diupload: 2
```

## 🔍 Verifikasi Sinkronisasi

Setelah sinkronisasi, Anda dapat memverifikasi dengan:

```bash
python src/google_drive_labeling.py --status
```

Output yang diharapkan:
```
============================================================
📊 GOOGLE DRIVE LABELING STATUS
============================================================
Time Status: ⏰ Regular Time
Cloud Status: ☁️ Connected
Local Checkpoint: ✅ Found
Local Results: ✅ Found
============================================================
```

## 🚀 Melanjutkan Proses Labeling

Setelah sinkronisasi berhasil, Anda dapat melanjutkan proses labeling:

```bash
python src/google_drive_labeling.py --dataset src/data_collection/raw-dataset.csv --output hasil-labeling
```

Sistem akan:
- ✅ Otomatis mendeteksi checkpoint yang sudah ada
- ✅ Melanjutkan dari posisi terakhir
- ✅ Sinkronisasi otomatis ke Google Drive setiap checkpoint

## 🔧 Fitur Sinkronisasi Otomatis

Setelah integrasi cloud yang baru, sistem sekarang memiliki:

### 1. **Sinkronisasi Checkpoint Otomatis**
- Setiap checkpoint lokal otomatis disinkronkan ke Google Drive
- Interval sinkronisasi dapat diatur (default: setiap 50 batch)

### 2. **Backup Emergency**
- Saat terjadi error atau interrupsi (Ctrl+C), checkpoint emergency otomatis dibackup ke cloud
- Format: `checkpoint_id_emergency_timestamp`

### 3. **Sinkronisasi Hasil Akhir**
- File hasil labeling otomatis diupload saat proses selesai
- Backup dengan timestamp untuk menghindari overwrite

## 📁 Struktur Data di Google Drive

Setelah sinkronisasi, struktur di Google Drive:

```
ujaran-kebencian-datasets/
├── checkpoints/
│   └── labeling_raw-dataset_hasil-labeling.json
└── datasets/
    └── hasil-labeling_20250701_092343.csv
```

## ⚠️ Troubleshooting

### 1. Error Autentikasi
```bash
❌ Gagal autentikasi Google Drive
```
**Solusi:**
- Pastikan file `credentials.json` ada
- Jalankan `python src/google_drive_labeling.py --setup` untuk setup ulang

### 2. Tidak Ada Data Lokal
```bash
📁 Tidak ada checkpoint lokal yang ditemukan
📊 Tidak ada file hasil labeling yang ditemukan
```
**Solusi:**
- Periksa apakah ada file di `src/checkpoints/`
- Periksa apakah ada file `hasil-labeling*.csv` di root directory

### 3. Upload Gagal
```bash
❌ Gagal upload checkpoint_name
```
**Solusi:**
- Periksa koneksi internet
- Periksa quota Google Drive
- Coba jalankan ulang script

## 💡 Tips Penggunaan

1. **Backup Reguler**: Jalankan `sync_local_to_cloud.py` secara berkala untuk backup manual

2. **Monitoring**: Gunakan `--status` untuk memantau status sinkronisasi

3. **Recovery**: Jika terjadi masalah, data selalu tersedia di kedua tempat (lokal + cloud)

4. **Cleanup**: Gunakan script di `src/scripts/` untuk maintenance cloud storage

## 🎯 Kesimpulan

✅ **Ya, Anda bisa sinkronisasi dari lokal ke cloud setelah cloud dibersihkan!**

Sistem telah dirancang untuk:
- Fleksibilitas dalam sinkronisasi data
- Recovery yang mudah dari berbagai skenario
- Backup otomatis untuk mencegah kehilangan data
- Integrasi seamless antara storage lokal dan cloud

Dengan script `sync_local_to_cloud.py`, Anda dapat dengan mudah memulihkan semua data di Google Drive dari backup lokal yang ada.