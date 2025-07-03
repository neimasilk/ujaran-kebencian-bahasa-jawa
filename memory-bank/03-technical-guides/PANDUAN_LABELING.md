# 🏷️ Panduan Labeling Ujaran Kebencian Bahasa Jawa

Panduan lengkap untuk melakukan labeling dataset ujaran kebencian bahasa Jawa menggunakan DeepSeek API dengan backup otomatis ke Google Drive.

## 📋 Persyaratan

### 1. Software yang Diperlukan
- Python 3.8 atau lebih baru
- Git (untuk clone repository)
- Akun Google (untuk Google Drive backup)
- Akun DeepSeek API

### 2. Setup Awal

#### A. Clone Repository
```bash
git clone <repository-url>
cd ujaran-kebencian-bahasa-jawa
```

#### B. Install Dependencies
```bash
pip install -r requirements.txt
```

#### C. Setup Environment Variables
1. Copy file `.env.template` menjadi `.env`:
   ```bash
   copy .env.template .env
   ```

2. Edit file `.env` dan isi dengan API key DeepSeek Anda:
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

#### D. Setup Google Drive (Opsional tapi Direkomendasikan)
1. Ikuti panduan di `GOOGLE_DRIVE_SETUP_GUIDE.md`
2. Download file `credentials.json` dari Google Cloud Console
3. Letakkan file `credentials.json` di root folder project

## 🚀 Cara Menggunakan

### Perintah Sederhana (Recommended)
```bash
python labeling.py
```

**Itu saja!** Script akan otomatis:
- ✅ Membaca dataset dari `src/data_collection/raw-dataset.csv`
- ✅ Melakukan labeling menggunakan DeepSeek API
- ✅ Menyimpan hasil ke `hasil-labeling.csv`
- ✅ Backup otomatis ke Google Drive
- ✅ Checkpoint otomatis (bisa resume kapan saja)

### Perintah Manual (Advanced)
Jika ingin kontrol lebih detail:
```bash
python src/google_drive_labeling.py --dataset src/data_collection/raw-dataset.csv --output hasil-labeling
```

### Parameter Tambahan
```bash
# Lihat status labeling
python src/google_drive_labeling.py --status

# Setup Google Drive (panduan)
python src/google_drive_labeling.py --setup

# Mulai dari awal (tidak resume)
python src/google_drive_labeling.py --dataset src/data_collection/raw-dataset.csv --output hasil-labeling --no-resume
```

## 🔄 Fitur Utama

### 1. Auto-Resume
- Jika proses terhenti (Ctrl+C, error, dll), jalankan perintah yang sama untuk melanjutkan
- Progress disimpan otomatis setiap 50 data
- Tidak akan mengulang data yang sudah dilabeling

### 2. Google Drive Backup
- Hasil otomatis tersimpan ke Google Drive
- Bisa diakses dari device lain
- Sinkronisasi otomatis setiap 100 data

### 3. Cost Optimization
- Deteksi jam promo DeepSeek otomatis
- Batch processing untuk efisiensi
- Monitoring biaya real-time

### 4. Error Handling
- Retry otomatis jika API error
- Emergency save jika terjadi crash
- Log detail untuk debugging

## 📁 Output Files

Setelah labeling selesai, Anda akan mendapatkan:

1. **`hasil-labeling.csv`** - Hasil labeling utama
2. **`checkpoints/labeling_raw-dataset_hasil-labeling.json`** - Checkpoint file
3. **`logs/`** - Log files untuk debugging
4. **Google Drive folder** - Backup online

## 🛠️ Troubleshooting

### Error: "DEEPSEEK_API_KEY not found"
- Pastikan file `.env` sudah dibuat dan berisi API key yang valid
- Cek apakah API key masih aktif di dashboard DeepSeek

### Error: "Google Drive authentication failed"
- Pastikan file `credentials.json` ada di root folder
- Jalankan `python src/google_drive_labeling.py --setup` untuk panduan

### Error: "Dataset not found"
- Pastikan file `src/data_collection/raw-dataset.csv` ada
- Cek format CSV (harus ada kolom 'text')

### Proses Lambat
- Tunggu jam promo DeepSeek (biasanya malam hari)
- Atau gunakan `--no-promo-wait` untuk mulai langsung

## 📊 Monitoring Progress

```bash
# Lihat status real-time
python src/google_drive_labeling.py --status

# Lihat log
tail -f logs/labeling_*.log

# Cek hasil sementara
head -20 hasil-labeling.csv
```

## 💡 Tips

1. **Jalankan di Background**: Gunakan `nohup` atau `screen` untuk proses panjang
2. **Monitor Biaya**: Cek dashboard DeepSeek secara berkala
3. **Backup Manual**: Sesekali copy file hasil ke tempat aman
4. **Batch Kecil**: Untuk dataset besar, bagi menjadi beberapa batch

## 🆘 Bantuan

Jika mengalami masalah:
1. Cek file log di folder `logs/`
2. Jalankan `python src/google_drive_labeling.py --status`
3. Baca error message dengan teliti
4. Hubungi tim development dengan menyertakan log error

---

**Happy Labeling! 🎯**

*Dibuat dengan ❤️ untuk penelitian ujaran kebencian bahasa Jawa*