# 🚀 Google Drive Labeling Setup Guide

## ✅ Semua Syarat Anda Sudah Terpenuhi!

### 📋 **Checklist Syarat:**
- ✅ **Bisa di-cancel sewaktu-waktu dan bisa diresume** → Checkpoint otomatis + Ctrl+C support
- ✅ **Labeling pas jam promo** → Deteksi otomatis jam promo DeepSeek (00:00-06:00, 18:00-23:59)
- ✅ **Dataset hasil labeling di Google Drive** → Auto-sync setiap 100 samples

---

## 🔧 **Setup Steps (5-10 menit)**

### **Step 1: Setup Google Drive API**

1. **Buka Google Cloud Console:**
   - Go to: https://console.cloud.google.com/
   - Login dengan akun Google Anda

2. **Create Project Baru:**
   - Klik "Select a project" → "New Project"
   - Nama project: `ujaran-kebencian-labeling`
   - Klik "Create"

3. **Enable Google Drive API:**
   - Search "Google Drive API" di search bar
   - Klik "Google Drive API" → "Enable"

4. **Create OAuth Credentials:**
   - Go to "Credentials" (di sidebar kiri)
   - Klik "+ CREATE CREDENTIALS" → "OAuth client ID"
   - Application type: **"Desktop application"**
   - Name: `Ujaran Kebencian Labeling`
   - Klik "Create"

5. **Download Credentials:**
   - Klik "Download JSON" pada credentials yang baru dibuat
   - Rename file menjadi `credentials.json`
   - **Pindahkan ke folder project:** `c:/Users/neima/Documents/ujaran-kebencian-bahasa-jawa/credentials.json`

### **Step 2: Test Setup**

```bash
# Check status (harus menunjukkan "Cloud Status: ☁️ Connected")
python src/google_drive_labeling.py --status
```

---

## 🚀 **Cara Menjalankan Labeling**

### **Mode 1: Otomatis (Recommended)**
```bash
# Tunggu jam promo, lalu mulai labeling otomatis
python src/google_drive_labeling.py
```

**Features:**
- ⏰ Tunggu sampai jam promo (00:00-06:00 atau 18:00-23:59)
- 🔄 Resume otomatis dari checkpoint terakhir
- ☁️ Sync ke Google Drive setiap 100 samples
- 🛑 Pause otomatis jika jam promo habis

### **Mode 2: Mulai Sekarang (Tanpa Tunggu Promo)**
```bash
# Mulai labeling sekarang juga
python src/google_drive_labeling.py --no-promo-wait
```

### **Mode 3: Fresh Start (Tanpa Resume)**
```bash
# Mulai dari awal tanpa resume checkpoint
python src/google_drive_labeling.py --no-resume
```

---

## 🎛️ **Advanced Options**

```bash
# Custom dataset
python src/google_drive_labeling.py --dataset path/to/your/dataset.csv

# Custom output name
python src/google_drive_labeling.py --output my-labeling-results

# Custom batch size (default: 10)
python src/google_drive_labeling.py --batch-size 20

# Custom sync interval (default: 100 samples)
python src/google_drive_labeling.py --cloud-sync-interval 50
```

---

## 🔄 **Workflow Multi-Device**

### **Skenario: Labeling di Kampus, Lanjut di Rumah**

**Di Kampus:**
```bash
# Mulai labeling
python src/google_drive_labeling.py
# ... labeling 500 samples ...
# Tekan Ctrl+C untuk stop
# ✅ Otomatis tersimpan ke Google Drive
```

**Di Rumah:**
```bash
# Setup project yang sama
git clone <your-repo>
cd ujaran-kebencian-bahasa-jawa
pip install -r requirements.txt

# Copy credentials.json ke folder ini
# Lalu jalankan:
python src/google_drive_labeling.py
# ✅ Otomatis download checkpoint dari Google Drive
# ✅ Lanjut dari sample ke-501
```

---

## 📊 **Monitoring & Status**

### **Check Status:**
```bash
python src/google_drive_labeling.py --status
```

**Output Example:**
```
📊 GOOGLE DRIVE LABELING STATUS
Time Status: 🎉 PROMO TIME
Cloud Status: ☁️ Connected
Local Checkpoint: ✅ 2.5 KB
Local Results: ✅ 150 KB (1,250 samples)
```

### **Check Promo Hours:**
- **Promo Hours:** 00:00-06:00 dan 18:00-23:59
- **Regular Hours:** 06:00-18:00
- Sistem otomatis pause di regular hours dan resume di promo hours

---

## 🛑 **Cancel & Resume**

### **Cara Cancel:**
- Tekan **Ctrl+C** kapan saja
- Sistem akan:
  1. Save checkpoint lokal
  2. Upload ke Google Drive
  3. Exit gracefully

### **Cara Resume:**
```bash
# Jalankan command yang sama
python src/google_drive_labeling.py
# ✅ Otomatis resume dari checkpoint terakhir
```

---

## 📁 **File Output**

### **Lokal:**
- `google-drive-labeling.csv` → Hasil labeling
- `checkpoints/labeling_google-drive-labeling.json` → Checkpoint
- `logs/` → Log files

### **Google Drive:**
- Folder: `ujaran-kebencian-labeling/`
- `results_google-drive-labeling_YYYYMMDD_HHMMSS.csv`
- `checkpoint_google-drive-labeling_YYYYMMDD_HHMMSS.json`

---

## 🚨 **Troubleshooting**

### **Error: "credentials.json not found"**
```bash
# Check apakah file ada
ls credentials.json

# Jika tidak ada, download lagi dari Google Cloud Console
```

### **Error: "Google Drive authentication failed"**
```bash
# Hapus token lama dan coba lagi
rm token.json
python src/google_drive_labeling.py --status
```

### **Error: "DeepSeek API failed"**
```bash
# Check API key di .env
cat .env | grep DEEPSEEK

# Test API connection
python -c "from src.utils.deepseek_client import create_deepseek_client; print('✅ OK' if create_deepseek_client() else '❌ Failed')"
```

---

## 💡 **Tips Optimasi**

### **Untuk Jam Promo:**
1. **Setup dulu sebelum jam promo** (18:00 atau 00:00)
2. **Jalankan dengan mode otomatis** untuk maksimal efficiency
3. **Monitor progress** dengan `--status`

### **Untuk Multi-Device:**
1. **Sync credentials.json** ke semua device
2. **Gunakan output name yang sama** di semua device
3. **Check status** sebelum mulai labeling

### **Untuk Dataset Besar:**
1. **Kurangi cloud-sync-interval** untuk backup lebih sering
2. **Gunakan batch-size lebih kecil** untuk checkpoint lebih granular
3. **Monitor storage Google Drive** (15GB free)

---

## 🎯 **Ready to Start!**

**Current Status:** 🎉 **PROMO TIME DETECTED!** (Jam 01:xx)

**Next Steps:**
1. ✅ Dependencies installed
2. ✅ DeepSeek API ready
3. ⏳ **Setup credentials.json** (5 menit)
4. 🚀 **Start labeling!**

```bash
# Setelah setup credentials.json:
python src/google_drive_labeling.py
```

**Estimated Time:**
- Setup: 5-10 menit
- Labeling: Tergantung dataset size
- 41,759 samples ≈ 8-12 jam (dengan promo hours optimization)

---

**🎉 Selamat labeling! Semua syarat Anda sudah terpenuhi dengan sistem yang robust dan user-friendly!**