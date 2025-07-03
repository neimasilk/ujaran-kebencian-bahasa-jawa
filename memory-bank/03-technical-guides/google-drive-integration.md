# Google Drive Integration - Sistem Pelabelan Dataset

**Dokumentator:** AI Assistant  
**Update Terakhir:** 2024-12-29  
**Versi:** 2.0 (Konsolidasi)  
**Status:** Aktif  

---

## 🎯 Overview

Dokumen ini mengkonsolidasikan semua informasi terkait integrasi Google Drive untuk sistem pelabelan dataset ujaran kebencian Bahasa Jawa. Menggabungkan setup guide, troubleshooting, dan strategi persistence dalam satu dokumen komprehensif.

## 🏗️ Arsitektur Sistem

### Latar Belakang Masalah
- Pengguna sering berpindah komputer (kampus, rumah, komputer berbeda)
- Dataset dan checkpoint perlu tetap persistent dan dapat diakses dari mana saja
- Solusi lokal tidak memadai untuk skenario multi-device
- **Tim kolaboratif**: Multiple users dapat melabeli secara bersamaan dari device berbeda
- **Sinkronisasi dua arah**: Data di cloud bisa lebih baru (tim lain melabeli) atau data lokal lebih baru
- **Conflict resolution**: Perlu strategi untuk menangani konflik data antara local dan cloud

### Struktur Cloud Storage
```
Google Drive (ujaran-kebencian-datasets/)
├── checkpoints/                    ✅ # Cloud checkpoints
│   ├── labeling_raw-dataset_hasil-labeling.json
│   ├── emergency_checkpoints/
│   └── interrupted_session_backups/
├── datasets/                       ✅ # Hasil labeling
│   ├── hasil-labeling.csv
│   └── backup_datasets/
└── logs/                          ✅ # System logs
    ├── labeling_logs/
    └── error_logs/
```

## 🔧 Setup Guide

### Prasyarat
- Google account dengan akses Google Drive
- Python 3.8+
- Akses internet stabil

### Step 1: Setup Google Drive API

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
   - Klik "Create Credentials" → "OAuth client ID"
   - Application type: "Desktop application"
   - Name: "Ujaran Kebencian Labeling"
   - Download JSON file

5. **Setup Credentials:**
   ```bash
   # Rename downloaded file
   mv ~/Downloads/client_secret_*.json credentials.json
   
   # Copy ke project directory
   cp credentials.json /path/to/ujaran-kebencian-bahasa-jawa/
   ```

### Step 2: Environment Setup

```bash
# Install dependencies
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Test authentication
python src/scripts/test_gdrive_sync.py
```

### Step 3: Verifikasi Setup

```bash
# Test upload/download
python src/scripts/test_emergency_sync.py

# Check folder structure
python src/scripts/check_gdrive_results.py
```

## 🛠️ Troubleshooting

### Masalah Umum dan Solusi

#### 1. Ctrl+C Tidak Keluar dari Program
**Gejala**: Ketika menekan Ctrl+C, program hanya pause dan kemudian lanjut lagi.

**Solusi**:
- Signal handling telah diperbaiki di `labeling.py` dan `google_drive_labeling.py`
- Ctrl+C sekarang akan:
  - Menghentikan proses labeling
  - Menyimpan checkpoint emergency
  - Melakukan sync paksa ke Google Drive
  - Keluar dari program dengan pesan instruksi

#### 2. Google Drive Kosong
**Gejala**: File tidak ter-upload ke Google Drive meskipun tidak ada error.

**Solusi**:
- Method `sync_to_cloud()` telah diperbaiki dengan:
  - Pengecekan authentication status
  - Logging yang lebih detail
  - Error handling yang lebih baik
  - Method `upload_dataset()` yang sebelumnya tidak ada

#### 3. Authentication Errors
**Gejala**: Error "credentials not found" atau "authentication failed".

**Solusi**:
```bash
# Hapus token lama
rm token.json

# Re-authenticate
python src/scripts/test_gdrive_sync.py
```

#### 4. Quota Exceeded
**Gejala**: Error "quota exceeded" saat upload.

**Solusi**:
- Gunakan akun Google dengan storage yang cukup
- Bersihkan file lama di Google Drive
- Implementasi batch upload untuk file besar

## 🔄 Operasional

### Sync Strategies

#### 1. Automatic Sync
- Checkpoint disimpan setiap 100 samples
- Emergency sync saat Ctrl+C
- Periodic sync setiap 30 menit

#### 2. Manual Sync
```bash
# Sync manual ke cloud
python sync_local_to_cloud.py

# Download dari cloud
python src/scripts/test_emergency_sync_with_auth.py
```

#### 3. Conflict Resolution
- Cloud checkpoint selalu prioritas (Strict Cloud-First Policy)
- Local checkpoint sebagai backup
- Timestamp comparison untuk resolusi konflik

### Monitoring dan Maintenance

#### Check Status
```bash
# Monitor current labeling
python src/scripts/monitor_current_labeling.py

# Analyze checkpoint
python src/scripts/analyze_checkpoint.py

# Calculate progress
python src/scripts/calculate_progress.py
```

#### Cleanup
```bash
# Clear local checkpoints
python src/scripts/clear_all_checkpoints.py

# Delete cloud checkpoints
python src/scripts/delete_cloud_checkpoints.py

# Delete cloud results
python src/scripts/delete_gdrive_results.py
```

## 🔒 Security Best Practices

1. **Credentials Management**:
   - Jangan commit `credentials.json` ke repository
   - Gunakan `.env` untuk sensitive data
   - Rotate credentials secara berkala

2. **Access Control**:
   - Limit scope OAuth ke Google Drive saja
   - Gunakan service account untuk production
   - Monitor access logs secara berkala

3. **Data Protection**:
   - Encrypt sensitive data sebelum upload
   - Backup credentials di tempat aman
   - Implementasi audit trail

## 📊 Performance Optimization

### Upload Optimization
- Batch upload untuk multiple files
- Compression untuk file besar
- Resume capability untuk upload yang terputus

### Download Optimization
- Incremental download
- Cache management
- Parallel download untuk multiple files

## 🧪 Testing

### Test Scenarios
1. **Graceful Interruption**: Test Ctrl+C handling
2. **Network Interruption**: Test offline/online transitions
3. **Authentication Expiry**: Test token refresh
4. **Conflict Resolution**: Test multi-user scenarios
5. **Large File Handling**: Test dengan dataset besar

### Test Commands
```bash
# Run all tests
python src/scripts/test_recovery_scenarios.py

# Specific test scenarios
python src/scripts/test_recovery_scenarios.py --scenario graceful
python src/scripts/test_recovery_scenarios.py --scenario network
python src/scripts/test_recovery_scenarios.py --scenario auth
```

## 📚 Referensi

- **Architecture**: `architecture.md` - Arsitektur sistem lengkap
- **API Strategy**: `deepseek-api-strategy.md` - Strategi penggunaan DeepSeek API
- **Cost Optimization**: `cost-optimization-strategy.md` - Optimasi biaya operasional
- **Quick Start**: `quick-start-guide.md` - Panduan cepat memulai
- **Team Guide**: `../vibe-guide/team-manifest.md` - Panduan tim

---

*Dokumen ini mengkonsolidasikan informasi dari GDRIVE_SYNC_FIXES.md, GOOGLE_DRIVE_SETUP_GUIDE.md, dan google-drive-persistence-strategy.md untuk kemudahan maintenance dan referensi.*