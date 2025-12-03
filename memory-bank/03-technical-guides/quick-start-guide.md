# Quick Start Guide - Javanese Hate Speech Detection Labeling System

## Untuk Tim Baru

### 1. Setup Awal (5 menit)

#### Prasyarat
- Python 3.8+
- Akses internet untuk DeepSeek API
- Google Drive account (untuk persistence)

#### Langkah Setup
```bash
# 1. Clone dan masuk ke direktori
cd ujaran-kebencian-bahasa-jawa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment
cp .env.template .env
# Edit .env dengan API keys Anda

# 4. Test environment
python src/check_env.py
```

### 2. Memulai Labeling (2 menit)

#### Quick Demo
```bash
# Demo cepat dengan 10 data
python src/demo_cost_efficient_labeling.py
```

#### Production Labeling
```bash
# Labeling dengan persistence
python src/demo_persistent_labeling.py
```

### 3. Monitoring Progress

#### Cek Status
- **Hasil labeling**: `src/quick-demo-results.csv`
- **Checkpoint**: `src/checkpoints/`
- **Logs**: `src/logs/`

#### Cost Tracking
```bash
# Lihat estimasi biaya
python src/demo_cost_optimization.py
```

## Workflow Harian

### Morning Routine
1. **Cek status checkpoint** - Lihat progress kemarin
2. **Review cost budget** - Pastikan dalam batas anggaran
3. **Start labeling session** - Jalankan persistent labeling

### Evening Routine
1. **Backup checkpoint** - Sync ke Google Drive
2. **Review results** - Cek kualitas labeling
3. **Update progress** - Catat di `memory-bank/papan-proyek.md`

## Troubleshooting Cepat

### Error Umum

#### "DeepSeek API Error"
```bash
# Cek API key
python src/check_env.py
# Cek quota/billing di DeepSeek dashboard
```

#### "Google Drive Error"
```bash
# Test koneksi
python src/test_google_drive_integration.py
# Cek credentials di .env
```

#### "Checkpoint Not Found"
```bash
# List available checkpoints
ls src/checkpoints/
# Restore dari Google Drive jika perlu
```

### Performance Issues

#### Labeling Terlalu Lambat
- Gunakan `deepseek-chat` untuk data sederhana
- Batch size lebih kecil (5-10 items)
- Labeling pada jam discount (19:00-08:00 UTC+8)

#### Cost Terlalu Tinggi
- Aktifkan "Discount Only" mode
- Review strategi di `memory-bank/cost-optimization-strategy.md`
- Split positive/negative data untuk efisiensi

## Resources

### Dokumentasi Lengkap
- **Architecture**: `memory-bank/architecture.md`
- **API Strategy**: `memory-bank/deepseek-api-strategy.md`
- **Cost Optimization**: `memory-bank/cost-optimization-strategy.md`
- **Google Drive Setup**: `memory-bank/google-drive-persistence-strategy.md`

### Panduan Manual
- **Labeling Guidelines**: `memory-bank/petunjuk-pekerjaan-manual.md`
- **Product Spec**: `memory-bank/spesifikasi-produk.md`

### Tim & Roles
- **Team Manifest**: `vibe-guide/team-manifest.md`
- **Coding Guide**: `vibe-guide/VIBE_CODING_GUIDE.md`

## Kontak & Support

Jika mengalami masalah:
1. Cek dokumentasi di `memory-bank/`
2. Review error logs di `src/logs/`
3. Konsultasi dengan tim arsitek
4. Update issue di `memory-bank/papan-proyek.md`

---

**💡 Tips**: Selalu backup checkpoint sebelum eksperimen besar, dan monitor cost secara real-time untuk menghindari overspending.