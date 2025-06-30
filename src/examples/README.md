# Examples - Ujaran Kebencian Bahasa Jawa

Direktori ini berisi file-file demo dan contoh penggunaan untuk berbagai komponen dalam proyek Ujaran Kebencian Bahasa Jawa.

## 📁 Daftar File Demo

### 1. `demo_cloud_checkpoint.py`
**Tujuan**: Demo untuk CloudCheckpointManager - mengelola checkpoint di Google Drive

**Fitur yang didemonstrasikan**:
- Inisialisasi CloudCheckpointManager
- Upload dan download checkpoint ke/dari Google Drive
- Sinkronisasi otomatis dengan cloud storage
- Manajemen cache lokal

**Cara menjalankan**:
```bash
cd src/examples
python demo_cloud_checkpoint.py
```

**Prasyarat**:
- File `credentials.json` untuk Google Drive API
- Koneksi internet
- Folder `demo-checkpoints` akan dibuat otomatis

---

### 2. `demo_cost_efficient_labeling.py`
**Tujuan**: Demo optimasi biaya untuk labeling menggunakan DeepSeek API

**Fitur yang didemonstrasikan**:
- Pembagian data untuk efisiensi biaya
- Analisis penghematan biaya
- Perbandingan performa
- Metrik kualitas labeling

**Cara menjalankan**:
```bash
cd src/examples
python demo_cost_efficient_labeling.py
```

**Prasyarat**:
- Environment variable `DEEPSEEK_API_KEY`
- File dataset di `src/data_collection/raw-dataset.csv`

---

### 3. `demo_cost_optimization.py`
**Tujuan**: Demo optimasi biaya real-time untuk DeepSeek API

**Fitur yang didemonstrasikan**:
- Deteksi periode diskon otomatis
- Estimasi biaya real-time
- Strategi optimasi biaya
- Monitoring penggunaan API

**Cara menjalankan**:
```bash
cd src/examples
python demo_cost_optimization.py
```

**Prasyarat**:
- Environment variable `DEEPSEEK_API_KEY`
- Koneksi internet untuk cek harga real-time

---

### 4. `demo_persistent_labeling.py`
**Tujuan**: Demo pipeline labeling dengan checkpoint persistence

**Fitur yang didemonstrasikan**:
- Labeling dengan checkpoint otomatis
- Simulasi interupsi (Ctrl+C)
- Resume dari checkpoint
- Manajemen daftar checkpoint

**Cara menjalankan**:
```bash
cd src/examples
python demo_persistent_labeling.py
```

**Prasyarat**:
- Environment variable `DEEPSEEK_API_KEY`
- File `credentials.json` untuk Google Drive
- File dataset di `src/data_collection/raw-dataset.csv`

---

## 🚀 Cara Menggunakan Demo

### Setup Awal
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment variables**:
   ```bash
   # Buat file .env di root project
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

3. **Setup Google Drive API**:
   - Download `credentials.json` dari Google Cloud Console
   - Letakkan di root project
   - Jalankan demo pertama kali untuk autentikasi

### Menjalankan Demo
1. **Pilih demo yang ingin dijalankan**
2. **Baca prasyarat di atas**
3. **Jalankan dari direktori examples/**:
   ```bash
   cd src/examples
   python nama_demo.py
   ```

### Tips Penggunaan
- **Mulai dengan `demo_cloud_checkpoint.py`** untuk memastikan Google Drive integration berfungsi
- **Gunakan `demo_cost_optimization.py`** untuk memahami strategi penghematan biaya
- **Jalankan `demo_persistent_labeling.py`** untuk memahami workflow labeling lengkap
- **Eksperimen dengan `demo_cost_efficient_labeling.py`** untuk optimasi batch processing

---

## 🔧 Troubleshooting

### Error: "No module named 'utils'"
**Solusi**: Pastikan menjalankan dari direktori `src/examples/` atau tambahkan src ke PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../"
```

### Error: "credentials.json not found"
**Solusi**: 
1. Download credentials dari Google Cloud Console
2. Letakkan di root project (bukan di src/)
3. Pastikan nama file exact: `credentials.json`

### Error: "DEEPSEEK_API_KEY not found"
**Solusi**:
1. Buat file `.env` di root project
2. Tambahkan: `DEEPSEEK_API_KEY=your_key_here`
3. Atau set environment variable langsung

### Error: "raw-dataset.csv not found"
**Solusi**: Pastikan file dataset ada di `src/data_collection/raw-dataset.csv`

---

## 📚 Dokumentasi Terkait

- **Vibe Coding Guide**: `../vibe-guide/VIBE_CODING_GUIDE.md`
- **Product Specification**: `../memory-bank/spesifikasi-produk.md`
- **Project Board**: `../memory-bank/papan-proyek.md`
- **Team Manifest**: `../memory-bank/team-manifest.md`

---

## 🤝 Kontribusi

Jika Anda ingin menambahkan demo baru:
1. Ikuti naming convention: `demo_[feature_name].py`
2. Tambahkan dokumentasi di README ini
3. Pastikan demo bisa dijalankan standalone
4. Tambahkan error handling yang memadai
5. Update refactoring plan di `memory-bank/refactoring-plan.md`

---

**Author**: AI Assistant & Human Team  
**Last Updated**: 2025-01-01  
**Version**: 1.0.0