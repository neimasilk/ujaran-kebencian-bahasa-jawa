# Baby Steps - Panduan Detail Implementasi Awal

**Versi:** 1.0  
**Tanggal:** 26 Mei 2025  
**Target:** Junior Developer  
**Tujuan:** Menghilangkan ambiguitas dalam langkah implementasi awal

## Prasyarat

- **Sistem Operasi:** Windows (sesuai environment saat ini)
- **Git:** Sudah terinstall dan dikonfigurasi
- **Anaconda/Miniconda:** Sudah terinstall
- **Akses Internet:** Untuk download dependencies
- **Text Editor/IDE:** VS Code, PyCharm, atau Cursor

## Baby Step 1: Inisialisasi Proyek & Lingkungan Dasar

### 1.1 Clone Repositori Git ke Lokal

**Tujuan:** Mendapatkan salinan lokal dari repositori proyek

**Langkah:**
```bash
# Buka Command Prompt atau PowerShell
# Navigasi ke direktori tempat Anda ingin menyimpan proyek
cd d:\documents

# Clone repositori (ganti URL dengan URL repositori yang sebenarnya)
git clone <URL_REPOSITORI> ujaran-kebencian-bahasa-jawa

# Masuk ke direktori proyek
cd ujaran-kebencian-bahasa-jawa
```

**Validasi:**
- Direktori `ujaran-kebencian-bahasa-jawa` terbuat
- File `.gitignore`, `readme.md`, `requirements.txt` ada
- Folder `src`, `data`, `memory-bank`, `tests` ada

**Troubleshooting:**
- Jika error "git not found": Install Git dari https://git-scm.com/
- Jika error permission: Pastikan Anda memiliki akses ke repositori

### 1.2 Buat Branch Development

**Tujuan:** Membuat branch terpisah untuk development agar tidak mengubah main branch

**Langkah:**
```bash
# Pastikan Anda berada di direktori proyek
pwd  # Harus menunjukkan path ke ujaran-kebencian-bahasa-jawa

# Cek branch saat ini
git branch

# Buat dan pindah ke branch development
git checkout -b development

# Verifikasi branch aktif
git branch
```

**Validasi:**
- Output `git branch` menunjukkan `* development` (tanda * menunjukkan branch aktif)
- Branch `main` atau `master` juga terlihat dalam daftar

**Troubleshooting:**
- Jika branch development sudah ada: `git checkout development`
- Jika ada uncommitted changes: `git stash` terlebih dahulu

### 1.3 Verifikasi Struktur Folder Proyek

**Tujuan:** Memastikan struktur folder sesuai dengan rencana di readme.md

**Langkah:**
```bash
# Lihat struktur folder saat ini
dir  # Windows
# atau
ls   # jika menggunakan Git Bash

# Lihat isi folder src
dir src

# Lihat isi folder data
dir data
```

**Struktur yang Diharapkan:**
```
ujaran-kebencian-bahasa-jawa/
├── .gitignore
├── data/
│   ├── processed/
│   └── raw/
├── docs/
├── memory-bank/
├── models/
├── notebooks/
├── readme.md
├── requirements.txt
├── src/
│   ├── api/
│   ├── data_collection/
│   ├── modelling/
│   ├── preprocessing/
│   └── utils/
└── tests/
```

**Validasi:**
- Semua folder utama ada: `data`, `src`, `memory-bank`, `tests`, `models`, `notebooks`
- File `readme.md` dan `requirements.txt` ada
- Subfolder dalam `src` ada: `api`, `data_collection`, `modelling`, `preprocessing`, `utils`
- Subfolder dalam `data` ada: `raw`, `processed`

**Troubleshooting:**
- Jika folder tidak ada: Buat manual dengan `mkdir nama_folder`
- Jika struktur berbeda: Sesuaikan dengan struktur yang diharapkan

### 1.4 Setup Conda Environment

**Tujuan:** Membuat environment Python terisolasi untuk proyek

**Langkah:**
```bash
# Cek apakah conda sudah terinstall
conda --version

# Cek environment yang sudah ada
conda env list

# Buat environment baru bernama 'ujaran' dengan Python 3.11
conda create -n ujaran python=3.11 -y

# Aktifkan environment
conda activate ujaran

# Verifikasi environment aktif
conda info --envs
```

**Validasi:**
- Command prompt menunjukkan `(ujaran)` di awal
- `conda info --envs` menunjukkan tanda `*` di environment ujaran
- `python --version` menunjukkan Python 3.11.x

**Troubleshooting:**
- Jika conda tidak ditemukan: Restart terminal atau tambahkan conda ke PATH
- Jika environment sudah ada: `conda activate ujaran`
- Jika Python version salah: Hapus environment dan buat ulang

### 1.5 Persiapkan requirements.txt

**Tujuan:** Memastikan file requirements.txt mencakup library yang dibutuhkan untuk Fase 1

**Langkah:**

**5.1 Periksa isi requirements.txt saat ini:**
```bash
# Lihat isi file requirements.txt
type requirements.txt  # Windows
# atau
cat requirements.txt   # Git Bash/Linux
```

**5.2 Cek apakah Google Sheets API libraries ada:**
```bash
# Cari library Google API dalam requirements.txt
findstr "google" requirements.txt  # Windows
# atau
grep "google" requirements.txt     # Git Bash/Linux
```

**5.3 Jika library Google API belum ada, tambahkan:**

Buka `requirements.txt` dengan text editor dan tambahkan baris berikut di akhir file:
```
google-api-python-client==2.88.0
google-auth-httplib2==0.1.0
google-auth-oauthlib==1.0.0
```

**Validasi:**
- File `requirements.txt` berisi library Google API
- File dapat dibuka tanpa error
- Library memiliki versi yang spesifik

**Troubleshooting:**
- Jika file tidak bisa diedit: Pastikan tidak read-only
- Jika format salah: Pastikan satu library per baris

### 1.6 Install Dependencies

**Tujuan:** Menginstall semua library yang dibutuhkan dari requirements.txt

**Langkah:**
```bash
# Pastikan environment ujaran aktif
conda activate ujaran

# Update pip ke versi terbaru
python -m pip install --upgrade pip

# Install dependencies dari requirements.txt
pip install -r requirements.txt

# Verifikasi instalasi library krusial
python -c "import pandas; print('Pandas version:', pandas.__version__)"
python -c "import googleapiclient; print('Google API Client installed successfully')"
```

**Validasi:**
- Tidak ada error saat `pip install -r requirements.txt`
- Import pandas berhasil dan menampilkan versi
- Import googleapiclient berhasil
- `pip list` menunjukkan library yang terinstall

**Troubleshooting:**
- Jika ada error dependency conflict: Gunakan `pip install --force-reinstall`
- Jika download gagal: Periksa koneksi internet
- Jika permission error: Gunakan `pip install --user`

### 1.7 Buat Script Validasi Environment

**Tujuan:** Membuat script untuk memvalidasi bahwa environment setup berhasil

**Langkah:**

**7.1 Buat file check_env.py di root proyek:**
```python
#!/usr/bin/env python3
"""
Script validasi environment untuk proyek Ujaran Kebencian Bahasa Jawa
Tujuan: Memastikan semua library krusial terinstall dengan benar
"""

import sys
import importlib
from datetime import datetime

def check_library(library_name, import_name=None):
    """Check if a library can be imported successfully"""
    if import_name is None:
        import_name = library_name
    
    try:
        lib = importlib.import_module(import_name)
        version = getattr(lib, '__version__', 'Unknown')
        print(f"✅ {library_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {library_name}: FAILED - {e}")
        return False

def main():
    print("=" * 50)
    print("VALIDASI ENVIRONMENT PROYEK UJARAN KEBENCIAN")
    print(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Libraries to check
    libraries = [
        ('Pandas', 'pandas'),
        ('NumPy', 'numpy'),
        ('Google API Client', 'googleapiclient'),
        ('Google Auth', 'google.auth'),
        ('Google Auth HTTPLib2', 'google_auth_httplib2'),
        ('Google Auth OAuthLib', 'google_auth_oauthlib'),
        ('Scikit-learn', 'sklearn'),
        ('Jupyter Lab', 'jupyterlab')
    ]
    
    success_count = 0
    total_count = len(libraries)
    
    print("Checking Libraries:")
    print("-" * 30)
    
    for lib_name, import_name in libraries:
        if check_library(lib_name, import_name):
            success_count += 1
    
    print()
    print("=" * 50)
    print(f"HASIL: {success_count}/{total_count} libraries berhasil")
    
    if success_count == total_count:
        print("🎉 ENVIRONMENT SETUP BERHASIL!")
        print("Anda siap melanjutkan ke Baby Step 2")
        return True
    else:
        print("⚠️  ENVIRONMENT SETUP BELUM LENGKAP")
        print("Silakan install library yang gagal sebelum melanjutkan")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**7.2 Jalankan script validasi:**
```bash
# Pastikan berada di root proyek dan environment ujaran aktif
python check_env.py
```

**Validasi:**
- Script berjalan tanpa error
- Semua library menampilkan ✅ (checkmark)
- Pesan "ENVIRONMENT SETUP BERHASIL!" muncul
- Exit code 0 (sukses)

**Troubleshooting:**
- Jika ada library yang ❌: Install library tersebut dengan `pip install nama_library`
- Jika script tidak bisa dijalankan: Periksa syntax dan indentasi
- Jika permission error: Pastikan file tidak read-only

### 1.8 Dokumentasi dan Commit

**Tujuan:** Mendokumentasikan progress dan menyimpan perubahan ke Git

**Langkah:**

**8.1 Update progress.md:**
```bash
# Buka file memory-bank/progress.md dengan text editor
# Tambahkan entri baru:
```

Tambahkan di file `memory-bank/progress.md`:
```markdown
2025-05-26 - [Baby Step 1] - Setup environment dan validasi berhasil
- Environment conda 'ujaran' dengan Python 3.11 dibuat
- Dependencies dari requirements.txt terinstall
- Google Sheets API libraries tersedia
- Script check_env.py berjalan sukses (8/8 libraries)
- Struktur folder proyek sesuai rencana
```

**8.2 Commit perubahan:**
```bash
# Add semua perubahan
git add .

# Commit dengan pesan yang jelas
git commit -m "Baby Step 1: Setup environment dan validasi awal

- Tambah Google API libraries ke requirements.txt
- Buat script check_env.py untuk validasi
- Update progress.md dengan hasil Baby Step 1
- Environment ujaran dengan Python 3.11 siap digunakan"

# Push ke remote repository (opsional)
git push origin development
```

**Validasi:**
- `git status` menunjukkan "working tree clean"
- `git log --oneline -1` menunjukkan commit terbaru
- File progress.md terupdate dengan entri baru

## Kriteria Keberhasilan Baby Step 1

✅ **Environment Setup:**
- Conda environment 'ujaran' dengan Python 3.11 aktif
- Semua dependencies terinstall tanpa error

✅ **Project Structure:**
- Struktur folder sesuai dengan rencana
- Branch development aktif

✅ **Validation:**
- Script check_env.py berjalan sukses (8/8 libraries)
- Google Sheets API libraries dapat diimport

✅ **Documentation:**
- Progress terdokumentasi di progress.md
- Perubahan di-commit ke Git

## Langkah Selanjutnya

Setelah Baby Step 1 berhasil, Anda siap untuk:
- **Baby Step 2:** Setup Google Sheets API credentials
- **Baby Step 3:** Implementasi script pengumpulan data dari Google Sheets

## Troubleshooting Umum

### Problem: Conda command not found
**Solution:**
1. Restart terminal/command prompt
2. Atau tambahkan Anaconda ke PATH environment variable
3. Atau gunakan Anaconda Prompt

### Problem: Permission denied saat install
**Solution:**
1. Jalankan command prompt sebagai Administrator
2. Atau gunakan `pip install --user`

### Problem: Library conflict
**Solution:**
1. Hapus environment: `conda remove -n ujaran --all`
2. Buat ulang environment dari awal

### Problem: Git authentication error
**Solution:**
1. Setup Git credentials: `git config --global user.name "Your Name"`
2. Setup email: `git config --global user.email "your.email@example.com"`
3. Atau gunakan SSH key untuk authentication

---

**Catatan:** Dokumen ini dirancang untuk menghilangkan ambiguitas dalam implementasi. Jika ada langkah yang tidak jelas, silakan update dokumen ini dengan detail yang lebih spesifik.