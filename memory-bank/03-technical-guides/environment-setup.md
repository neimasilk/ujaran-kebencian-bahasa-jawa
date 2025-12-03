# Environment Setup - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Mengikuti:** Vibe Coding Guide v1.4
**Update:** 29 Desember 2024

## Spesifikasi Environment

* **Nama Environment:** `ujaran`
* **Python Version:** 3.11.x (Sesuai requirements.txt)
* **Package Manager:** Anaconda/Miniconda
* **OS Support:** Windows, macOS, Linux

## Library Dasar Awal (untuk MVP Fase 0)
* pandas
* numpy
* jupyterlab
* scikit-learn
* google-api-python-client
* google-auth-httplib2
* google-auth-oauthlib

## Cara Membuat dan Mengaktifkan Environment Awal
```bash
# Pastikan Anaconda/Miniconda sudah terinstall
# Buat environment baru dengan versi Python spesifik (3.11, sesuai requirements.txt yang ada)
conda create -n ujaran python=3.11
conda activate ujaran
```

## Cara Install Library Dasar Awal
```bash
# Setelah environment aktif, install library dasar yang dibutuhkan
conda install pandas numpy jupyterlab scikit-learn
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

**Catatan mengenai `requirements.txt`:**
File `requirements.txt` berisi daftar lengkap semua dependencies. Untuk setup awal proyek, Anda bisa memulai dengan menginstall library dasar di atas. Nantinya, `requirements.txt` dapat digunakan untuk mereplikasi environment secara keseluruhan.

## Cara Install Semua Dependencies dari requirements.txt
```bash
pip install -r requirements.txt
```

## Cara Export Dependencies
```bash
# Setelah menginstall semua library yang dibutuhkan untuk proyek
conda list --export > requirements.txt
# atau untuk format yang lebih umum kompatibel dengan pip
pip freeze > requirements.txt
```

## Verifikasi Setup

Setelah setup selesai, jalankan script verifikasi:
```bash
python check_env.py
```

Script ini akan memverifikasi:
- ✅ Python version compatibility
- ✅ Required packages installation
- ✅ Dataset accessibility
- ✅ Basic functionality

## Troubleshooting

### Common Issues
1. **Conda command not found:** Pastikan Anaconda/Miniconda sudah terinstall dan PATH sudah dikonfigurasi
2. **Package conflicts:** Gunakan `conda clean --all` kemudian reinstall
3. **Permission errors:** Jalankan terminal sebagai administrator (Windows) atau gunakan `sudo` (Linux/macOS)

### Referensi Terkait
- **Panduan Utama:** [`../vibe-guide/VIBE_CODING_GUIDE.md`](../vibe-guide/VIBE_CODING_GUIDE.md)
- **Quick Start:** [`../readme.md`](../readme.md)
- **Tim Support:** [`../vibe-guide/team-manifest.md`](../vibe-guide/team-manifest.md)

---
**Catatan:** File ini berisi informasi teknis tentang setup environment proyek. Gunakan sebagai referensi untuk setup ulang atau dokumentasi. Untuk bantuan lebih lanjut, hubungi tim sesuai manifest.