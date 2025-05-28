# Setup Environment Proyek

## Conda Environment

* Nama Environment: `ujaran`
* Python Version: Python 3.8+ (Direkomendasikan 3.11.x seperti pada `requirements.txt` yang ada)
* Package Manager: Anaconda

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

---
**Catatan:** File ini berisi informasi teknis tentang setup environment proyek. Gunakan sebagai referensi untuk setup ulang atau dokumentasi. 