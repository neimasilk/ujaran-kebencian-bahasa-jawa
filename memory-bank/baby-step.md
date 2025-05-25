# Baby Step Implementasi

**Tanggal Pembuatan:** 26 Mei 2025

## Judul Baby Step: Inisialisasi Proyek & Lingkungan Dasar

**Tujuan:** Menyiapkan fondasi teknis proyek, termasuk manajemen versi, struktur direktori, dan lingkungan Python dasar beserta dependensinya.

**Referensi Dokumen:**
* `rencana-implementasi.md` (Fase 0)
* `status-todolist-saran.md` (Saran "Baby-Step To-Do List")
* `tumpukan-teknologi.md` (untuk versi Python)

**Detail Tugas:**

1.  **Buat Repositori Baru di GitHub:**
    * Nama Repositori (saran): `deteksi-ujarankebencian-jawa-bert`
    * Visibilitas: Publik (sesuai tujuan publikasi dataset/model) atau Privat (sesuai preferensi awal).
    * Inisialisasi dengan README: Ya (bisa diisi dengan konten `readme.md` yang sudah dibuat).
    * Tambahkan .gitignore: Pilih template Python.
    * Pilih Lisensi: (Opsional di awal, bisa ditambahkan nanti, misalnya MIT atau Apache 2.0).
    * **Validasi:** Repositori berhasil dibuat di GitHub dan dapat diakses melalui URL.

2.  **Clone Repositori ke Lokal:**
    * Gunakan perintah `git clone [URL_REPO_ANDA_DARI_LANGKAH_1]`.
    * **Validasi:** Folder proyek dengan nama repositori muncul di direktori lokal dan berisi file dari GitHub (misalnya, `README.md`, `.gitignore`).

3.  **Buat Struktur Folder Proyek Dasar:**
    * Di dalam folder proyek yang sudah di-clone, buat direktori berikut:
        * `data/`
            * `data/raw/`
            * `data/processed/`
        * `notebooks/`
        * `src/`
            * `src/data_collection/` (buat file `__init__.py` kosong di dalamnya)
            * `src/preprocessing/` (buat file `__init__.py` kosong di dalamnya)
            * `src/modelling/` (buat file `__init__.py` kosong di dalamnya)
            * `src/api/` (buat file `__init__.py` kosong di dalamnya)
            * `src/utils/` (buat file `__init__.py` kosong di dalamnya, untuk fungsi utilitas umum)
        * `models/`
        * `tests/`
        * (Opsional) `docs/`
    * **Validasi:** Verifikasi manual bahwa semua folder dan subfolder telah dibuat dengan benar.

4.  **Inisialisasi dan Aktivasi Virtual Environment Python:**
    * Pastikan Anda memiliki Python versi 3.8+ terinstall di sistem Anda (sesuai `tumpukan-teknologi.md`). Anda bisa memeriksanya dengan `python --version` atau `python3 --version`.
    * Dari root folder proyek di terminal, jalankan: `python -m venv .venv` (atau `python3 -m venv .venv` jika perintah `python` Anda merujuk ke versi Python 2.x).
    * Aktivasi environment:
        * Windows: `.venv\Scripts\activate`
        * macOS/Linux: `source .venv/bin/activate`
    * **Validasi:** Prompt terminal berubah, menunjukkan nama environment (misalnya, `(.venv) ...`). Setelah aktivasi, Anda bisa memeriksa versi Python di dalam venv dengan `python --version` untuk memastikan sesuai.

5.  **Install Library Python Dasar:**
    * Pastikan virtual environment aktif.
    * Jalankan: `pip install pandas numpy jupyterlab scikit-learn`
    * **Validasi:** Tidak ada error saat instalasi. Jalankan `pip list` atau `pip freeze` untuk melihat library yang terinstall. Coba import di interpreter Python: `import pandas as pd; import numpy as np; import sklearn; print("Libraries imported successfully")`.

6.  **Buat File `requirements.txt` Awal:**
    * Pastikan virtual environment aktif.
    * Jalankan: `pip freeze > requirements.txt`
    * **Validasi:** File `requirements.txt` dibuat di root folder proyek dan berisi daftar library yang baru diinstall beserta versinya.

7.  **Update/Periksa File `.gitignore`:**
    * Pastikan file `.gitignore` yang di-generate GitHub (atau buat baru jika tidak ada) mencakup:
        ```
        .venv/
        __pycache__/
        *.pyc
        *.ipynb_checkpoints
        # Tambahkan file/folder data besar jika tidak ingin di-commit
        # data/raw/*
        # models/*
        # .DS_Store
        ```
    * **Validasi:** Periksa isi `.gitignore`.

8.  **Commit dan Push Perubahan Awal:**
    * Jalankan `git add .`
    * Jalankan `git commit -m "Initial project setup: directory structure, venv, and base libraries"`
    * Jalankan `git push origin main` (atau `master` tergantung nama branch default Anda).
    * **Validasi:** Perubahan (struktur folder, `requirements.txt`) terlihat di repositori GitHub.

--- 