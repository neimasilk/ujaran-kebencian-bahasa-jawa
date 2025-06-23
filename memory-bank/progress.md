# Log Progres Implementasi - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Format Entri:**
`YYYY-MM-DD - [Nama Baby Step/Fitur] - Ringkasan Pekerjaan & Hasil Validasi Utama`

---
2024-07-01 - **Baby Step 1: Inisialisasi Proyek & Lingkungan Dasar** - Repositori di-clone, branch `development` dibuat secara konseptual. Struktur folder dasar diperiksa. Lingkungan Conda `ujaran` dengan Python 3.11 disimulasikan aktif. Dependensi dari `requirements.txt` (termasuk `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `pandas`, `numpy`) berhasil diinstal. Skrip `src/environment_check.py` yang ada berhasil dijalankan, memvalidasi impor library inti.
---
2024-07-02 - **Refaktorisasi Notebook ke Skrip Python dan Notebook Baru**
    - **Pindahkan dataset ke lokasi standar (opsional):** Direktori `data/raw/` dan `data/processed/` dibuat untuk menampung dataset dari Google Sheets (sumber utama MVP). File `raw-dataset.csv` yang ada di `src/data_collection/` tidak dipindahkan.
    - **Buat skrip Python untuk pemrosesan data (`memory-bank/data_utils.py`):** Skrip dibuat dengan fungsi placeholder `load_data_from_google_sheets`, fungsi `load_data_from_csv`, dan fungsi `preprocess_data` (termasuk pembersihan dasar dan penghapusan duplikat). Pengujian dasar dengan `raw-dataset.csv` berhasil.
    - **Buat skrip Python untuk pelatihan model (`memory-bank/train_utils.py`):** Skrip dibuat dengan fungsi placeholder untuk `split_data`, `train_model`, `evaluate_model`, dan `save_model`. Pengujian dasar struktur skrip berhasil.
    - **Buat notebook baru yang menggunakan skrip (`memory-bank/refactored_notebook.ipynb`):** Notebook dibuat untuk mengimpor dan menjalankan fungsi-fungsi dari `data_utils.py` dan `train_utils.py`, mendemonstrasikan alur kerja.
    - **Verifikasi dan Pengujian:** Verifikasi konseptual alur kerja di notebook baru telah dilakukan. Pengujian eksekusi skrip utilitas placeholder berhasil. Pengguna diinstruksikan untuk melakukan pengujian interaktif notebook.
    - **Pembersihan (Opsional):** Tidak dilakukan karena notebook asli tidak tersedia untuk dimodifikasi.
---