# Status, To-Do List, dan Saran

**Update Terakhir:** 2024-07-01 (Menggantikan tanggal lama)

## 1. Status Proyek Saat Ini:

*   **Tahap 0 (Pengaturan Awal Proyek & Lingkungan Dasar) selesai.**
    *   Repositori Git di-clone, branch `development` aktif (konseptual).
    *   Struktur folder dasar diperiksa.
    *   Lingkungan Conda `ujaran` (Python 3.11) disimulasikan aktif.
    *   Dependensi dari `requirements.txt` (termasuk Google API Client, Pandas, Numpy) berhasil diinstal.
    *   Validasi environment berhasil menggunakan `src/environment_check.py`.
*   Dokumen perencanaan awal (`dokumen-desain-produk.md`, `tumpukan-teknologi.md`, `rencana-implementasi.md`, `architecture.md`) telah dibuat (versi 0.1).
*   Proyek siap untuk memasuki **Tahap 1: Pengumpulan & Preprocessing Data Awal.**

## 2. To-Do List (Fokus MVP - Berdasarkan `rencana-implementasi.md`):

### Fase 1: Pengumpulan & Preprocessing Data Awal (Sedang Berlangsung)
*   [ ] **Persiapkan akses ke dataset pribadi di Google Sheets melalui Google Sheets API.** (Catatan: Pengguna mengindikasikan memiliki CSV, langkah ini akan disesuaikan)
    *   [ ] Pastikan file kredensial (misalnya, `credentials.json`) untuk Google Cloud Platform (GCP) tersedia dan dikonfigurasi dengan benar dengan akses ke Google Sheets API. (Perlu didiskusikan bagaimana cara aman mengelola file ini - jangan di-commit ke repo jika berisi informasi sensitif).
    *   [ ] Identifikasi ID Google Sheet yang akan digunakan.
    *   [ ] Identifikasi nama sheet/tab dan range data yang akan diambil.
*   [ ] **Kembangkan script untuk mengambil data (~500-1000 sampel) dari Google Sheets API.** (Catatan: Akan diubah menjadi memuat dari CSV)
    *   [ ] Buat script Python di `src/data_collection/` (misalnya, `sheet_reader.py`).
    *   [ ] Implementasikan autentikasi ke Google Sheets API menggunakan file kredensial.
    *   [ ] Implementasikan fungsi untuk membaca data dari sheet dan range yang ditentukan.
    *   [ ] Simpan data mentah yang diambil (misalnya, ke file CSV di folder `/data/raw/` - folder ini perlu dibuat).
*   [ ] **Implementasi preprocessing dasar untuk data dari Google Sheets.** (Catatan: Sumber data adalah CSV)
    *   [ ] Kembangkan script di `src/preprocessing/` (misalnya, `clean_data.py`).
    *   [ ] Fungsi untuk memuat data mentah.
    *   [ ] Lakukan pembersihan dan normalisasi data (misalnya, menghapus spasi berlebih, menangani nilai kosong sederhana).
    *   [ ] Simpan data yang sudah diproses (misalnya, ke file CSV di folder `/data/processed/` - folder ini perlu dibuat).
*   [ ] Implementasikan fungsi filtering dasar untuk konten duplikat.

### Fase 2: Pelabelan Data Awal & Persiapan Dataset
*   [ ] Definisikan pedoman pelabelan awal (jika belum final dari `petunjuk-pekerjaan-manual.md`).
*   [ ] Lakukan pelabelan manual pada sampel data awal (~200-500 sampel).
*   [ ] Bagi dataset berlabel menjadi set pelatihan dan validasi.

### Fase 3: Pengembangan Model Dasar (Fine-tuning IndoBERT)
*   [ ] Setup environment untuk Hugging Face Transformers dan PyTorch/TensorFlow.
*   [ ] Kembangkan script untuk tokenisasi data.
*   [ ] Kembangkan script untuk fine-tuning IndoBERT.
*   [ ] Simpan model yang sudah di-fine-tune.

### Fase 4: Evaluasi Model Dasar
*   [ ] Kembangkan script untuk prediksi pada set validasi.
*   [ ] Implementasikan perhitungan metrik evaluasi dasar (Akurasi).
*   [ ] (Opsional MVP) Implementasikan confusion matrix.

### Fase 5: Pengembangan Prototipe API Sederhana
*   [ ] Pilih framework API dan setup struktur dasar.
*   [ ] Buat endpoint API `/detect`.
*   [ ] (Opsional MVP) Buat antarmuka web sangat sederhana.

## 3. Saran "Baby-Step To-Do List" untuk Langkah Berikutnya (Fase 1 - Disesuaikan untuk CSV):

### Baby Step 2: Memproses Dataset CSV yang Disediakan
1.  **Konfirmasi Lokasi CSV:**
    *   **Tugas Pengguna:** Pastikan file CSV (`raw-dataset.csv`) berada di `src/data_collection/dataset/raw-dataset.csv` di branch `main` (atau branch lain yang disepakati) dan sandbox saya telah sinkron dengan benar untuk melihatnya.
    *   **Saya (Jules):** Akan mencoba lagi untuk mengakses file di lokasi tersebut.
2.  **Buat Struktur Folder Data (jika perlu):**
    *   **Saya (Jules):** Jika belum ada dan jika file CSV tidak di `src/data_collection/dataset/`, buat folder `/data/raw/` dan `/data/processed/`. Tambahkan `.gitkeep` ke dalamnya agar folder kosong bisa di-commit.
3.  **Script untuk Memuat dan Inspeksi CSV:**
    *   **Saya (Jules):** Buat script Python di `src/data_collection/` (misalnya, `load_csv_dataset.py`).
    *   Script ini akan:
        *   Memuat `raw-dataset.csv` menggunakan Pandas.
        *   Menampilkan informasi dasar: jumlah baris & kolom, nama kolom, beberapa baris pertama (`df.head()`, `df.info()`).
        *   Menyimpan output inspeksi ini ke sebuah file teks di `memory-bank` atau mencetaknya.
4.  **Pindahkan CSV ke Lokasi Standar (Opsional, Rekomendasi):**
    *   **Diskusi:** Sebaiknya file data mentah seperti CSV berada di `data/raw/` bukan di dalam `src/`. Jika pengguna setuju, saya akan memindahkan file dari `src/data_collection/dataset/raw-dataset.csv` ke `data/raw/raw-dataset.csv` dan memperbarui script pemuatan.

---