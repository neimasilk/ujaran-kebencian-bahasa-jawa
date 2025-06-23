# Status, To-Do List, dan Saran

**Update Terakhir:** 2024-07-03

## 1. Status Proyek Saat Ini:

*   **Tahap 0 (Pengaturan Awal Proyek & Lingkungan Dasar) selesai.**
    *   Repositori Git, struktur folder, lingkungan Conda, dan dependensi dasar telah disiapkan.
*   **Refaktorisasi Notebook ke Skrip dan Notebook Baru Selesai (dengan Placeholder):**
    *   Direktori `data/raw/` dan `data/processed/` telah dibuat.
    *   Skrip `memory-bank/data_utils.py` dibuat dengan fungsi untuk memuat data (CSV & placeholder Google Sheets) dan pra-pemrosesan dasar. Telah diuji dengan `raw-dataset.csv`.
    *   Skrip `memory-bank/train_utils.py` dibuat dengan kerangka fungsi untuk pembagian data, pelatihan model, evaluasi, dan penyimpanan model (semua masih placeholder).
    *   Notebook baru `memory-bank/refactored_notebook.ipynb` dibuat untuk menggunakan skrip-skrip utilitas di atas.
*   Dokumen perencanaan awal (`dokumen-desain-produk.md`, `tumpukan-teknologi.md`, `rencana-implementasi.md`, `architecture.md`, `petunjuk-pekerjaan-manual.md`) telah dibuat (versi 0.1).
*   Proyek siap untuk pengembangan lebih lanjut dari fungsi-fungsi placeholder, terutama yang berkaitan dengan Google Sheets API dan implementasi model Machine Learning sebenarnya.

## 2. To-Do List (Fokus MVP - Berdasarkan `rencana-implementasi.md` dan progres saat ini):

### Fase 1: Pengumpulan & Preprocessing Data Awal (Melanjutkan dari placeholder)
*   [ ] **Implementasi Penuh `load_data_from_google_sheets` di `data_utils.py`:**
    *   [ ] **Tugas Pengguna:** Menyediakan file kredensial Google Cloud Platform (misalnya, `credentials.json`) yang aman dan tidak di-commit ke repo, serta ID Google Sheet, nama sheet, dan range data yang akan digunakan.
    *   [ ] **Saya (Jules):** Mengimplementasikan logika untuk autentikasi dan pengambilan data dari Google Sheets API menggunakan kredensial dan detail sheet yang diberikan.
    *   [ ] Simpan data mentah yang diambil dari Google Sheets ke `/data/raw/` (misalnya, `downloaded_from_gsheets.csv`).
*   [ ] **Sempurnakan `preprocess_data` di `data_utils.py`:**
    *   [ ] Tambahkan langkah pra-pemrosesan yang lebih spesifik untuk Bahasa Jawa jika diperlukan (misalnya, normalisasi slang/singkatan, penanganan tingkatan bahasa jika relevan untuk tahap ini) berdasarkan `petunjuk-pekerjaan-manual.md` dan analisis data lebih lanjut.
    *   [ ] Pastikan outputnya adalah data bersih yang siap untuk pelabelan atau tokenisasi. Simpan ke `/data/processed/`.
*   [ ] Implementasikan fungsi filtering duplikat secara lebih robas jika placeholder saat ini belum cukup.

### Fase 2: Pelabelan Data Awal & Persiapan Dataset
*   [ ] **Finalisasi Pedoman Pelabelan:** Review `petunjuk-pekerjaan-manual.md` dan pastikan sudah mencakup semua kasus yang mungkin.
*   [ ] **Pelabelan Manual:** Lakukan pelabelan manual pada sampel data yang sudah diproses dari Google Sheets (~200-500 sampel awal).
    *   **Tugas Pengguna/Tim Pelabel:** Melakukan pelabelan ini.
*   [ ] **Implementasi Penuh `split_data` di `train_utils.py`:**
    *   [ ] Setelah data dilabeli, gunakan fungsi ini untuk membagi dataset berlabel menjadi set pelatihan dan validasi (misalnya, 80% train, 20% val). Simpan sebagai file terpisah di `/data/processed/` (misalnya, `train_set.csv`, `val_set.csv`).

### Fase 3: Pengembangan Model Dasar (Fine-tuning IndoBERT)
*   [ ] **Setup Environment ML Lengkap:**
    *   **Saya (Jules) & Tugas Pengguna:** Pastikan semua library ML (`transformers`, `torch`/`tensorflow`, `scikit-learn` versi spesifik jika ada) terinstal di environment `ujaran`. Perbarui `requirements.txt` jika ada tambahan.
*   [ ] **Implementasi Penuh Fungsi di `train_utils.py`:**
    *   [ ] **`train_model`**: Kembangkan script untuk tokenisasi data teks Bahasa Jawa menggunakan tokenizer IndoBERT. Implementasikan logika fine-tuning model IndoBERT dengan dataset pelatihan yang telah dilabeli (tambahkan lapisan klasifikasi, loop pelatihan dasar).
    *   [ ] **`save_model`**: Implementasikan penyimpanan model yang sudah di-fine-tune ke direktori `/models/`.
*   [ ] Latih model awal.

### Fase 4: Evaluasi Model Dasar
*   [ ] **Implementasi Penuh `evaluate_model` di `train_utils.py`**:
    *   Kembangkan script untuk melakukan prediksi menggunakan model yang sudah di-fine-tune pada set validasi.
    *   Implementasikan perhitungan metrik evaluasi (Akurasi, Presisi, Recall, F1-score, Confusion Matrix).
*   [ ] Lakukan evaluasi dan analisis hasil.

### Fase 5: Pengembangan Prototipe API Sederhana
*   [ ] Pilih framework API (FastAPI direkomendasikan dalam dokumen `tumpukan-teknologi.md`) dan setup struktur dasar.
*   [ ] Buat endpoint API `/detect` yang memuat model terlatih dan melakukan prediksi.
*   [ ] (Opsional MVP) Buat antarmuka web sangat sederhana untuk interaksi dengan API.

## 3. Saran "Baby-Step To-Do List" untuk Langkah Berikutnya:

Fokus pada **Fase 1: Implementasi Penuh Pengambilan Data dari Google Sheets.**

1.  **Manajemen Kredensial Google Sheets:**
    *   **Tugas Pengguna:** Menyediakan file kredensial (misalnya, `credentials.json`) dan instruksi bagaimana saya bisa mengaksesnya dengan aman (misalnya, path ke file yang di-mount di sandbox, atau variabel environment). **Jangan berikan isi kredensial secara langsung kepada saya.**
    *   **Tugas Pengguna:** Menyediakan Google Sheet ID, nama sheet, dan range data.
2.  **Implementasi Fungsi `load_data_from_google_sheets`:**
    *   **Saya (Jules):** Update fungsi di `memory-bank/data_utils.py` untuk menggunakan Google Sheets API client library (misalnya, `google-api-python-client`, `gspread` jika lebih mudah) untuk membaca data berdasarkan informasi yang diberikan pengguna.
    *   Fungsi ini harus mengembalikan Pandas DataFrame.
3.  **Pengujian Fungsi Pengambilan Data:**
    *   **Saya (Jules):** Modifikasi bagian `if __name__ == '__main__':` di `data_utils.py` untuk menguji fungsi `load_data_from_google_sheets` (setelah pengguna menyediakan detailnya).
    *   **Saya (Jules):** Jalankan skrip dari root direktori proyek untuk memastikan path kredensial (jika menggunakan file) dan output penyimpanan data (misalnya ke `data/raw/downloaded_gsheet.csv`) berfungsi.
4.  **Integrasi ke Notebook `refactored_notebook.ipynb`:**
    *   **Saya (Jules):** Tambahkan atau modifikasi sel di notebook untuk memanggil `load_data_from_google_sheets` sebagai alternatif atau pengganti `load_data_from_csv`.

---