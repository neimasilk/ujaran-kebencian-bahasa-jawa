# Status, To-Do List, dan Saran

**Update Terakhir:** 26 Mei 2025, 14:30 WIB

## 1. Status Proyek Saat Ini:

* Tahap 0 (Pengaturan Awal Proyek) selesai.
* Dokumen perencanaan awal (`dokumen-desain-produk.md`, `tumpukan-teknologi.md`, `rencana-implementasi.md`, `architecture.md`) telah dibuat (versi 0.1).
* Repositori Git telah dibuat dan di-clone ke lokal.
* Struktur folder proyek dasar telah dibuat.
* Virtual environment Python telah disetup.
* Proyek siap untuk memasuki Tahap 1: Pengumpulan & Preprocessing Data Awal.

## 2. To-Do List (Masa Depan - Berdasarkan `rencana-implementasi.md` MVP):

### Fase 0: Setup Proyek & Lingkungan (Lanjutan)
* Setup conda environment dan pastikan semua dependensi terinstall dengan benar
* Validasi environment dengan menjalankan script test sederhana

### Fase 1: Pengumpulan & Preprocessing Data Awal
* Persiapkan akses ke dataset pribadi di Google Sheets melalui Google Sheets API
* Kembangkan script untuk mengambil data (~500-1000 sampel) dari Google Sheets API
* Implementasi preprocessing dasar untuk data dari Google Sheets

### Fase 2: Pelabelan Data Awal & Persiapan Dataset
* [ ] Definisikan pedoman pelabelan awal.
* [ ] Lakukan pelabelan manual pada sampel data awal (~200-500 sampel).
* [ ] Bagi dataset berlabel menjadi set pelatihan dan validasi.

### Fase 3: Pengembangan Model Dasar (Fine-tuning IndoBERT)
* [ ] Setup environment untuk Hugging Face Transformers dan PyTorch/TensorFlow.
* [ ] Kembangkan script untuk tokenisasi data.
* [ ] Kembangkan script untuk fine-tuning IndoBERT.
* [ ] Simpan model yang sudah di-fine-tune.

### Fase 4: Evaluasi Model Dasar
* [ ] Kembangkan script untuk prediksi pada set validasi.
* [ ] Implementasikan perhitungan metrik evaluasi dasar (Akurasi).
* [ ] (Opsional MVP) Implementasikan confusion matrix.

### Fase 5: Pengembangan Prototipe API Sederhana
* [ ] Pilih framework API dan setup struktur dasar.
* [ ] Buat endpoint API `/detect`.
* [ ] (Opsional MVP) Buat antarmuka web sangat sederhana.

## 3. Saran "Baby-Step To-Do List" untuk Langkah Berikutnya:

### Baby Step 1: Inisialisasi Proyek & Lingkungan Dasar
1. Clone repositori Git ke lokal.
2. Buat branch `development` dari `main` (atau branch default Anda).
3. Periksa dan pastikan struktur folder dasar proyek sudah sesuai dengan yang direncanakan di `readme.md`.
4. Aktifkan atau buat conda environment `ujaran` sesuai panduan di `memory-bank/environment-setup.md`. Pastikan versi Python (misalnya 3.11.x) sesuai.
5. **Persiapkan `requirements.txt`**:
   * File `requirements.txt` yang ada terlihat seperti ekspor lengkap dari sebuah environment conda. Untuk langkah awal, pastikan library krusial untuk Fase 1 (Pengumpulan Data dari Google Sheets) yaitu `google-api-python-client`, `google-auth-httplib2`, dan `google-auth-oauthlib` tercantum di dalamnya.
   * Jika belum ada, tambahkan library tersebut ke file `requirements.txt` yang ada. Anda mungkin juga ingin mempertimbangkan untuk membuat file `requirements.txt` yang lebih minimal dan terkurasi untuk proyek ini di masa mendatang.
6. Setelah memastikan `requirements.txt` mencakup semua paket yang diperlukan untuk tahap awal (terutama Google API Client Libraries), install semua library Python dari environment `ujaran` yang aktif menggunakan:
   ```bash
   pip install -r requirements.txt
   ```
7. Buat script Python sederhana (misalnya, `check_env.py` di root proyek atau di folder `tests`) yang mengimpor library-library utama yang baru diinstal (contoh: `import pandas`, `from googleapiclient.discovery import build`). Jalankan script ini untuk memvalidasi bahwa environment dan instalasi library berhasil tanpa error.

---