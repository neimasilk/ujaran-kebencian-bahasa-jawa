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
1. Clone repositori Git ke lokal
2. Buat branch development
3. Periksa struktur folder dasar
4. Aktifkan atau buat conda environment `ujaran` (lihat `memory-bank/environment-setup.md`). Pastikan Python versi 3.11+ (sesuai `requirements.txt`)
5. Install library Python yang dibutuhkan menggunakan `pip install -r requirements.txt`. Pastikan library untuk Google Sheets API terinstall: `google-api-python-client google-auth-httplib2 google-auth-oauthlib`
6. Pastikan file `requirements.txt` sudah terbaru dan mencakup library Google Sheets API
7. Buat script test sederhana untuk memvalidasi environment

--- 