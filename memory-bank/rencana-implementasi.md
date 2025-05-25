# Rencana Implementasi Awal (MVP) - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 0.1
**Tanggal:** 26 Mei 2025

Rencana ini berfokus pada pencapaian Minimum Viable Product (MVP) untuk sistem deteksi ujaran kebencian Bahasa Jawa.

## Tujuan MVP:

Membangun sistem dasar yang dapat menerima input teks Bahasa Jawa, mengklasifikasikannya sebagai ujaran kebencian (dengan tingkatan dasar) atau bukan, dan menyediakannya melalui API sederhana.

## Fase Implementasi MVP:

### Fase 0: Setup Proyek & Lingkungan (Mengikuti Panduan Vibe Coding)
* **Langkah 0.1:** Inisialisasi repositori Git & GitHub.
* **Langkah 0.2:** Buat struktur folder proyek dasar (misalnya, `/data`, `/notebooks`, `/src`, `/models`, `/tests`).
* **Langkah 0.3:** Setup virtual environment (Python) dan install library awal (`pandas`, `numpy`, `jupyter`).
    * **Validasi:** Berhasil import library di script Python.
* **Langkah 0.4:** Buat file `requirements.txt` awal.

### Fase 1: Pengumpulan & Preprocessing Data Awal
* **Langkah 1.1:** Identifikasi sumber data awal (minimal 2-3 platform online yang disebutkan dalam proposal seperti Twitter, Facebook, forum lokal). [cite: 56, 91]
    * **Validasi:** Daftar sumber data terdokumentasi.
* **Langkah 1.2:** Lakukan pengumpulan data manual atau dengan script sederhana untuk mendapatkan ~500-1000 sampel teks Bahasa Jawa mentah. [cite: 92]
    * **Validasi:** File dataset mentah (misalnya, `.csv`, `.txt`) tersedia.
* **Langkah 1.3:** Kembangkan script dasar untuk preprocessing data:
    * Case folding (mengubah ke huruf kecil).
    * Penghapusan karakter tidak relevan (URL, emoji, tanda baca berlebih). [cite: 94]
    * Normalisasi teks dasar (misalnya, penanganan slang umum jika diketahui, perbaikan typo sederhana). [cite: 94]
    * **Validasi:** Script dapat memproses file data mentah dan menghasilkan data bersih.
* **Langkah 1.4:** Implementasikan fungsi filtering dasar untuk konten duplikat. [cite: 57, 94]
    * **Validasi:** Jumlah data berkurang setelah filtering duplikat.

### Fase 2: Pelabelan Data Awal & Persiapan Dataset
* **Langkah 2.1:** Definisikan pedoman pelabelan awal yang jelas untuk kategori: "Ujaran Kebencian - Ringan", "Ujaran Kebencian - Sedang", "Ujaran Kebencian - Berat", "Bukan Ujaran Kebencian". [cite: 7, 57, 96, 97] Konsultasikan dengan ahli bahasa/budaya Jawa jika memungkinkan pada tahap ini. [cite: 7, 55, 85]
    * **Validasi:** Dokumen pedoman pelabelan tersedia.
* **Langkah 2.2:** Lakukan pelabelan manual pada sampel data yang sudah diproses (~200-500 sampel awal). [cite: 95]
    * **Validasi:** File dataset berlabel tersedia (misalnya, teks, label).
* **Langkah 2.3:** Bagi dataset berlabel menjadi set pelatihan dan validasi (misalnya, 80% train, 20% val).
    * **Validasi:** Dua file dataset terpisah (train dan val) tersedia.

### Fase 3: Pengembangan Model Dasar (Fine-tuning IndoBERT)
* **Langkah 3.1:** Setup environment untuk Hugging Face Transformers dan PyTorch/TensorFlow. Install library yang dibutuhkan.
    * **Validasi:** Berhasil import library dan load model IndoBERT pre-trained. [cite: 58, 100]
* **Langkah 3.2:** Kembangkan script untuk tokenisasi data teks Bahasa Jawa menggunakan tokenizer IndoBERT.
    * **Validasi:** Teks dapat ditokenisasi tanpa error.
* **Langkah 3.3:** Kembangkan script untuk fine-tuning model IndoBERT dengan dataset pelatihan yang telah dilabeli:
    * Tambahkan lapisan klasifikasi di atas IndoBERT untuk 4 kelas output. [cite: 59, 103]
    * Implementasikan loop pelatihan dasar.
    * **Validasi:** Model dapat dilatih untuk beberapa epoch tanpa error. Loss pelatihan menurun.
* **Langkah 3.4:** Simpan model yang sudah di-fine-tune.
    * **Validasi:** File model tersimpan.

### Fase 4: Evaluasi Model Dasar
* **Langkah 4.1:** Kembangkan script untuk melakukan prediksi menggunakan model yang sudah di-fine-tune pada set validasi.
    * **Validasi:** Prediksi dapat dihasilkan untuk data validasi.
* **Langkah 4.2:** Implementasikan perhitungan metrik evaluasi dasar: Akurasi. [cite: 54, 106]
    * **Validasi:** Nilai akurasi tercetak.
* **Langkah 4.3 (Opsional MVP Awal):** Implementasikan confusion matrix. [cite: 61, 107]
    * **Validasi:** Confusion matrix dapat ditampilkan/disimpan.

### Fase 5: Pengembangan Prototipe API Sederhana
* **Langkah 5.1:** Pilih framework API (Flask atau FastAPI). [cite: 108] Setup struktur dasar aplikasi API.
    * **Validasi:** Endpoint "hello world" API berfungsi.
* **Langkah 5.2:** Buat endpoint API (misalnya, `/detect`) yang:
    * Menerima input teks Bahasa Jawa.
    * Memuat model fine-tuned yang sudah disimpan.
    * Melakukan preprocessing pada input teks.
    * Melakukan tokenisasi.
    * Melakukan prediksi menggunakan model.
    * Mengembalikan hasil klasifikasi (misalnya, dalam format JSON).
    * **Validasi:** API dapat menerima teks, memprosesnya dengan model, dan mengembalikan hasil prediksi.
* **Langkah 5.3 (Opsional MVP Awal):** Buat antarmuka web sangat sederhana dengan HTML form untuk mengirim teks ke API dan menampilkan hasilnya. [cite: 109]
    * **Validasi:** Antarmuka dapat berinteraksi dengan API.

### Iterasi Berikutnya (Pasca MVP):
* Pengumpulan data lebih banyak dan pelabelan lanjutan. [cite: 92]
* Optimalisasi hyperparameter model.
* Implementasi metrik evaluasi yang lebih lengkap (presisi, recall, F1-score). [cite: 54, 106]
* Analisis kualitatif kesalahan. [cite: 54, 107]
* Uji coba prototipe dengan pengguna. [cite: 7, 62, 110]

--- 