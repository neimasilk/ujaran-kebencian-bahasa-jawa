# Dokumen Desain Produk: Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 0.1
**Tanggal:** 26 Mei 2025

## 1. Pendahuluan dan Visi

**Visi Produk:**
Menciptakan sebuah sistem yang mampu secara cerdas dan akurat mendeteksi ujaran kebencian dalam teks berbahasa Jawa, dengan mempertimbangkan konteks linguistik dan kearifan lokal. [cite: 20] Sistem ini diharapkan dapat berkontribusi dalam menjaga ruang digital yang lebih sehat bagi penutur Bahasa Jawa. [cite: 50, 51]

**Masalah yang Diselesaikan:**
* Meningkatnya ujaran kebencian berbahasa Jawa di platform online. [cite: 18, 32, 38]
* Kurangnya alat deteksi ujaran kebencian yang efektif untuk Bahasa Jawa, yang memiliki karakteristik unik (misalnya, tingkatan bahasa ngoko dan krama, variasi dialek). [cite: 39, 40, 42]
* Potensi dampak negatif ujaran kebencian yang tidak termoderasi, seperti konflik sosial dan degradasi makna bahasa. [cite: 37, 50, 52]
* Keterbatasan dataset publik berlabel untuk ujaran kebencian Bahasa Jawa. [cite: 42]

## 2. Target Pengguna

1.  **Platform Digital Lokal/Nasional:** Untuk membantu memfilter konten berbahaya berbahasa Jawa secara proaktif. [cite: 51]
2.  **Peneliti NLP dan Linguistik:** Sebagai alat bantu dan penyedia dataset untuk penelitian lebih lanjut mengenai Bahasa Jawa dan deteksi konten negatif. [cite: 26, 122]
3.  **Komunitas Pengguna Bahasa Jawa:** Sebagai alat untuk meningkatkan kesadaran dan pemahaman tentang ujaran kebencian dalam konteks budaya mereka. [cite: 7, 62]
4.  **Regulator/Pemerintah (misalnya Kominfo):** Sebagai dasar untuk pembuatan rekomendasi kebijakan terkait konten berbahasa daerah. [cite: 69]

## 3. Fitur Utama (MVP dan Pengembangan Lanjutan)

### MVP (Minimum Viable Product):

1.  **Input Teks Bahasa Jawa:** Pengguna dapat memasukkan potongan teks Bahasa Jawa.
2.  **Deteksi Ujaran Kebencian:** Sistem mengklasifikasikan teks sebagai:
    * Ujaran Kebencian
    * Bukan Ujaran Kebencian
3.  **Klasifikasi Tingkat Keparahan (Dasar):** Jika terdeteksi sebagai ujaran kebencian, memberikan klasifikasi dasar (misalnya, ringan, sedang, berat). [cite: 7, 57]
4.  **API Sederhana:** Endpoint API untuk menerima teks dan mengembalikan hasil deteksi. [cite: 7, 62, 108]

### Fitur Pengembangan Lanjutan:

1.  **Integrasi Kearifan Lokal yang Lebih Dalam:**
    * Kemampuan untuk mengidentifikasi metafora budaya spesifik atau sindiran halus (pasemon) yang relevan dengan ujaran kebencian. [cite: 49, 55, 68]
    * Penyesuaian model berdasarkan tingkatan bahasa (ngoko, krama). [cite: 49, 68]
2.  **Penjelasan Hasil Deteksi:** Memberikan justifikasi singkat mengapa sebuah teks diklasifikasikan sebagai ujaran kebencian (misalnya, kata kunci pemicu, konteks).
3.  **Dasbor Analitik (untuk peneliti/platform):** Visualisasi data ujaran kebencian yang terdeteksi.
4.  **Plugin Browser/Integrasi Platform:** Memudahkan penggunaan di berbagai platform.
5.  **Dukungan Dialek:** Peningkatan akurasi untuk berbagai dialek Bahasa Jawa. [cite: 40, 56]

## 4. Arsitektur Sistem (Gambaran Umum)

(Detail akan ada di `architecture.md`)
* **Modul Pengumpulan & Preprocessing Data:** Mengumpulkan data teks Bahasa Jawa dari berbagai sumber, membersihkan, dan melakukan normalisasi. [cite: 24, 91, 93, 94]
* **Modul Pelabelan Data:** Proses pelabelan manual oleh ahli bahasa dan budaya Jawa dengan kategori bertingkat. [cite: 7, 57, 95, 96, 97]
* **Modul Pelatihan Model:**
    * Menggunakan arsitektur BERT (fine-tuning dari IndoBERT). [cite: 58, 100]
    * Penambahan lapisan klasifikasi. [cite: 59, 103]
* **Modul Evaluasi Model:** Mengukur performa model menggunakan metrik standar (akurasi, presisi, recall, F1-score) dan analisis kualitatif. [cite: 54, 60, 106, 107]
* **Modul API:** Menyediakan antarmuka untuk interaksi dengan model yang sudah dilatih. [cite: 7, 62, 108]
* **Antarmuka Pengguna (Prototipe Web):** Aplikasi web sederhana untuk demonstrasi input teks dan output klasifikasi. [cite: 109]

## 5. Kriteria Keberhasilan (MVP)

* Model fine-tuned BERT mencapai akurasi minimal (misalnya, ≥75-80% pada data uji awal, target akhir ≥90% [cite: 119]).
* Dataset awal (misalnya, 1000-2000 sampel) berhasil dikumpulkan, diproses, dan dilabeli. [cite: 92]
* Prototipe API berfungsi dan dapat mengklasifikasikan input teks Bahasa Jawa.
* Mampu membedakan setidaknya antara "ujaran kebencian" dan "bukan ujaran kebencian".

## 6. Rencana Rilis (Gambaran Umum)

* **Fase 1 (Penelitian & Pengembangan Awal):**
    * Pengumpulan dan pelabelan dataset awal.
    * Pengembangan dan pelatihan model BERT (MVP).
    * Pengembangan prototipe API.
* **Fase 2 (Evaluasi & Iterasi):**
    * Evaluasi kuantitatif dan kualitatif model. [cite: 54, 60, 106, 107]
    * Uji coba prototipe dengan komunitas pengguna. [cite: 7, 62, 110]
    * Iterasi model berdasarkan feedback dan hasil evaluasi.
* **Fase 3 (Publikasi & Diseminasi):**
    * Publikasi dataset dan hasil penelitian (artikel ilmiah). [cite: 26, 122, 124]
    * Pengembangan panduan untuk platform. [cite: 123]

## 7. Pertimbangan Masa Depan

* Optimalisasi model untuk kecepatan dan efisiensi *real-time*. [cite: 75]
* Ekspansi ke bahasa daerah lain di Indonesia. [cite: 26]
* Pengembangan kerangka etik AI yang memadukan norma budaya Jawa. [cite: 75]

--- 