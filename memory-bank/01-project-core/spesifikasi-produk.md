# Dokumen Kebutuhan Produk (PRD): Sistem Deteksi Ujaran Kebencian Bahasa Jawa

## 1. Tinjauan Produk

**Visi Produk:** Mengembangkan sistem yang mampu mendeteksi dan mengklasifikasikan ujaran kebencian dalam teks berbahasa Jawa secara akurat untuk mendukung lingkungan digital yang lebih aman dan positif.

**Target Pengguna:** Platform media sosial, forum online, aplikasi perpesanan, dan peneliti keamanan siber.

**Tujuan Bisnis:** Menyediakan solusi terdepan untuk moderasi konten berbahasa Jawa, mengurangi penyebaran ujaran kebencian, dan membangun reputasi sebagai pemimpin dalam teknologi AI untuk bahasa daerah.

**Metrik Kesuksesan:**
- Akurasi klasifikasi model > 90%
- Pengurangan laporan ujaran kebencian manual sebesar 50% pada platform mitra
- Waktu respons API < 200ms
- Dataset berlabel minimal 1000 sampel untuk MVP

## 2. Persona Pengguna

### Persona 1: "Moderator Konten Platform"
- **Demografi:** 25-40 tahun, bekerja untuk perusahaan media sosial, terbiasa dengan alat moderasi
- **Tujuan:** Memfilter konten berbahaya dengan cepat dan akurat untuk menjaga keamanan komunitas
- **Masalah (Pain Points):** Kesulitan memahami konteks budaya dan linguistik Bahasa Jawa, volume konten yang tinggi, kelelahan akibat paparan konten negatif
- **Perjalanan Pengguna:** Menggunakan dasbor untuk meninjau konten yang ditandai oleh sistem, memberikan umpan balik pada klasifikasi, dan melihat analitik tren

### Persona 2: "Peneliti NLP dan Linguistik"
- **Demografi:** 25-45 tahun, akademisi atau peneliti di bidang NLP/linguistik, familiar dengan tools penelitian
- **Tujuan:** Menggunakan dataset dan model untuk penelitian lebih lanjut tentang Bahasa Jawa dan deteksi konten negatif
- **Masalah (Pain Points):** Keterbatasan dataset berlabel untuk bahasa daerah, kesulitan akses ke tools yang spesifik untuk Bahasa Jawa
- **Perjalanan Pengguna:** Mengakses API untuk eksperimen, mengunduh dataset untuk penelitian, menggunakan model untuk analisis linguistik

## 3. Kebutuhan Fitur

| Fitur | Deskripsi | User Stories | Prioritas | Kriteria Penerimaan | Ketergantungan |
|---|---|---|---|---|---|
| **Klasifikasi Teks** | Menganalisis teks input dan mengklasifikasikannya ke dalam 4 tingkat keparahan ujaran kebencian. | Sebagai moderator, saya ingin sistem secara otomatis memberi label pada sebuah komentar agar saya bisa fokus pada kasus yang paling parah. | Wajib | API menerima teks dan mengembalikan label prediksi dengan skor kepercayaan. | Model terlatih |
| **API Endpoint** | Menyediakan antarmuka RESTful untuk integrasi dengan sistem lain. | Sebagai pengembang, saya ingin dapat mengirim permintaan ke API dan menerima respons JSON yang terstruktur. | Wajib | Endpoint `/predict` tersedia, terdokumentasi, dan aman. | Infrastruktur Server |
| **Dasbor Analitik** | Menampilkan visualisasi data tentang tren ujaran kebencian yang terdeteksi. | Sebagai manajer komunitas, saya ingin melihat laporan mingguan tentang jenis ujaran kebencian yang paling sering muncul. | Sebaiknya | Dasbor menampilkan grafik jumlah deteksi per hari dan distribusi per kategori. | Database Log |

## 4. Alur Pengguna (User Flows)

### Alur 1: Deteksi via API
1. Sistem eksternal (misal: aplikasi media sosial) mengirim permintaan POST ke `/api/predict` dengan konten teks
2. Sistem memproses teks melalui model machine learning
3. Sistem mengembalikan respons JSON berisi label klasifikasi (misal: "Ujaran Kebencian Berat") dan skor kepercayaan
    - Kondisi error: Jika teks input kosong atau format tidak valid, kembalikan error 400
    - Kondisi error: Jika API key tidak valid, kembalikan error 401

### Alur 2: Penggunaan Web Interface (MVP)
1. Pengguna mengakses halaman web sederhana
2. Pengguna memasukkan teks Bahasa Jawa di form input
3. Pengguna menekan tombol "Analisis"
4. Sistem menampilkan hasil klasifikasi dan tingkat kepercayaan
    - Kondisi error: Jika teks kosong, tampilkan pesan error
    - Kondisi error: Jika server tidak merespons, tampilkan pesan error koneksi

### Alur 3: Akses Dataset untuk Penelitian
1. Peneliti mengakses dokumentasi API
2. Peneliti mendaftar untuk mendapatkan akses dataset
3. Peneliti mengunduh dataset berlabel dalam format CSV
4. Peneliti menggunakan dataset untuk penelitian lanjutan

## 5. Kebutuhan Non-Fungsional

### Performa
- **Waktu Respons:** Rata-rata < 200ms per permintaan API.
- **Pengguna Bersamaan:** Mampu menangani 100 permintaan per detik.

### Keamanan
- **Otentikasi:** Akses API menggunakan kunci API (API Key).
- **Perlindungan Data:** Tidak menyimpan data teks yang dikirim ke API setelah diproses.

### Kompatibilitas
- **API:** RESTful, dapat diakses oleh semua bahasa pemrograman modern.

## 6. Spesifikasi Teknis (Gambaran Umum)

### Backend
- **Tumpukan Teknologi:** Python, FastAPI/Flask, Hugging Face Transformers, PyTorch/TensorFlow.
- **Database:** Tidak ada untuk logika inti, mungkin PostgreSQL untuk logging jika diperlukan.

### Infrastruktur
- **Hosting:** Solusi berbasis kontainer (Docker) di platform cloud (misal: AWS, GCP).
- **CI/CD:** Otomatisasi testing dan deployment menggunakan GitHub Actions.

## 7. Rencana Rilis

### MVP (v1.0)
- **Fitur:** API klasifikasi teks untuk 4 kategori ujaran kebencian.
- **Target Waktu:** 3 bulan dari sekarang.
- **Kriteria Sukses MVP:** API berfungsi sesuai spesifikasi dan mencapai akurasi > 85% pada data validasi.

## 8. Pertanyaan Terbuka & Asumsi

### Pertanyaan Terbuka
- **Pertanyaan 1:** Bagaimana strategi untuk menangani dialek atau variasi bahasa Jawa yang berbeda?
- **Pertanyaan 2:** Seberapa besar dataset yang dibutuhkan untuk mencapai akurasi target 90%?
- **Pertanyaan 3:** Bagaimana cara mengintegrasikan kearifan lokal secara sistematis dalam proses pelabelan?
- **Pertanyaan 4:** Apakah perlu membuat model terpisah untuk tingkatan bahasa (ngoko vs krama)?

### Asumsi
- **Asumsi 1:** Dataset `raw-dataset.csv` cukup representatif untuk melatih model yang general
- **Asumsi 2:** IndoBERT sebagai base model akan memberikan performa yang baik untuk Bahasa Jawa
- **Asumsi 3:** Tim pelabel memiliki pemahaman yang cukup tentang konteks budaya Jawa
- **Asumsi 4:** Platform target memiliki infrastruktur untuk integrasi API

## 9. Lampiran

### Wawasan dari Percakapan dengan AI
- **Percakapan 1:** [Tanggal akan diisi] - Validasi konsep dan arsitektur sistem
- **Edge Case dari AI:** Penanganan teks campuran (Jawa-Indonesia), deteksi sarkasme, konteks budaya spesifik
- **Saran Perbaikan dari AI:** Implementasi sistem feedback loop untuk continuous learning

### Glosarium
- **BERT:** Bidirectional Encoder Representations from Transformers, model bahasa yang digunakan
- **API:** Application Programming Interface, antarmuka untuk interaksi perangkat lunak
- **IndoBERT:** Versi BERT yang dilatih khusus untuk Bahasa Indonesia
- **Fine-tuning:** Proses penyesuaian model pre-trained untuk tugas spesifik
- **Ngoko:** Tingkatan bahasa Jawa yang informal
- **Krama:** Tingkatan bahasa Jawa yang formal/halus
- **Pasemon:** Sindiran halus dalam budaya Jawa