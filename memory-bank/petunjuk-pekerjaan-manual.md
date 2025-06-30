# Petunjuk Pekerjaan Manual - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 1.0
**Tanggal:** 29 Desember 2024
**Mengikuti:** Vibe Coding Guide v1.4

## 1. Pendahuluan

Dokumen ini memberikan panduan terstruktur untuk pekerjaan manual yang esensial dalam proyek "Sistem Deteksi Ujaran Kebencian Bahasa Jawa Menggunakan BERT". Proyek ini dikembangkan menggunakan metodologi Vibe Coding v1.4 dengan tim hibrida (manusia + AI).

Pekerjaan manual terutama berkaitan dengan aspek-aspek yang memerlukan interpretasi manusia, pemahaman konteks budaya mendalam, dan penilaian subjektif yang sulit diotomatisasi, khususnya dalam pembuatan dan pelabelan dataset.

**Referensi Terkait:**
- Spesifikasi Produk: [`spesifikasi-produk.md`](spesifikasi-produk.md)
- Tim Manifest: [`../vibe-guide/team-manifest.md`](../vibe-guide/team-manifest.md)
- Panduan Utama: [`../vibe-guide/VIBE_CODING_GUIDE.md`](../vibe-guide/VIBE_CODING_GUIDE.md)

Kualitas pekerjaan manual ini akan berdampak langsung pada kualitas dataset dan performa model machine learning yang dikembangkan.

## 2. Panduan Pelabelan Data Ujaran Kebencian Bahasa Jawa

Proses pelabelan data adalah salah satu tahapan manual paling kritis dalam proyek ini. Tujuannya adalah untuk menghasilkan dataset Bahasa Jawa berlabel berkualitas tinggi yang akan digunakan untuk melatih dan mengevaluasi model deteksi ujaran kebencian.

### 2.1. Tujuan Pelabelan

* Menyediakan "ground truth" bagi model machine learning untuk belajar membedakan berbagai kategori ujaran kebencian dan teks netral/bukan ujaran kebencian.
* Memastikan dataset mencerminkan nuansa linguistik dan budaya Bahasa Jawa yang relevan dengan ujaran kebencian.
* Menghasilkan dataset yang dapat digunakan untuk penelitian selanjutnya.

### 2.2. Tim Pelabel

* Proses pelabelan idealnya melibatkan individu yang memiliki pemahaman baik tentang Bahasa Jawa (termasuk dialek dan tingkatan bahasa) dan konteks budayanya.
* Sangat dianjurkan untuk melibatkan atau setidaknya melakukan konsultasi rutin dengan ahli bahasa Jawa dan budayawan Jawa, sebagaimana direncanakan dalam proposal penelitian.

### 2.3. Definisi Kategori Label

Setiap sampel teks akan diklasifikasikan ke dalam salah satu kategori berikut. Konsistensi dalam pemahaman dan penerapan definisi ini sangat penting.

1.  **Bukan Ujaran Kebencian:**
    * Teks yang tidak mengandung unsur hinaan, provokasi, hasutan, ancaman, atau ekspresi negatif yang merendahkan berdasarkan SARA (Suku, Agama, Ras, Antargolongan), gender, orientasi seksual, kondisi fisik, atau atribut pribadi/kelompok lainnya.
    * Termasuk di dalamnya adalah teks netral, positif, kritik membangun (tanpa merendahkan), diskusi, berita, dll.

2.  **Ujaran Kebencian - Ringan:**
    * Teks yang mengandung sindiran halus, ejekan terselubung, atau metafora budaya yang menyiratkan ketidaksukaan atau kerendahan terhadap target, namun tidak secara eksplisit kasar atau mengancam.
    * Mungkin memerlukan pemahaman konteks budaya Jawa untuk mengidentifikasinya.
    * Contoh: Penggunaan pasemon atau peribahasa Jawa yang bertujuan menyindir secara tidak langsung.

3.  **Ujaran Kebencian - Sedang:**
    * Teks yang mengandung hinaan langsung, cercaan, atau penggunaan bahasa kasar (misalnya, dalam konteks ngoko yang tidak pantas) yang jelas ditujukan untuk merendahkan atau melecehkan target.
    * Intensitasnya lebih tinggi dari kategori "Ringan" dan lebih eksplisit.
    * Tidak sampai pada ancaman kekerasan fisik langsung atau diskriminasi sistematis.

4.  **Ujaran Kebencian - Berat:**
    * Teks yang mengandung ancaman kekerasan fisik, hasutan untuk melakukan kekerasan, dehumanisasi, pernyataan yang mendukung diskriminasi sistematis, atau penghinaan ekstrem terkait SARA.
    * Memiliki potensi dampak paling merusak bagi individu atau kelompok target.

### 2.4. Proses Pelabelan (Langkah-langkah Manual)

#### Strategi Cost-Efficient dengan DeepSeek API

**Pembagian Data:**
- **Data Positif** (label: "positive") → Langsung dilabeli sebagai "Bukan Ujaran Kebencian" (tidak perlu API)
- **Data Negatif** (label: "negative") → Dilabeli detail menggunakan DeepSeek API

1.  **Persiapan Environment:**
    * Setup file `.env` dengan konfigurasi DeepSeek API
    * Install dependencies: `pip install -r requirements.txt`
    * Test koneksi API dengan mode mock
    * Siapkan file `raw-dataset.csv`

2.  **Labeling Otomatis:**
    ```bash
    # Mode testing (mock API)
    python src/data_collection/deepseek_labeling_pipeline.py --mock --sample 50
    
    # Mode production (real API)
    python src/data_collection/deepseek_labeling_pipeline.py --input raw-dataset.csv --output labeled-dataset.csv
    ```

3.  **Analisis Teks dan Konteks (untuk validasi manual):**
    * Baca setiap sampel teks Bahasa Jawa secara menyeluruh.
    * Pertimbangkan konteks linguistik:
        * **Tingkatan Bahasa:** Apakah teks menggunakan Ngoko, Krama Madya, atau Krama Inggil? Penggunaan tingkatan bahasa yang tidak sesuai dapat menjadi indikator.
        * **Dialek:** Jika memungkinkan, identifikasi variasi dialek (misalnya, Jawa Timuran, Jawa Tengahan, Yogyakarta) karena beberapa istilah mungkin memiliki konotasi berbeda.
        * **Metafora & Ungkapan Lokal (Pasemon):** Perhatikan penggunaan metafora, peribahasa, atau ungkapan khas Jawa yang mungkin mengandung makna tersirat terkait ujaran kebencian.
    * Pertimbangkan konteks sosial dan budaya yang lebih luas jika informasi tersebut tersedia atau dapat diinferensikan secara wajar.

4.  **Validasi Manual:**
    * **Review hasil API** untuk data negatif
    * **Validasi confidence score** rendah (<0.6)
    * **Spot check** 10% hasil secara random
    * **Koreksi manual** jika diperlukan

5.  **Quality Assurance:**
    * Analisis distribusi label hasil
    * Review consistency antar kategori
    * Dokumentasi edge cases
    * Update guidelines jika diperlukan

6.  **Verifikasi dan Iterasi (Direkomendasikan):**
    * Setelah sejumlah data dilabeli, sebagian sampel dapat diverifikasi oleh pelabel lain (proses *cross-validation* atau *inter-annotator agreement check*).
    * Hasil verifikasi digunakan untuk mengidentifikasi ketidakkonsistenan dan memperbaiki pemahaman atau pedoman pelabelan jika perlu. Proses pelabelan mungkin bersifat iteratif.

### 2.5. Hal-hal yang Perlu Diperhatikan Saat Pelabelan

* **Objektivitas dan Netralitas:** Upayakan untuk melabeli berdasarkan definisi dan pedoman yang ada, bukan berdasarkan opini atau bias pribadi.
* **Konteks adalah Kunci:** Makna sebuah ujaran sangat bergantung pada konteksnya. Jangan melabeli kata atau frasa secara terisolasi.
* **Konsistensi:** Usahakan untuk menerapkan kriteria pelabelan secara konsisten di seluruh dataset.
* **Fokus pada Maksud (Intent):** Meskipun sulit, cobalah untuk mempertimbangkan maksud di balik ujaran, terutama untuk membedakan kritik dari ujaran kebencian.
* **Dokumentasi Kasus Sulit:** Catat contoh-contoh teks yang sulit dilabeli beserta alasannya. Ini akan membantu dalam diskusi tim dan penyempurnaan pedoman.
* **Keterbatasan Data Awal:** Ingat bahwa dataset awal mungkin belum mencakup semua variasi ujaran kebencian. Tetap terbuka untuk menemukan pola-pola baru.

## 3. Target dan Metrik Kualitas

### 3.1. Target Kuantitatif
- **Target Minimum:** 200 sampel berlabel untuk MVP
- **Target Optimal:** 500-1000 sampel berlabel untuk model yang lebih robust
- **Distribusi Label:** Usahakan distribusi yang relatif seimbang antar kategori (tidak ada kategori yang kurang dari 5% dari total)

### 3.2. Metrik Kualitas
- **Inter-annotator Agreement:** Target >80% untuk konsistensi antar pelabel
- **Consistency Rate:** >90% konsistensi dalam pelabelan ulang sampel yang sama
- **Coverage:** Semua kategori harus terwakili dalam dataset final
- **Documentation:** Semua edge cases dan keputusan sulit harus terdokumentasi
- **API Confidence:** >70% average confidence score dari DeepSeek API
- **Cost Efficiency:** >50% penghematan biaya API melalui strategi positif/negatif

### 3.3. Metrik DeepSeek API
- **Success Rate:** >95% API calls berhasil
- **Response Time:** <2 detik average per sample
- **Cost Optimization:** Maksimal 50% data menggunakan API (sisanya rule-based)
- **Quality Validation:** Manual review untuk confidence score <0.6

## 4. Sumber Data

Fokus utama pekerjaan manual adalah pelabelan data yang sudah ada di `raw-dataset.csv`. Proses pengumpulan data baru tidak menjadi bagian dari lingkup pekerjaan manual saat ini, karena dataset awal dianggap sudah cukup untuk fase MVP. Semua upaya akan difokuskan pada kualitas pelabelan.