# Arsitektur Sistem - Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 0.1
**Tanggal:** 26 Mei 2025

Dokumen ini menjelaskan arsitektur yang direncanakan untuk sistem deteksi ujaran kebencian Bahasa Jawa. Arsitektur ini bersifat modular untuk memungkinkan pengembangan dan iterasi yang lebih mudah.

## Komponen Utama:

```

+-------------------------+      +--------------------------+      +---------------------+
|   Pengumpulan Data      |----->|  Preprocessing &         |----->|  Pelabelan Data     |
| (Online Platforms, etc.)|      |  Filtering Data          |      | (Ahli Bahasa/Budaya)|
+-------------------------+      +--------------------------+      +---------------------+
|                                  |
| (Data Bersih)                    | (Dataset Berlabel)
|                                  |
v                                  v
+---------------------------------------------------------------------------------------------+
|                                   Modul Machine Learning                                    |
|                                                                                             |
|  +--------------------------+      +--------------------------+      +--------------------+ |
|  |  Dataset Loader &        |----->|  Model BERT (IndoBERT    |----->|  Evaluasi Model    | |
|  |  Tokenizer               |      |  Fine-tuning &           |      | (Metrik, Analisis) | |
|  |                          |      |  Lapisan Klasifikasi)    |      +--------------------+ |
|  +--------------------------+      +--------------------------+                             |
|                                           ^         | (Model Terlatih)                       |
|                                           |_________| (Iterasi)                              |
+---------------------------------------------------------------------------------------------+
|
| (Model Terlatih)
v
+---------------------------------------------------------------------------------------------+
|                                       Prototipe Aplikasi                                    |
|                                                                                             |
|  +--------------------------+      +--------------------------+      +--------------------+ |
|  |  API Server              |<---->|  Logika Aplikasi         |<---->|  Antarmuka Pengguna| |
|  |  (Flask/FastAPI)         |      |  (Preprocessing Input,   |      |  (Web - Sederhana) | |
|  |                          |      |   Prediksi Model)        |      +--------------------+ |
|  +--------------------------+      +--------------------------+                             |
|                                                                                             |
+---------------------------------------------------------------------------------------------+

## Deskripsi Komponen:

1.  **Pengumpulan Data:**
    * **Tanggung Jawab:** Mengakses dan mengambil dataset teks berbahasa Jawa yang sudah ada dari Google Sheets menggunakan Google Sheets API.
    * **Teknologi:** Skrip Python, Google Sheets API, library klien Google untuk Python, Pandas.
    * **Input:** Akses ke Google Sheets yang berisi dataset.
    * **Output:** Dataset mentah dalam format yang dapat diproses (CSV/DataFrame).

2.  **Preprocessing & Filtering Data:**
    * **Tanggung Jawab:** Membersihkan dan memformat data mentah.
    * **Teknologi:** Python, Pandas, NLTK/Sastrawi.
    * **Input:** Dataset mentah dari Google Sheets.
    * **Output:** Dataset yang sudah dibersihkan dan diformat.

3.  **Pelabelan Data:** [cite: 7, 55, 57, 85, 95, 96, 97]
    * **Tanggung Jawab:** Melakukan pelabelan data secara manual dengan melibatkan ahli bahasa Jawa dan budayawan. Kategori label: ringan, sedang, berat, bukan ujaran kebencian. Konteks budaya dan tingkatan bahasa menjadi pertimbangan.
    * **Output:** Dataset teks berlabel yang siap untuk pelatihan model. Pedoman pelabelan.

4.  **Modul Machine Learning:**
    * **Dataset Loader & Tokenizer:**
        * **Tanggung Jawab:** Memuat dataset berlabel, melakukan tokenisasi teks menggunakan tokenizer dari model BERT (IndoBERT).
        * **Teknologi:** Python, Hugging Face Datasets/Pandas, Hugging Face Tokenizers.
    * **Model BERT (Fine-tuning & Lapisan Klasifikasi):** [cite: 58, 59, 100, 101, 103]
        * **Tanggung Jawab:** Menggunakan model IndoBERT pre-trained, melakukan fine-tuning dengan dataset Bahasa Jawa berlabel. Menambahkan lapisan klasifikasi untuk tugas deteksi ujaran kebencian (4 kelas).
        * **Teknologi:** Python, Hugging Face Transformers, PyTorch/TensorFlow.
    * **Evaluasi Model:** [cite: 54, 60, 61, 106, 107]
        * **Tanggung Jawab:** Mengukur performa model menggunakan metrik kuantitatif (akurasi, presisi, recall, F1-score) dan analisis confusion matrix. Melakukan analisis kualitatif terhadap kesalahan klasifikasi.
        * **Teknologi:** Python, Scikit-learn, Matplotlib/Seaborn.
    * **Output:** Model terlatih (.bin, .pt, atau format lain), laporan evaluasi, visualisasi.

5.  **Prototipe Aplikasi:**
    * **API Server:** [cite: 7, 62, 108]
        * **Tanggung Jawab:** Menyediakan endpoint untuk prediksi.
        * **Teknologi:** FastAPI/Flask.
    * **Logika Aplikasi:**
        * **Tanggung Jawab:** Menangani request API, melakukan preprocessing pada input pengguna, memanggil model untuk prediksi, dan memformat output.
    * **Antarmuka Pengguna (Web - Sederhana):** [cite: 109]
        * **Tanggung Jawab:** Menyediakan antarmuka web sederhana bagi pengguna untuk memasukkan teks dan melihat hasil klasifikasi.
        * **Teknologi:** HTML, CSS, JavaScript.
    * **Output:** Prototipe API yang berfungsi dan antarmuka pengguna web.

## Alur Kerja Utama (Prediksi):

1.  Pengguna memasukkan teks Bahasa Jawa melalui Antarmuka Pengguna atau langsung ke API Server.
2.  API Server menerima teks dan meneruskannya ke Logika Aplikasi.
3.  Logika Aplikasi melakukan preprocessing pada teks input (mirip dengan preprocessing saat training).
4.  Teks yang sudah diproses ditokenisasi.
5.  Model BERT yang sudah di-fine-tune melakukan prediksi.
6.  Hasil prediksi (kelas ujaran kebencian dan/atau tingkatannya) dikembalikan oleh Logika Aplikasi ke API Server.
7.  API Server mengirimkan respons (misalnya, format JSON) kembali ke Antarmuka Pengguna atau pemanggil API.
8.  Antarmuka Pengguna menampilkan hasil klasifikasi kepada pengguna.

## Pertimbangan Desain:

* **Modularitas:** Setiap komponen dirancang untuk independen sejauh mungkin untuk memudahkan pengembangan, pengujian, dan pembaruan.
* **Iterasi:** Arsitektur mendukung iterasi, terutama pada Modul Machine Learning (pelatihan ulang dengan data lebih banyak, penyesuaian model) dan Prototipe Aplikasi (penambahan fitur).
* **Konfigurasi:** Parameter model, path file, dan konfigurasi lainnya akan dikelola secara eksternal (misalnya, file `.env` atau konfigurasi).

## Diagram Alir Data
```
[Google Sheets] -> [Data Collection Module] -> [Preprocessing] -> [Model Training/Inference] -> [API Endpoints]
```

## Interaksi Antar Komponen
1. Data Collection mengambil data dari Google Sheets
2. Preprocessing menerima data mentah dan menghasilkan data bersih
3. Model menerima data bersih untuk training/inference
4. API menerima request dan mengembalikan hasil prediksi model

--- 