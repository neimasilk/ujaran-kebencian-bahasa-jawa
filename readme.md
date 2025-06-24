# Sistem Deteksi Ujaran Kebencian Bahasa Jawa Menggunakan BERT

Proyek ini bertujuan untuk membangun sebuah sistem yang mampu secara cerdas dan akurat mendeteksi ujaran kebencian dalam teks berbahasa Jawa, dengan memanfaatkan model IndoBERT yang di-fine-tuning.

## Dataset

Dataset yang digunakan dalam proyek ini adalah kumpulan teks Bahasa Jawa yang telah diberi label. Dataset disimpan dalam format CSV dan terletak di `src/data_collection/raw-dataset.csv`. Proses pemuatan dan inspeksi data ditangani oleh script di `src/data_collection/load_csv_dataset.py`.

## Ringkasan Proyek

Proyek ini bertujuan untuk mengembangkan sistem deteksi ujaran kebencian dalam Bahasa Jawa dengan mengintegrasikan Kecerdasan Buatan (AI) dan kearifan lokal. [cite: 20] Model utama yang akan digunakan adalah BERT (Bidirectional Encoder Representations from Transformers)[cite: 19, 22], yang akan dilatih dan disesuaikan untuk mengenali nuansa linguistik dan budaya Bahasa Jawa. [cite: 23, 49]

Meningkatnya penyebaran ujaran kebencian di platform daring, terutama dalam bahasa daerah yang sumber dayanya terbatas, menjadi urgensi utama penelitian ini. [cite: 18, 38] Bahasa Jawa, dengan kompleksitas seperti tingkatan bahasa (ngoko, krama) dan variasi dialek, menyajikan tantangan unik untuk deteksi otomatis. [cite: 39, 40]

## Tujuan Utama

1.  Mengembangkan model *machine learning* berbasis BERT yang akurat untuk mendeteksi ujaran kebencian dalam Bahasa Jawa. [cite: 19, 21]
2.  Mengintegrasikan aspek kearifan lokal dalam proses pelabelan data dan analisis model untuk meningkatkan pemahaman konteks budaya. [cite: 20, 23]
3.  Menghasilkan *dataset* ujaran kebencian Bahasa Jawa berlabel (ringan, sedang, berat) yang dapat digunakan untuk penelitian selanjutnya. [cite: 7, 57, 96, 121]
4.  Membangun prototipe API berbasis web untuk demonstrasi dan pengujian sistem deteksi. [cite: 7, 62, 108]

## Teknologi yang Digunakan (Direncanakan)

* **Model AI:** BERT (IndoBERT sebagai basis) [cite: 58, 100]
* **Bahasa Pemrograman:** Python
* **Framework/Library:**
    * Hugging Face Transformers (untuk BERT)
    * Scikit-learn (untuk evaluasi)
    * Pandas, NumPy (untuk manipulasi data)
    * NLTK, Sastrawi (atau library NLP Indonesia/Jawa lainnya untuk preprocessing)
    * Flask/FastAPI (untuk API prototipe) [cite: 108]
* **Version Control:** Git, GitHub

## Struktur Proyek

```
.
├── data/               # Data mentah dan yang sudah diproses
│   ├── raw/           # Data mentah
│   └── processed/     # Data yang sudah diproses
├── notebooks/         # Jupyter notebooks untuk eksperimen
├── src/              # Source code
│   ├── data_collection/  # Script pengumpulan data
│   ├── preprocessing/    # Script preprocessing
│   ├── modelling/       # Script model ML
│   ├── api/             # API endpoints
│   └── utils/           # Fungsi utilitas
├── models/           # Model yang sudah dilatih
├── tests/           # Unit tests
└── docs/            # Dokumentasi tambahan
```

## Setup Environment

Proyek ini menggunakan Anaconda untuk manajemen environment dan package Python. Berikut langkah-langkah untuk setup environment:

1. Install [Anaconda](https://www.anaconda.com/download)
2. Buat environment conda baru:
   ```bash
   conda create -n ujaran python=3.11
   ```
3. Aktifkan environment:
   ```bash
   conda activate ujaran
   ```
4. Install semua dependencies dari requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

Detail lebih lanjut tentang setup environment dapat dilihat di `memory-bank/environment-setup.md`.

## Panduan Kontribusi

Untuk berkontribusi pada proyek ini, silakan baca:
1. Petunjuk pekerjaan manual di [`memory-bank/petunjuk pekerjaan manual.md`](memory-bank/petunjuk-pekerjaan-manual.md)
2. Setup environment di [`memory-bank/environment-setup.md`](memory-bank/environment-setup.md)

(Akan dilengkapi dengan panduan kontribusi lainnya jika proyek bersifat kolaboratif)

## Lisensi

(Akan ditentukan kemudian)

---