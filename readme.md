# Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Status Proyek:** Development Phase | **Vibe Coding Guide:** v1.4 | **Tim:** Hibrida (Manusia + AI)

Sistem machine learning untuk mendeteksi ujaran kebencian dalam teks berbahasa Jawa menggunakan model IndoBERT yang di-fine-tuning. Proyek ini dikembangkan menggunakan metodologi Vibe Coding v1.4 dengan kolaborasi tim hibrida antara manusia dan AI.

## 🚀 Quick Start

### Instalasi
```bash
# Clone repository
git clone <repository-url>
cd ujaran-kebencian-bahasa-jawa

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.template .env
# Edit .env dengan konfigurasi Anda
```

### Labeling Dataset
```bash
# Mulai proses labeling
python labeling.py

# Atau gunakan mode parallel untuk performa lebih baik
python production_parallel_labeling.py
```

### Training Model
```bash
# Train model dengan dataset yang sudah dilabel
python src/modelling/train_model.py

# Evaluasi model
python src/modelling/evaluate_model.py
```

## 📊 Dataset Overview

- **Ukuran**: 41,759 samples
- **Bahasa**: Bahasa Jawa
- **Format**: CSV (text, label)
- **Distribusi**: Balanced (50% hate speech, 50% normal)
- **Lokasi**: `src/data_collection/raw-dataset.csv`

## 📚 Dokumentasi

### Panduan Utama
- **[Memory Bank](memory-bank/README.md)** - Pusat dokumentasi proyek
- **[Panduan Labeling](memory-bank/PANDUAN_LABELING.md)** - Tutorial lengkap labeling
- **[Quick Start Guide](memory-bank/quick-start-guide.md)** - Setup cepat untuk tim baru

### Dokumentasi Teknis
- **[Architecture](memory-bank/architecture.md)** - Arsitektur sistem
- **[API Strategy](memory-bank/deepseek-api-strategy.md)** - Strategi penggunaan DeepSeek API
- **[Cost Optimization](memory-bank/cost-optimization-strategy.md)** - Optimasi biaya operasional
- **[Google Drive Integration](memory-bank/google-drive-integration.md)** - Setup cloud persistence

### Panduan Tim
- **[Vibe Coding Guide](vibe-guide/VIBE_CODING_GUIDE.md)** - Metodologi pengembangan
- **[Team Roles](vibe-guide/PERAN_TIM.md)** - Peran dan tanggung jawab tim
- **[Project Board](memory-bank/papan-proyek.md)** - Status dan task management

## 🛠️ Fitur Utama

### Cost Optimization
- **Smart Scheduling**: Deteksi otomatis periode diskon DeepSeek API (50% OFF)
- **Real-time Monitoring**: Tracking biaya dan penghematan
- **Multiple Strategies**: 3 strategi optimasi sesuai kebutuhan

### Cloud Persistence
- **Cross-device**: Sinkronisasi otomatis dengan Google Drive
- **Offline Mode**: Fallback ke local storage
- **Conflict Resolution**: Handling concurrent modifications

### Parallel Processing
- **Multi-threading**: Proses labeling paralel untuk performa optimal
- **Checkpoint System**: Resume otomatis jika terinterupsi
- **Progress Tracking**: Monitoring real-time progress

## 🤝 Kontribusi

Proyek ini menggunakan metodologi Vibe Coding v1.4. Silakan baca [panduan kontribusi](vibe-guide/VIBE_CODING_GUIDE.md) sebelum berkontribusi.

## 📄 Lisensi

MIT License - Lihat file LICENSE untuk detail lengkap.

---

**Dikembangkan dengan ❤️ menggunakan Vibe Coding v1.4**

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
ujaran-kebencian-bahasa-jawa/
├── src/                    # Source code, data, dan semua file proyek
│   ├── api/               # API endpoints
│   ├── config/            # Konfigurasi
│   ├── data_collection/   # Modul pengumpulan data
│   ├── preprocessing/     # Modul preprocessing
│   ├── modelling/         # Modul machine learning
│   ├── utils/             # Utility functions
│   ├── tests/             # Unit tests
│   ├── notebooks/         # Jupyter notebooks untuk eksperimen
│   ├── models/            # Model yang sudah dilatih
│   ├── checkpoints/       # Checkpoint untuk recovery
│   ├── logs/              # Log files
│   └── *.py               # Script demo dan testing
├── memory-bank/           # Dokumentasi dan konteks proyek
├── vibe-guide/            # Panduan tim dan workflow
├── requirements.txt       # Dependencies
└── README.md             # Dokumentasi utama
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

## 🚀 Quick Start

### Untuk Kontributor Baru
1. **Baca Panduan Utama:** [`vibe-guide/VIBE_CODING_GUIDE.md`](vibe-guide/VIBE_CODING_GUIDE.md)
2. **Pahami Tim:** [`vibe-guide/team-manifest.md`](vibe-guide/team-manifest.md)
3. **Setup Environment:** [`memory-bank/environment-setup.md`](memory-bank/environment-setup.md)
4. **Lihat Status Terkini:** [`memory-bank/papan-proyek.md`](memory-bank/papan-proyek.md)

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd ujaran-kebencian-bahasa-jawa

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.template .env
# Edit .env file dengan API keys Anda
```

### 2. Demo Cost Optimization
```bash
# Lihat status harga saat ini
python demo_cost_optimization.py --demo status

# Demo semua fitur cost optimization
python demo_cost_optimization.py --demo all
```

### 3. Jalankan Pipeline dengan Optimasi Biaya
```bash
# Test dengan 10 samples
python test_deepseek_negative_10.py

# Pipeline dengan persistence dan optimasi biaya (hanya saat diskon)
python src/data_collection/persistent_labeling_pipeline.py \
  --input src/data_collection/raw-dataset.csv \
  --output labeled-results.csv \
  --cost-strategy discount_only

# Pipeline dengan peringatan biaya (default)
python src/data_collection/persistent_labeling_pipeline.py \
  --input src/data_collection/raw-dataset.csv \
  --output labeled-results.csv
```

## 📋 Dokumentasi Proyek

| Dokumen | Deskripsi | Status |
|---------|-----------|--------|
| [`memory-bank/spesifikasi-produk.md`](memory-bank/spesifikasi-produk.md) | Kebutuhan dan spesifikasi produk lengkap | ✅ Selesai |
| [`memory-bank/architecture.md`](memory-bank/architecture.md) | Arsitektur sistem dan komponen | ✅ Selesai |
| [`memory-bank/papan-proyek.md`](memory-bank/papan-proyek.md) | Status dan tugas terkini | 🔄 Aktif |
| [`memory-bank/progress.md`](memory-bank/progress.md) | Log progress dan milestone | 🔄 Aktif |

## 🤝 Tim Pengembang

Proyek ini dikembangkan oleh tim hibrida yang terdiri dari:
- **Mukhlis Amien** (Manusia) - Arsitek/Kepala Tim
- **Hashfi** (Manusia) - Developer
- **jules_dokumen** (AI) - Maintainer Dokumentasi
- **jules_dev1 & jules_dev2** (AI) - Developer

Detail lengkap tim tersedia di [`vibe-guide/team-manifest.md`](vibe-guide/team-manifest.md).

## Lisensi

(Akan ditentukan kemudian)

---