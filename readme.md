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

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)
- DeepSeek API Key (untuk pelabelan data)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ujaran-kebencian-bahasa-jawa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-deepseek.txt

# Setup DeepSeek API (untuk pelabelan data)
python setup_deepseek_env.py

# Verify installation
python src/data_collection/verify_dataset.py
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

## 📊 Current Progress

### ✅ Completed
- [x] Project structure setup
- [x] Data collection pipeline
- [x] Basic preprocessing functions
- [x] Unit testing infrastructure
- [x] API documentation
- [x] **DeepSeek V3 API integration untuk pelabelan**

### 🔄 In Progress
- [x] **Data labeling dengan DeepSeek V3 API**
- [ ] Quality assurance hasil pelabelan
- [ ] Model development dengan dataset berlabel
- [ ] API implementation

### ⏳ Planned
- [ ] Training pipeline dengan dataset DeepSeek
- [ ] Web interface
- [ ] Deployment setup
- [ ] Performance optimization

## 🛠️ Development Workflow

Proyek ini menggunakan **Vibe Coding V1.4** dengan kolaborasi AI-Human:

### Team Structure
- **Human**: Mukhlis Amien (Architect), Hashfi (Developer)
- **AI**: jules_dokumen (Documentation), jules_dev1 & jules_dev2 (Development)

### Baby-steps Approach
1. Setiap iterasi dibagi menjadi baby-steps kecil
2. Setiap baby-step memiliki deliverable yang jelas
3. Testing dan dokumentasi terintegrasi
4. AI commit hasil kerja mereka sendiri

### Current Baby-step
**Data Labeling dengan DeepSeek V3 API** 🔄 DALAM PROGRESS
- ✅ Setup DeepSeek API integration
- ✅ Dokumentasi lengkap penggunaan DeepSeek
- ⏳ Processing dataset dengan AI labeling
- ⏳ Quality assurance dan validasi manual

## 📁 Project Structure

```
ujaran-kebencian-bahasa-jawa/
├── src/
│   ├── data_collection/     # Data loading, preprocessing, DeepSeek labeling
│   ├── model/              # Model development
│   └── api/                # API implementation
├── data/
│   ├── raw/                # Raw dataset
│   └── processed/          # Processed dataset, DeepSeek results
├── tests/                  # Unit tests
├── docs/                   # Documentation, DeepSeek guide
├── vibe-guide/            # Development guidelines
├── memory-bank/           # Project memory dan progress
├── baby-steps-archive/    # Archive completed steps
├── setup_deepseek_env.py  # DeepSeek API setup
├── requirements-deepseek.txt # DeepSeek dependencies
└── .env                   # API keys (tidak di-commit)
```

## 🔧 Key Features

### Data Pipeline
- Robust CSV loading dengan error handling
- **DeepSeek V3 API integration untuk auto-labeling**
- Data validation dan cleaning
- Preprocessing untuk Bahasa Jawa

### AI-Powered Labeling
- **Automated labeling dengan DeepSeek V3**
- Pemahaman konteks budaya Bahasa Jawa
- Confidence scoring dan reasoning
- Quality assurance workflow

### Testing Infrastructure
- Unit tests dengan pytest
- Coverage reporting
- Automated testing pipeline

### Documentation
- API reference lengkap
- **DeepSeek integration guide**
- Development guidelines
- Progress tracking

## Panduan Kontribusi

Untuk berkontribusi pada proyek ini, silakan baca:
1. Petunjuk pekerjaan manual di [`memory-bank/petunjuk pekerjaan manual.md`](memory-bank/petunjuk-pekerjaan-manual.md)
2. Setup environment di [`memory-bank/environment-setup.md`](memory-bank/environment-setup.md)

(Akan dilengkapi dengan panduan kontribusi lainnya jika proyek bersifat kolaboratif)

## Lisensi

(Akan ditentukan kemudian)

---