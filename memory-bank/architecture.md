# Arsitektur Sistem - Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 1.0
**Tanggal:** 2 Januari 2025
**Arsitek:** Mukhlis Amien
**Review Status:** ✅ Reviewed & Updated

Dokumen ini menjelaskan arsitektur sistem deteksi ujaran kebencian Bahasa Jawa yang telah dioptimalkan berdasarkan analisis mendalam terhadap implementasi saat ini dan kebutuhan proyek. Arsitektur ini mengikuti prinsip modular, scalable, dan maintainable sesuai dengan Vibe Coding Guide v1.4.

## Komponen Utama:

```mermaid
graph TD
    subgraph "Persiapan Data"
        A[Pengumpulan Data: raw-dataset.csv] --> B(Preprocessing & Filtering)
        B --> C{Pelabelan Data Manual}
    end

    subgraph "Modul Machine Learning"
        C -- Dataset Berlabel --> D[Dataset Loader & Tokenizer]
        D --> E{Fine-tuning Model BERT}
        E --> F[Evaluasi Model]
        F -- Model Terlatih --> G((Simpan Model))
        E -- Iterasi --> E
    end

    subgraph "Prototipe Aplikasi"
        G -- Model Terlatih --> H{API Server: FastAPI}
        H <--> I[Logika Aplikasi]
        I <--> J[Antarmuka Pengguna Web]
    end

    J -- Input Teks --> I
    I -- Prediksi --> J
```

## Deskripsi Komponen:

1.  **Pengumpulan Data:**
    * **Tanggung Jawab:** Menggunakan dataset teks berbahasa Jawa yang sudah ada dan tersimpan dalam format CSV (`raw-dataset.csv`).
    * **Teknologi:** Skrip Python, Pandas.
    * **Input:** File `raw-dataset.csv` yang berada di direktori `src/data_collection/`.
    * **Output:** Dataset mentah dalam bentuk DataFrame Pandas.

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
        * **Tanggung Jawab:** Menggunakan model IndoBERT pre-trained, melakukan fine-tuning dengan dataset Bahasa Jawa berlabel. Menambahkan lapisan klasifikasi untuk tugas deteksi ujaran kebencian dengan 4 kelas: **Bukan Ujaran Kebencian, Ujaran Kebencian Ringan, Ujaran Kebencian Sedang,** dan **Ujaran Kebencian Berat**.
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

## Analisis Arsitektur Saat Ini (Review Arsitek)

### ✅ Kekuatan Implementasi
1. **Struktur Modular yang Baik**: Kode sudah terorganisir dengan baik dalam folder `src/` dengan pemisahan yang jelas:
   - `data_collection/`: Handling dataset loading
   - `preprocessing/`: Text preprocessing utilities
   - `modelling/`: Model training dan evaluation
   - `utils/`: General utilities

2. **Testing Infrastructure**: Unit tests sudah diimplementasi dengan coverage >80%
3. **Documentation**: API documentation sudah tersedia
4. **Data Pipeline**: Dataset loading dan inspection sudah berfungsi dengan baik

### ⚠️ Area yang Perlu Diperbaiki
1. **Dependencies Management**: File `requirements.txt` kosong - perlu diisi dengan dependencies yang tepat
2. **Configuration Management**: Belum ada sistem konfigurasi terpusat
3. **Error Handling**: Perlu standardisasi error handling di seluruh aplikasi
4. **Logging System**: Belum ada sistem logging yang terstruktur
5. **Model Versioning**: Belum ada strategi untuk model versioning dan deployment

## Rekomendasi Arsitektur (Prioritas Tinggi)

### 1. Dependency Management
```python
# requirements.txt yang direkomendasikan:
torch>=1.9.0
transformers>=4.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
pytest>=6.0.0
numpy>=1.21.0
```

### 2. Configuration Management
```python
# src/config/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    model_name: str = "indolem/indobert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    class Config:
        env_file = ".env"
```

### 3. Logging System
```python
# src/utils/logger.py
import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
```

## Pertimbangan Desain (Updated):

* **Modularitas**: ✅ Sudah diimplementasi dengan baik
* **Scalability**: Perlu ditambahkan load balancing dan caching untuk production
* **Maintainability**: Perlu standardisasi coding standards dan documentation
* **Security**: Perlu implementasi API authentication dan rate limiting
* **Monitoring**: Perlu sistem monitoring untuk model performance dan API health

## Diagram Alir Data (Sederhana)

```
[raw-dataset.csv] -> [Preprocessing] -> [Pelabelan] -> [Training Model] -> [Model Tersimpan] -> [API Server]
```

## Interaksi Antar Komponen
1.  Skrip preprocessing memuat data dari `raw-dataset.csv`.
2.  Data yang sudah bersih dan berlabel digunakan untuk melatih model.
3.  Model yang sudah terlatih disimpan dalam direktori `models/`.
4.  API Server memuat model yang tersimpan untuk melakukan prediksi pada data baru yang masuk.

---