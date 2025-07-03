# Deteksi Ujaran Kebencian Bahasa Jawa

Proyek ini mengembangkan sistem deteksi ujaran kebencian untuk teks berbahasa Jawa menggunakan model IndoBERT dan API DeepSeek untuk labeling otomatis.

## 📋 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Konfigurasi](#konfigurasi)
- [Testing](#testing)
- [Kontribusi](#kontribusi)

## 🚀 Fitur Utama

- **Labeling Otomatis**: Menggunakan API DeepSeek untuk melabeli data teks Jawa secara otomatis
- **Parallel Processing**: Mendukung pemrosesan paralel untuk labeling data dalam jumlah besar
- **Model Training**: Fine-tuning model IndoBERT untuk klasifikasi ujaran kebencian
- **Model Evaluation**: Evaluasi komprehensif dengan berbagai metrik
- **Google Drive Integration**: Integrasi dengan Google Drive untuk penyimpanan data
- **Force Mode**: Mode paksa untuk melabeli ulang data yang sudah ada
- **Comprehensive Testing**: Test suite lengkap untuk semua komponen
- **GPU Optimization**: Optimasi training menggunakan NVIDIA GPU dengan CUDA support

## 📁 Struktur Proyek

```
ujaran-kebencian-bahasa-jawa/
├── src/
│   ├── config/
│   │   └── settings.py              # Konfigurasi aplikasi
│   ├── data_processing/
│   │   ├── google_drive_client.py   # Client Google Drive
│   │   ├── labeling_pipeline.py     # Pipeline labeling utama
│   │   └── parallel_labeling.py     # Labeling paralel
│   ├── modelling/
│   │   ├── evaluate_model.py        # Evaluasi model
│   │   └── train_model.py           # Training model
│   ├── tests/
│   │   ├── integration/
│   │   │   └── test_deepseek_api.py # Test integrasi API
│   │   ├── test_evaluation.py       # Test evaluasi
│   │   ├── test_force_mode.py       # Test force mode
│   │   ├── test_parallel_labeling.py # Test labeling paralel
│   │   └── test_training.py         # Test training
│   └── utils/
│       ├── deepseek_client.py       # Client DeepSeek API
│       └── logger.py                # Utilitas logging
├── train_model.py                   # Script training standalone
├── evaluate_model.py                # Script evaluasi standalone
├── requirements.txt                 # Dependencies
└── README.md                        # Dokumentasi ini
```

## 🛠 Instalasi

### Prerequisites

- Python 3.8+
- pip
- Git

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd ujaran-kebencian-bahasa-jawa
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages**
   ```bash
   pip install tf-keras accelerate pydantic-settings tenacity
   ```

## ⚙️ Konfigurasi

### Environment Variables

Buat file `.env` di root directory dengan konfigurasi berikut:

```env
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Google Drive (opsional)
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

# Model Configuration
MODEL_OUTPUT_DIR=./models/hate_speech_model
LABELED_DATA_PATH=./hasil-labeling.csv

# Logging
LOG_LEVEL=INFO
```

### Konfigurasi API DeepSeek

1. Daftar di [DeepSeek](https://platform.deepseek.com/)
2. Dapatkan API key
3. Masukkan API key ke file `.env`

## 🎯 Penggunaan

### 1. Labeling Data

#### Labeling Dasar
```bash
python -m src.data_processing.labeling_pipeline --input_file data.csv --output_file hasil-labeling.csv
```

#### Labeling Paralel
```bash
python -m src.data_processing.parallel_labeling --input_file data.csv --output_file hasil-labeling.csv --max_workers 4
```

#### Force Mode (Labeling Ulang)
```bash
python -m src.data_processing.labeling_pipeline --input_file data.csv --output_file hasil-labeling.csv --force
```

### 2. Training Model

#### Training Dasar
```bash
python train_model.py --data_path ./hasil-labeling.csv --output_dir ./models/hate_speech_model
```

#### Training dengan Parameter Kustom
```bash
python train_model.py \
    --data_path ./hasil-labeling.csv \
    --output_dir ./models/hate_speech_model \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5
```

#### Parameter Training Lengkap
```bash
python train_model.py \
    --data_path ./hasil-labeling.csv \
    --output_dir ./models/hate_speech_model \
    --epochs 5 \
    --batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --test_size 0.2 \
    --logging_steps 10 \
    --save_total_limit 2 \
    --log_level INFO \
    --force
```

### 3. Evaluasi Model

#### Evaluasi dengan Data Berlabel
```bash
python evaluate_model.py \
    --model_path ./models/hate_speech_model \
    --data_path ./data/test_data.csv \
    --output_dir ./evaluation_results
```

#### Prediksi Data Baru (Tanpa Label)
```bash
python evaluate_model.py \
    --model_path ./models/hate_speech_model \
    --data_path ./data/new_data.csv \
    --output_dir ./evaluation_results \
    --no_labels
```

#### Parameter Evaluasi Lengkap
```bash
python evaluate_model.py \
    --model_path ./models/hate_speech_model \
    --data_path ./data/test_data.csv \
    --output_dir ./evaluation_results \
    --text_column "text" \
    --label_column "label" \
    --batch_size 8 \
    --log_level INFO
```

## 🧪 Testing

### Menjalankan Semua Test
```bash
python -m pytest src/tests/ --verbose
```

### Test Spesifik
```bash
# Test training
python -m pytest src/tests/test_training.py --verbose

# Test evaluation
python -m pytest src/tests/test_evaluation.py --verbose

# Test labeling paralel
python -m pytest src/tests/test_parallel_labeling.py --verbose

# Test force mode
python -m pytest src/tests/test_force_mode.py --verbose

# Test integrasi API
python -m pytest src/tests/integration/test_deepseek_api.py --verbose
```

### Test dengan Coverage
```bash
pip install pytest-cov
python -m pytest src/tests/ --cov=src --cov-report=html
```

## 🎯 Hasil Training Model

### Status Training Terbaru
- **Tanggal:** 3 Juli 2025
- **Durasi:** 13 menit (GPU RTX 4080)
- **Status:** ✅ **BERHASIL**
- **Model:** IndoBERT fine-tuned untuk hate speech detection

### Metrik Performa
- **Accuracy:** 95.5%
- **F1-Score (Weighted):** 97.7%
- **Precision (Weighted):** 100.0%
- **Training Samples:** 33,076
- **Validation Samples:** 8,270

### Model Files
```
models/trained_model/
├── config.json              # Model configuration
├── model.safetensors         # Trained weights
├── tokenizer.json           # Tokenizer
└── evaluation_results.json  # Evaluation metrics
```

### Penggunaan Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model
model = AutoModelForSequenceClassification.from_pretrained('models/trained_model')
tokenizer = AutoTokenizer.from_pretrained('models/trained_model')

# Inference
text = "Contoh teks bahasa Jawa"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

### Dokumentasi Lengkap
- **Training Report:** `TRAINING_EVALUATION_REPORT.md`
- **GPU Optimization:** `GPU_OPTIMIZATION_GUIDE.md`
- **Quick Evaluation:** `quick_evaluation.py`

## 📊 Format Data

### Input Data untuk Labeling
File CSV dengan kolom:
- `text`: Teks bahasa Jawa yang akan dilabeli

### Output Data Labeling
File CSV dengan kolom:
- `text`: Teks asli
- `final_label`: Label hasil klasifikasi
- `confidence_score`: Skor kepercayaan (0-1)
- `error`: Pesan error (jika ada)

### Label Categories
1. **Bukan Ujaran Kebencian** (0)
2. **Ujaran Kebencian - Ringan** (1)
3. **Ujaran Kebencian - Sedang** (2)
4. **Ujaran Kebencian - Berat** (3)

## 🔧 Troubleshooting

### Error Umum

1. **ModuleNotFoundError**
   ```bash
   # Pastikan virtual environment aktif
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Install PyTorch dengan CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**
   - Kurangi batch size
   - Gunakan gradient accumulation
   - Gunakan mixed precision training

4. **API Rate Limiting**
   - Tambahkan delay antar request
   - Gunakan retry mechanism
   - Kurangi jumlah worker paralel

### Performance Tips

1. **Training**
   - Gunakan GPU jika tersedia
   - Sesuaikan batch size dengan memory
   - Gunakan mixed precision (fp16)

2. **Labeling**
   - Gunakan parallel processing
   - Batch request ke API
   - Implement caching

## 📈 Monitoring dan Logging

Sistem menggunakan logging komprehensif:

- **INFO**: Informasi umum progress
- **DEBUG**: Detail teknis untuk debugging
- **WARNING**: Peringatan yang tidak menghentikan proses
- **ERROR**: Error yang menghentikan proses

Log disimpan di console dan dapat diarahkan ke file:

```bash
python train_model.py --log_level DEBUG 2>&1 | tee training.log
```

## 🤝 Kontribusi

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

### Guidelines Kontribusi

- Ikuti PEP 8 style guide
- Tambahkan test untuk fitur baru
- Update dokumentasi
- Pastikan semua test pass

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Kontak

Jika ada pertanyaan atau masalah, silakan buat issue di repository ini.

---

**Catatan**: Proyek ini dikembangkan untuk penelitian dan edukasi. Pastikan untuk mematuhi terms of service dari semua API yang digunakan.