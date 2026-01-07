# Javanese Hate Speech Detection API

API untuk deteksi ujaran kebencian dalam bahasa Jawa menggunakan model BERT yang telah dilatih khusus.

## 🚀 Fitur Utama

- **Deteksi Ujaran Kebencian**: Klasifikasi teks bahasa Jawa ke dalam 4 kategori:
  - Bukan Ujaran Kebencian
  - Ujaran Kebencian - Ringan
  - Ujaran Kebencian - Sedang
  - Ujaran Kebencian - Berat

- **API RESTful**: Endpoint yang mudah digunakan untuk integrasi
- **Batch Processing**: Prediksi multiple teks sekaligus
- **Demo Mode**: Fallback ketika model belum tersedia
- **Health Check**: Monitoring status aplikasi dan model

## 📁 Struktur Proyek

```
ujaran-kebencian-clean/
├── README.md
├── requirements.txt
├── .env.template
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints
│   │   ├── models.py        # Pydantic models
│   │   └── static/          # Web interface files
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration exports
│   │   └── settings.py      # Application settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predict.py       # Prediction logic
│   │   ├── train_model.py   # Model training
│   │   └── evaluate_model.py # Model evaluation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_preprocessing.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py        # Logging utilities
│       └── data_utils.py    # Data processing utilities
├── data/
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed datasets
│   └── models/              # Trained models
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # API tests
│   └── test_models.py       # Model tests
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
└── logs/                    # Application logs
```

## 🛠️ Instalasi

### Prerequisites

- Python 3.8+
- pip atau conda

### Setup Environment

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd ujaran-kebencian-clean
   ```

2. **Buat virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env file sesuai kebutuhan
   ```

## 🚀 Menjalankan Aplikasi

### Development Server

```bash
# Dari root directory
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Server

```bash
# Menggunakan Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

API akan tersedia di: `http://localhost:8000`

## 📖 Penggunaan API

### Endpoints

- **GET** `/` - Web interface (jika tersedia)
- **GET** `/health` - Health check
- **GET** `/api` - API information
- **POST** `/predict` - Single text prediction
- **POST** `/batch-predict` - Batch text prediction
- **POST** `/reload-model` - Reload model
- **GET** `/model-info` - Model information
- **GET** `/docs` - API documentation (Swagger UI)

### Contoh Penggunaan

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Aku seneng banget karo kowe"}'
```

Response:
```json
{
  "text": "Aku seneng banget karo kowe",
  "predicted_label": "Bukan Ujaran Kebencian",
  "confidence": 0.95,
  "label_id": 0,
  "processing_time": 0.123
}
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "Aku seneng banget karo kowe",
         "Kowe iku apik tenan",
         "Aku gak seneng karo kowe"
       ]
     }'
```

## 🧪 Testing

### Menjalankan Tests

```bash
# Semua tests
pytest

# Test specific file
pytest tests/test_api.py

# Test dengan coverage
pytest --cov=app tests/
```

### Manual Testing

```bash
# Test API endpoints
python tests/test_api.py

# Test model functionality
python tests/test_models.py
```

## 🎯 Model Training

### Training Model Baru

```bash
# Dari root directory
python scripts/train.py
```

### Evaluasi Model

```bash
# Evaluasi model yang sudah dilatih
python scripts/evaluate.py
```

## ⚙️ Konfigurasi

Konfigurasi aplikasi dapat diatur melalui file `.env` atau environment variables:

```env
# Model Configuration
MODEL_NAME=indolem/indobert-base-uncased
MODEL_MAX_LENGTH=512
NUM_LABELS=4

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs

# DeepSeek API (optional)
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

Logs tersimpan di direktori `logs/` dengan format:
- `hate_speech_detection.log` - Application logs
- Rotasi otomatis ketika file mencapai 10MB

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 📝 License

Project ini menggunakan MIT License. Lihat file `LICENSE` untuk detail.

## 🆘 Troubleshooting

### Model Tidak Ditemukan

Jika model belum tersedia, aplikasi akan berjalan dalam demo mode:
- Prediksi menggunakan rule-based logic sederhana
- Response akan ditandai dengan "(DEMO)"
- Untuk menggunakan model sesungguhnya, letakkan model di `data/models/bert_jawa_hate_speech/`

### Port Sudah Digunakan

```bash
# Ganti port dalam command
uvicorn app.main:app --port 8001
```

### Permission Error

```bash
# Pastikan direktori logs dapat ditulis
sudo chmod 755 logs/
```

## 📞 Support

Jika mengalami masalah atau memiliki pertanyaan:
1. Cek dokumentasi API di `/docs`
2. Lihat logs di direktori `logs/`
3. Buat issue di repository GitHub

---

**Dibuat dengan ❤️ untuk deteksi ujaran kebencian bahasa Jawa**