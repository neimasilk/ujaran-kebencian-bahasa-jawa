# Javanese Hate Speech Detection API

API untuk deteksi ujaran kebencian dalam Bahasa Jawa menggunakan model IndoBERT yang telah di-fine-tune.

## 🚀 Quick Start

### 1. Menjalankan API Server

```bash
# Dari direktori root proyek
cd src/api
python run_server.py

# Atau dengan konfigurasi custom
python run_server.py --host 0.0.0.0 --port 8000 --reload
```

### 2. Akses Web Interface

Buka browser dan kunjungi: http://localhost:8000

### 3. API Documentation

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

## 📋 API Endpoints

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "models/bert_jawa_hate_speech"
}
```

### Single Text Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Sugeng enjing, piye kabare?"
}
```

Response:
```json
{
  "text": "Sugeng enjing, piye kabare?",
  "predicted_label": "Bukan Ujaran Kebencian",
  "confidence": 0.9876,
  "label_id": 0,
  "processing_time": 0.045
}
```

### Batch Prediction
```http
POST /batch-predict
Content-Type: application/json

{
  "texts": [
    "Sugeng enjing, piye kabare?",
    "Kowe ki bodho tenan!",
    "Aku seneng banget karo kowe"
  ]
}
```

Response:
```json
{
  "predictions": [
    {
      "text": "Sugeng enjing, piye kabare?",
      "predicted_label": "Bukan Ujaran Kebencian",
      "confidence": 0.9876,
      "label_id": 0,
      "processing_time": 0.045
    }
  ],
  "total_processing_time": 0.123
}
```

### Model Information
```http
GET /model-info
```

Response:
```json
{
  "model_path": "models/bert_jawa_hate_speech",
  "device": "cuda",
  "total_parameters": 124439808,
  "trainable_parameters": 124439808,
  "model_size_mb": 474.77,
  "labels": {
    "0": "Bukan Ujaran Kebencian",
    "1": "Ujaran Kebencian - Ringan",
    "2": "Ujaran Kebencian - Sedang",
    "3": "Ujaran Kebencian - Berat"
  },
  "max_sequence_length": 128
}
```

## 🧪 Testing API

### Menggunakan Test Script

```bash
# Test komprehensif
python test_api.py

# Test single text
python test_api.py --text "Sugeng enjing, piye kabare?"

# Test batch
python test_api.py --batch "Text 1" "Text 2" "Text 3"

# Test dengan base URL custom
python test_api.py --base-url http://localhost:8000
```

### Menggunakan curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Sugeng enjing, piye kabare?"}'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2"]}'
```

### Menggunakan Python requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Sugeng enjing, piye kabare?"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch-predict",
    json={"texts": ["Text 1", "Text 2"]}
)
print(response.json())
```

## 🔧 Configuration

### Environment Variables

Buat file `.env` di direktori root:

```env
# Model Configuration
MODEL_PATH=models/bert_jawa_hate_speech
MAX_SEQUENCE_LENGTH=128

# Server Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=1

# Performance
BATCH_SIZE_LIMIT=100
REQUEST_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Server Options

```bash
python run_server.py --help

usage: run_server.py [-h] [--host HOST] [--port PORT] [--reload] 
                     [--workers WORKERS] [--log-level LOG_LEVEL]

Javanese Hate Speech Detection API Server

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           Host to bind the server (default: 127.0.0.1)
  --port PORT           Port to bind the server (default: 8000)
  --reload              Enable auto-reload for development
  --workers WORKERS     Number of worker processes (default: 1)
  --log-level LOG_LEVEL Log level (default: info)
```

## 📊 Performance

### Benchmarks

- **Single Prediction**: ~45ms (GPU) / ~200ms (CPU)
- **Batch Prediction (10 texts)**: ~120ms (GPU) / ~800ms (CPU)
- **Memory Usage**: ~2GB (dengan model loaded)
- **Throughput**: ~20 requests/second (single prediction)

### Optimization Tips

1. **GPU Usage**: Pastikan PyTorch dengan CUDA support terinstall
2. **Batch Processing**: Gunakan batch prediction untuk multiple texts
3. **Model Caching**: Model dimuat sekali saat startup
4. **Memory Management**: Gunakan `torch.no_grad()` untuk inference

## 🚨 Error Handling

### Common Errors

| Status Code | Error | Solution |
|-------------|-------|----------|
| 400 | Text tidak boleh kosong | Pastikan input text tidak kosong |
| 400 | Maksimal 100 teks per batch | Kurangi jumlah teks dalam batch |
| 422 | Validation Error | Periksa format JSON request |
| 503 | Model belum dimuat | Pastikan model tersedia di path yang benar |
| 500 | Internal Server Error | Periksa logs untuk detail error |

### Troubleshooting

1. **Model tidak ditemukan**:
   ```bash
   # Pastikan model sudah dilatih
   python src/modelling/train_model.py
   
   # Atau periksa path model
   ls -la models/bert_jawa_hate_speech/
   ```

2. **GPU tidak terdeteksi**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

3. **Memory error**:
   - Kurangi batch size
   - Gunakan CPU jika GPU memory tidak cukup
   - Restart server untuk clear memory

## 🔒 Security

### Production Considerations

1. **CORS**: Update `allow_origins` dengan domain yang spesifik
2. **Rate Limiting**: Implementasi rate limiting untuk production
3. **Authentication**: Tambahkan API key atau JWT authentication
4. **Input Validation**: Validasi dan sanitasi input text
5. **HTTPS**: Gunakan HTTPS untuk production deployment

### Example Production Config

```python
# production_config.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📁 File Structure

```
src/api/
├── __init__.py
├── main.py              # FastAPI application
├── run_server.py        # Server runner script
├── test_api.py          # API testing script
├── README.md            # This file
└── static/
    └── index.html       # Web interface
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

Jika mengalami masalah:

1. Periksa logs server
2. Jalankan test script
3. Periksa dokumentasi
4. Buat issue di repository

---

**Happy Coding! 🚀**