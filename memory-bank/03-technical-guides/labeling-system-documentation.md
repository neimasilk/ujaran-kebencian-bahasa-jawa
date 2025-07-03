# 📋 Dokumentasi Sistem Pelabelan Dataset Ujaran Kebencian Bahasa Jawa

**Dokumentator:** AI Assistant  
**Update Terakhir:** 2024-12-29  
**Versi:** 1.0  
**Status:** Aktif  

---

## 🎯 Overview

Dokumen ini merangkum implementasi sistem pelabelan dataset yang telah dikembangkan untuk proyek deteksi ujaran kebencian Bahasa Jawa. Sistem ini menggabungkan strategi cost-efficient, persistence mechanism, dan cloud integration untuk memastikan proses pelabelan yang robust dan scalable.

## 🏗️ Arsitektur Sistem

### Komponen Utama

1. **DeepSeek Labeling Pipeline** (`src/data_collection/deepseek_labeling_pipeline.py`)
   - Pipeline dasar untuk labeling menggunakan DeepSeek API
   - Implementasi strategi cost-efficient (positive/negative split)
   - Batch processing dan error handling

2. **Persistent Labeling Pipeline** (`src/data_collection/persistent_labeling_pipeline.py`)
   - Extension dari pipeline dasar dengan checkpoint mechanism
   - Resume capability untuk recovery dari interupsi
   - Integrasi dengan cost optimizer

3. **Cost Optimizer** (`src/utils/cost_optimizer.py`)
   - Monitoring periode diskon DeepSeek API (UTC 16:30-00:30)
   - Tiga strategi optimasi: `discount_only`, `always`, `warn_expensive`
   - Real-time cost estimation dan savings tracking

4. **Cloud Checkpoint Manager** (`src/utils/cloud_checkpoint_manager.py`)
   - Integrasi Google Drive untuk persistence lintas device
   - Dual storage (local + cloud) dengan conflict resolution
   - Offline mode fallback

## 💡 Strategi Cost-Efficient

### Konsep Dasar
Sistem menggunakan pendekatan hybrid untuk mengoptimalkan biaya API:

- **Data Positif** → Rule-based labeling sebagai "Bukan Ujaran Kebencian" (Cost: $0)
- **Data Negatif** → DeepSeek API untuk klasifikasi detail (Cost: sesuai usage)

### Estimasi Penghematan
Untuk dataset 41,759 sampel:
- **Tanpa strategi**: ~$5.69 (semua sampel ke API)
- **Dengan strategi**: ~$2.85 (hanya negatif ke API)
- **Penghematan**: 50% atau ~$2.84

### Implementasi
```python
# Contoh penggunaan strategi cost-efficient
from src.data_collection.persistent_labeling_pipeline import PersistentLabelingPipeline

pipeline = PersistentLabelingPipeline(
    mock_mode=False,
    cost_strategy="discount_only",  # Hanya proses saat diskon
    checkpoint_interval=10
)

report = pipeline.run_pipeline(
    input_file="dataset.csv",
    output_file="labeled_results.csv",
    resume=True
)
```

## ⏰ Optimasi Waktu & Biaya

### Periode Diskon DeepSeek API
- **Waktu**: UTC 16:30-00:30 (GMT+7: 23:30-07:30)
- **Durasi**: 8 jam per hari
- **Diskon**: 50% untuk semua komponen pricing

### Strategi Scheduling
1. **Optimal**: Jalankan saat periode diskon (23:30-07:30 WIB)
2. **Fleksibel**: Gunakan `warn_expensive` untuk monitoring real-time
3. **Urgent**: Gunakan `always` dengan monitoring biaya ketat

## 🔄 Persistence & Recovery

### Checkpoint Mechanism
- **Interval**: Configurable (default: 10 batch)
- **Format**: JSON dengan metadata lengkap
- **Location**: Local `checkpoints/` directory
- **Content**: Processed indices, results, metadata

### Resume Capability
```bash
# Resume dari checkpoint terakhir
python persistent_labeling_pipeline.py --input dataset.csv --output results.csv --resume

# List checkpoint yang tersedia
python persistent_labeling_pipeline.py --list-checkpoints
```

### Cloud Persistence
- **Platform**: Google Drive API
- **Authentication**: OAuth 2.0 untuk development, Service Account untuk production
- **Sync**: Automatic upload setelah checkpoint local
- **Conflict Resolution**: Timestamp-based dengan user confirmation

## 🛠️ Setup & Konfigurasi

### Environment Variables
```bash
# DeepSeek API
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Google Drive (Optional)
GOOGLE_DRIVE_ENABLED=true
GOOGLE_CREDENTIALS_PATH=credentials.json
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Google Drive integration (optional)
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## 📊 Monitoring & Metrics

### Real-time Monitoring
- **Cost Tracking**: Input/output tokens dengan estimasi biaya
- **Progress Tracking**: Processed samples, remaining batches
- **Quality Metrics**: Confidence scores, error rates
- **Performance**: Response times, throughput

### Reporting
Setiap run menghasilkan comprehensive report:
```json
{
  "summary": {
    "total_samples": 1000,
    "processing_time_seconds": 1800,
    "samples_per_second": 0.56
  },
  "cost_analysis": {
    "total_cost_usd": 2.85,
    "cost_savings_usd": 2.84,
    "savings_percentage": 50.0
  },
  "quality_metrics": {
    "average_confidence": 0.89,
    "low_confidence_samples": 12
  }
}
```

## 🎮 Demo & Testing

### Available Demos
1. **Cost Optimization Demo**
   ```bash
   python demo_cost_optimization.py --demo status
   python demo_cost_optimization.py --demo strategies
   ```

2. **Persistent Pipeline Demo**
   ```bash
   python demo_persistent_labeling.py
   ```

3. **Cloud Integration Demo**
   ```bash
   python demo_cloud_checkpoint.py
   ```

### Testing Scenarios
- **Mock Mode**: Testing tanpa API calls
- **Interruption Simulation**: Ctrl+C untuk test resume
- **Multi-device**: Test cloud sync antar device

## 🔒 Security & Best Practices

### API Key Management
- Gunakan environment variables, jangan hardcode
- File `.env` sudah di-gitignore
- Rotate API keys secara berkala

### Google Drive Security
- OAuth 2.0 untuk development (user consent)
- Service Account untuk production (automated)
- Credentials files di-gitignore
- Minimal permission scope

### Data Privacy
- Checkpoint files berisi data sensitif
- Enkripsi untuk cloud storage (future enhancement)
- Regular cleanup checkpoint lama

## 📈 Scalability Considerations

### Dataset Size Guidelines
- **Small** (<1K samples): Gunakan `always` strategy
- **Medium** (1K-10K): Gunakan `warn_expensive` dengan monitoring
- **Large** (>10K): Gunakan `discount_only` untuk optimal savings

### Performance Optimization
- Batch size: Default 5, adjust berdasarkan API limits
- Checkpoint interval: Balance antara safety dan performance
- Concurrent processing: Future enhancement

## 🚀 Roadmap & Future Enhancements

### Short Term
- [ ] Automated quality control dengan confidence thresholds
- [ ] Enhanced error recovery mechanisms
- [ ] Performance profiling dan optimization

### Medium Term
- [ ] Multi-model support (selain DeepSeek)
- [ ] Advanced cost prediction algorithms
- [ ] Real-time dashboard untuk monitoring

### Long Term
- [ ] Distributed processing untuk dataset massive
- [ ] ML-based cost optimization
- [ ] Integration dengan annotation tools

## 📚 Referensi & Resources

### Dokumentasi Terkait
- [`memory-bank/cost-optimization-strategy.md`](./cost-optimization-strategy.md) - Detail strategi optimasi biaya
- [`memory-bank/deepseek-api-strategy.md`](./deepseek-api-strategy.md) - Strategi penggunaan DeepSeek API
- [`memory-bank/google-drive-persistence-strategy.md`](./google-drive-persistence-strategy.md) - Planning cloud persistence
- [`memory-bank/architecture.md`](./architecture.md) - Arsitektur sistem lengkap
- [`memory-bank/spesifikasi-produk.md`](./spesifikasi-produk.md) - Spesifikasi produk dan requirements

### Code References
- **Main Pipeline**: `src/data_collection/persistent_labeling_pipeline.py`
- **Cost Optimizer**: `src/utils/cost_optimizer.py`
- **Cloud Manager**: `src/utils/cloud_checkpoint_manager.py`
- **Demo Scripts**: `demo_*.py` files

### External APIs
- [DeepSeek API Documentation](https://api.deepseek.com/docs)
- [Google Drive API Documentation](https://developers.google.com/drive/api)

---

## 🤝 Tim & Kontribusi

### Roles dalam Labeling System
- **Arsitek**: Design sistem dan integration strategy
- **Backend Developer**: Implementation core pipeline dan APIs
- **DevOps**: Setup cloud infrastructure dan deployment
- **Dokumentator**: Maintenance dokumentasi dan user guides
- **Tester**: Quality assurance dan testing scenarios

### Workflow Kolaborasi
1. **Planning**: Diskusi requirements di `memory-bank/papan-proyek.md`
2. **Implementation**: Development dengan checkpoint regular
3. **Testing**: Demo dan validation scenarios
4. **Documentation**: Update docs setelah setiap major change
5. **Review**: Code review dan architecture validation

---

## Dokumentasi Tambahan

### Quick Start & Onboarding
- **Quick Start Guide**: `memory-bank/quick-start-guide.md` - Panduan cepat untuk tim baru
- **Workflow Diagram**: `memory-bank/workflow-diagram.md` - Visualisasi alur kerja sistem
- **Error Handling**: `memory-bank/error-handling-guide.md` - Panduan troubleshooting komprehensif

### Dokumentasi Teknis
- **Architecture**: `memory-bank/architecture.md` - Arsitektur sistem lengkap
- **API Strategy**: `memory-bank/deepseek-api-strategy.md` - Strategi penggunaan DeepSeek API
- **Cost Optimization**: `memory-bank/cost-optimization-strategy.md` - Optimasi biaya operasional
- **Google Drive Integration**: `memory-bank/google-drive-integration.md` - Integrasi dan troubleshooting Google Drive
- **Implementation & Testing**: `memory-bank/implementation-testing.md` - Implementasi force mode dan testing results

### Panduan Tim
- **Manual Work Guide**: `memory-bank/manual-work-guide.md` - Panduan lengkap pekerjaan manual
- **Product Specification**: `memory-bank/spesifikasi-produk.md` - Spesifikasi produk lengkap
- **Team Manifest**: `vibe-guide/team-manifest.md` - Filosofi dan struktur tim

## Kesimpulan

Sistem labeling ini dirancang untuk efisiensi biaya dan kemudahan penggunaan. Dengan strategi cost-efficient dan persistent storage, tim dapat melakukan labeling data secara konsisten dengan biaya yang terkontrol.

**Untuk memulai**:
1. Baca `memory-bank/quick-start-guide.md` untuk setup cepat
2. Setup Google Drive dengan `memory-bank/google-drive-integration.md`
3. Pahami manual work dengan `memory-bank/manual-work-guide.md`
4. Siapkan error handling dengan `memory-bank/error-handling-guide.md`
5. Ikuti implementasi sesuai dokumentasi teknis

**Untuk troubleshooting**: Selalu rujuk ke error handling guide dan konsultasi dengan tim jika diperlukan.

---

*Dokumentasi ini akan diupdate seiring dengan perkembangan sistem. Untuk pertanyaan atau saran, silakan diskusikan di papan proyek atau buat issue di repository.*

**Next Update**: Setelah implementasi automated quality control dan enhanced monitoring features.