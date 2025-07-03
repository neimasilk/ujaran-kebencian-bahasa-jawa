# Strategi Cost-Efficient DeepSeek API untuk Labeling Data

## 📋 Overview

Dokumen ini menjelaskan strategi penggunaan DeepSeek API yang cost-efficient untuk labeling data ujaran kebencian bahasa Jawa. Strategi ini mengoptimalkan biaya dengan membagi data menjadi kategori positif dan negatif.

## 🎯 Konsep Dasar

### Pembagian Data

1. **Data Positif** (`label: "positive"`)
   - **Definisi**: Teks yang sudah dipastikan bukan ujaran kebencian
   - **Strategi**: Langsung dilabeli sebagai "Bukan Ujaran Kebencian"
   - **Biaya**: $0 (tidak menggunakan API)
   - **Metode**: Rule-based labeling

2. **Data Negatif** (`label: "negative"`)
   - **Definisi**: Teks yang berpotensi mengandung ujaran kebencian
   - **Strategi**: Dilabeli detail menggunakan DeepSeek API
   - **Biaya**: Sesuai penggunaan API
   - **Metode**: AI-assisted labeling dengan 4 kategori detail

## 💰 Analisis Cost Savings

### Estimasi Penghematan

```
Asumsi Dataset: 1000 sampel
- Data Positif: 600 sampel (60%)
- Data Negatif: 400 sampel (40%)

Tanpa Strategi:
- Semua 1000 sampel → DeepSeek API
- Biaya: 1000 × $0.002 = $2.00

Dengan Strategi:
- 600 sampel → Rule-based ($0)
- 400 sampel → DeepSeek API ($0.80)
- Total Biaya: $0.80
- Penghematan: 60% ($1.20)
```

### Faktor yang Mempengaruhi Penghematan

1. **Rasio Positif/Negatif**: Semakin tinggi rasio data positif, semakin besar penghematan
2. **Kualitas Pre-filtering**: Akurasi pembagian awal mempengaruhi efektivitas
3. **Kompleksitas Prompt**: Prompt yang lebih kompleks = biaya lebih tinggi

## 🔧 Implementasi Teknis

### Arsitektur Sistem

```
raw-dataset.csv
       ↓
[Data Filtering]
   ↙        ↘
Positive    Negative
   ↓           ↓
Rule-based  DeepSeek API
   ↓           ↓
"Bukan      [4 Kategori]
Ujaran      Detail
Kebencian"
   ↘        ↙
  [Combine Results]
       ↓
labeled-dataset.csv
```

### Komponen Utama

1. **DeepSeekLabelingStrategy** (`src/utils/deepseek_labeling.py`)
   - Filter data berdasarkan label awal
   - Prepare data untuk API calls
   - Combine hasil labeling

2. **DeepSeekAPIClient** (`src/utils/deepseek_client.py`)
   - Handle API communication
   - Rate limiting dan retry logic
   - Response parsing

3. **DeepSeekLabelingPipeline** (`src/data_collection/deepseek_labeling_pipeline.py`)
   - Orchestrate seluruh proses
   - Generate comprehensive reports
   - Handle batch processing

## 📊 Monitoring dan Metrics

### Key Performance Indicators (KPIs)

1. **Cost Efficiency**
   - Target: >50% penghematan biaya
   - Metrik: `(rule_based_samples / total_samples) × 100`

2. **API Performance**
   - Success Rate: >95%
   - Average Response Time: <2 detik
   - Error Rate: <5%

3. **Quality Metrics**
   - Average Confidence Score: >70%
   - Low Confidence Samples: <20%
   - Manual Review Required: <10%

### Reporting Dashboard

Setiap run pipeline menghasilkan laporan komprehensif:

```json
{
  "cost_analysis": {
    "samples_processed_by_rule": 600,
    "samples_processed_by_api": 400,
    "estimated_cost_saving_percentage": 60.0,
    "api_usage_ratio": 0.4
  },
  "performance_metrics": {
    "average_api_response_time": 1.2,
    "total_api_time": 480.0,
    "api_efficiency": 0.83
  },
  "confidence_analysis": {
    "average_confidence": 0.78,
    "low_confidence_samples": 45,
    "low_confidence_percentage": 11.25
  }
}
```

## 🚀 Penggunaan

### Setup Environment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API**
   ```bash
   cp .env.template .env
   # Edit .env dengan DeepSeek API key
   ```

3. **Test Setup**
   ```bash
   python src/data_collection/deepseek_labeling_pipeline.py --mock --sample 10
   ```

### Production Usage

```bash
# Full dataset processing
python src/data_collection/deepseek_labeling_pipeline.py \
  --input raw-dataset.csv \
  --output labeled-dataset.csv

# Sample processing untuk testing
python src/data_collection/deepseek_labeling_pipeline.py \
  --input raw-dataset.csv \
  --output test-labeled.csv \
  --sample 100
```

## ⚠️ Considerations dan Limitations

### Assumptions

1. **Data Quality**: Pembagian awal positif/negatif sudah akurat
2. **Label Stability**: Data positif memang konsisten bukan ujaran kebencian
3. **API Reliability**: DeepSeek API memiliki uptime dan performance yang stabil

### Potential Issues

1. **False Positives**: Data yang salah dikategorikan sebagai "positive"
   - **Mitigasi**: Manual spot-check 5-10% data positif

2. **API Rate Limits**: Pembatasan dari DeepSeek API
   - **Mitigasi**: Implement rate limiting dan retry logic

3. **Cost Overrun**: Biaya API melebihi estimasi
   - **Mitigasi**: Monitor real-time usage dan set budget alerts

### Quality Assurance

1. **Validation Process**
   - Manual review untuk confidence score <0.6
   - Random sampling 10% untuk quality check
   - Cross-validation dengan human annotators

2. **Continuous Improvement**
   - Update prompt berdasarkan hasil
   - Refine filtering strategy
   - Monitor dan adjust thresholds

## 📈 Future Enhancements

### Short Term (1-2 weeks)

1. **Adaptive Thresholds**: Dynamic confidence thresholds berdasarkan performance
2. **Batch Optimization**: Optimize batch size untuk performance
3. **Error Recovery**: Better handling untuk API failures

### Medium Term (1-2 months)

1. **Active Learning**: Prioritize uncertain samples untuk manual review
2. **Multi-Model Ensemble**: Combine multiple API providers
3. **Real-time Monitoring**: Dashboard untuk monitoring live performance

### Long Term (3+ months)

1. **Custom Model**: Train model lokal untuk reduce API dependency
2. **Automated Quality Control**: AI-powered quality assurance
3. **Cost Prediction**: Predictive modeling untuk budget planning

## 📞 Support dan Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Test API connectivity
   python -c "from src.utils.deepseek_client import create_deepseek_client; client = create_deepseek_client(); print('API OK')"
   ```

2. **Memory Issues**
   - Reduce batch size di `settings.py`
   - Process data dalam chunks

3. **Performance Issues**
   - Check network connectivity
   - Monitor API response times
   - Adjust rate limiting settings

### Contact Information

- **Technical Issues**: Check logs di `logs/` directory
- **API Issues**: Refer to DeepSeek documentation
- **Strategy Questions**: Review this document dan project documentation

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Maintainer**: Ujaran Kebencian Detection Team