# Strategi Optimasi Biaya DeepSeek API

## 📋 Overview

Dokumen ini menjelaskan strategi optimasi biaya untuk labeling dataset menggunakan DeepSeek API berdasarkan jam operasional dan periode diskon.

## 💰 Struktur Harga DeepSeek API

### Model yang Tersedia
- **deepseek-chat**: Model utama untuk chat dan labeling
- **deepseek-reasoner**: Model dengan reasoning capability

### Periode Harga

#### 🕐 Standard Price (UTC 00:30-16:30)
**GMT+7: 07:30-23:30 WIB**

| Komponen | deepseek-chat | deepseek-reasoner |
|----------|---------------|-------------------|
| Input (Cache Hit) | $0.07/1M tokens | $0.14/1M tokens |
| Input (Cache Miss) | $0.27/1M tokens | $0.55/1M tokens |
| Output | $1.10/1M tokens | $2.19/1M tokens |

#### 🕐 Discount Price (UTC 16:30-00:30)
**GMT+7: 23:30-07:30 WIB**

| Komponen | deepseek-chat | deepseek-reasoner |
|----------|---------------|-------------------|
| Input (Cache Hit) | $0.035/1M tokens (50% OFF) | $0.035/1M tokens (75% OFF) |
| Input (Cache Miss) | $0.135/1M tokens (50% OFF) | $0.135/1M tokens (75% OFF) |
| Output | $0.550/1M tokens (50% OFF) | $0.550/1M tokens (75% OFF) |

## 🎯 Strategi Optimasi

### 1. **Discount Only Strategy** (`discount_only`)
- **Deskripsi**: Hanya memproses saat periode diskon aktif
- **Keuntungan**: Penghematan maksimal 50% biaya
- **Kekurangan**: Waktu processing terbatas (8 jam per hari)
- **Rekomendasi**: Untuk dataset besar dengan budget terbatas

```python
# Contoh penggunaan
pipeline = PersistentLabelingPipeline(
    input_file="dataset.csv",
    output_file="results.csv",
    cost_strategy="discount_only"
)
```

### 2. **Always Strategy** (`always`)
- **Deskripsi**: Memproses kapan saja, dengan peringatan saat harga mahal
- **Keuntungan**: Fleksibilitas waktu maksimal
- **Kekurangan**: Biaya lebih tinggi saat periode standar
- **Rekomendasi**: Untuk deadline ketat atau dataset kecil

```python
# Contoh penggunaan
pipeline = PersistentLabelingPipeline(
    input_file="dataset.csv",
    output_file="results.csv",
    cost_strategy="always"
)
```

### 3. **Warn Expensive Strategy** (`warn_expensive`) - **DEFAULT**
- **Deskripsi**: Memproses kapan saja dengan peringatan biaya tinggi
- **Keuntungan**: Balance antara fleksibilitas dan awareness biaya
- **Kekurangan**: Memerlukan monitoring manual
- **Rekomendasi**: Untuk sebagian besar use case

```python
# Contoh penggunaan (default)
pipeline = PersistentLabelingPipeline(
    input_file="dataset.csv",
    output_file="results.csv",
    cost_strategy="warn_expensive"  # atau tidak perlu ditulis
)
```

## 🕒 Jadwal Optimal (GMT+7)

### Periode Diskon (50% OFF)
- **Waktu**: 23:30 - 07:30 WIB (8 jam)
- **Hari**: Setiap hari
- **Rekomendasi**: Jadwalkan processing otomatis pada jam ini

### Periode Standar (Harga Normal)
- **Waktu**: 07:30 - 23:30 WIB (16 jam)
- **Hari**: Setiap hari
- **Rekomendasi**: Hindari jika menggunakan strategi `discount_only`

## 📊 Estimasi Penghematan

### Dataset: 41,759 Samples
- **Input Tokens**: ~8.35M tokens
- **Output Tokens**: ~3.13M tokens
- **Cache Hit Ratio**: ~10%

### Perbandingan Biaya

| Periode | Input Cost | Output Cost | Total Cost | Penghematan |
|---------|------------|-------------|------------|-------------|
| Standard | $2.26 | $3.43 | **$5.69** | - |
| Discount | $1.13 | $1.72 | **$2.85** | **$2.84 (50%)** |

## 🚀 Implementasi Otomatis

### Cost Optimizer Class

Sistem menggunakan `CostOptimizer` class yang secara otomatis:

1. **Deteksi Periode**: Mendeteksi periode diskon/standar berdasarkan UTC
2. **Monitoring Real-time**: Memantau perubahan periode selama processing
3. **Estimasi Biaya**: Menghitung biaya real-time per batch
4. **Auto-pause**: Menunggu periode diskon jika menggunakan strategi `discount_only`
5. **Laporan Penghematan**: Melacak total penghematan vs harga standar

### Fitur Monitoring

```python
# Cek status saat ini
optimizer = CostOptimizer()
status = optimizer.get_status_report()
print(f"Periode: {status['pricing']['current_period']}")
print(f"Diskon aktif: {status['pricing']['is_discount']}")

# Estimasi biaya
cost_info = optimizer.calculate_cost(
    input_tokens=1000000,
    output_tokens=500000
)
print(f"Estimasi biaya: ${cost_info['total_cost']:.2f}")
```

## 📈 Monitoring dan Logging

### Log Messages

Sistem akan menampilkan log seperti:

```
💰 ✅ Periode diskon aktif: Discount (UTC 16:30-00:30)
💰 Batch cost: $0.0142 (Discount (UTC 16:30-00:30))
💰 Total cost so far: $0.0567
💰 Savings in this batch: $0.0142
💰 Total savings so far: $0.0567
```

### Statistik Akhir

Laporan akhir akan mencakup:
- Total biaya per periode
- Total penghematan
- Breakdown biaya per jam
- Efisiensi strategi yang dipilih

## 🎯 Rekomendasi Best Practices

### 1. **Untuk Dataset Besar (>10K samples)**
- Gunakan strategi `discount_only`
- Jadwalkan processing malam hari (23:30-07:30 WIB)
- Gunakan batch size besar (50-100) untuk efisiensi

### 2. **Untuk Dataset Sedang (1K-10K samples)**
- Gunakan strategi `warn_expensive` (default)
- Monitor log untuk optimasi timing
- Pertimbangkan split processing antara periode diskon/standar

### 3. **Untuk Dataset Kecil (<1K samples)**
- Gunakan strategi `always`
- Fokus pada kecepatan daripada optimasi biaya
- Biaya total relatif kecil

### 4. **Untuk Production Environment**
- Implementasikan cron job untuk auto-start saat periode diskon
- Setup monitoring alerts untuk perubahan biaya
- Backup checkpoint secara berkala

## 🔧 Konfigurasi Advanced

### Environment Variables

```bash
# Set timezone (optional, default: Asia/Jakarta)
TIMEZONE=Asia/Jakarta

# Set default cost strategy
DEFAULT_COST_STRATEGY=discount_only

# Set monitoring interval (seconds)
COST_CHECK_INTERVAL=300
```

### Custom Scheduling

```python
# Jadwal otomatis dengan cron-like syntax
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

# Jadwalkan setiap hari jam 23:30 WIB
scheduler.add_job(
    func=run_labeling_pipeline,
    trigger="cron",
    hour=23,
    minute=30,
    timezone="Asia/Jakarta"
)

scheduler.start()
```

## 📞 Troubleshooting

### Issue: Pipeline tidak menunggu periode diskon
**Solusi**: Pastikan strategi diset ke `discount_only` dan tidak dalam mock mode

### Issue: Biaya tidak terhitung dengan benar
**Solusi**: Periksa token counting dan pastikan cache hit ratio sesuai

### Issue: Timezone tidak sesuai
**Solusi**: Set timezone eksplisit saat inisialisasi `CostOptimizer(timezone="Asia/Jakarta")`

## 📚 Referensi

- [DeepSeek API Pricing](https://platform.deepseek.com/api-docs/pricing)
- [DeepSeek Context Caching](https://platform.deepseek.com/api-docs/context-caching)
- [Timezone Converter](https://www.timeanddate.com/worldclock/converter.html)

---

**Catatan**: Harga dan kebijakan dapat berubah. Selalu cek dokumentasi resmi DeepSeek untuk informasi terbaru.