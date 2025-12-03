# 🚀 Panduan Eksperimen Ensemble Otomatis

## 📋 Overview
Sistem otomatis untuk menjalankan beberapa eksperimen ensemble secara berurutan **tanpa intervensi manual**. Anda bisa menjalankannya dan meninggalkannya hingga selesai.

## 🎯 Fitur Utama

### ✅ Yang Bisa Dilakukan Sistem Ini
- **Menjalankan 3 eksperimen ensemble** secara berurutan
- **Fine-tune setiap model** dengan dataset Javanese hate speech
- **Monitoring real-time** progress dan resource usage
- **Auto-save checkpoints** setiap 200 steps
- **Error handling** dan recovery otomatis
- **Comprehensive logging** semua aktivitas
- **Final results summary** dengan metrics lengkap
- **Notifikasi completion** (Windows)

### 🔧 3 Eksperimen yang Akan Dijalankan

1. **Baseline Ensemble** (30 menit)
   - 3 model: IndoBERT, IndoBERT-uncased, RoBERTa-Indo
   - 3 epochs fine-tuning
   - Equal weights ensemble

2. **Extended Ensemble** (50 menit)
   - Same 3 models
   - **5 epochs** fine-tuning (lebih thorough)
   - Advanced optimization
   - Early stopping

3. **Augmented Ensemble** (40 menit)
   - Same 3 models
   - **Data augmentation** during training
   - Enhanced text preprocessing

**Total Estimated Time: ~2 jam**

## 🚀 Cara Menjalankan

### Option 1: Jalankan Semua Eksperimen (Recommended)
```bash
# Jalankan semua eksperimen dengan monitoring
python run_all_experiments.py --monitor
```

### Option 2: Eksperimen Cepat Saja
```bash
# Hanya baseline ensemble (30 menit)
python run_all_experiments.py --quick --monitor
```

### Option 3: Manual Control
```bash
# Jalankan automated experiments saja
python automated_ensemble_experiments.py

# Di terminal terpisah, jalankan monitoring
python monitor_experiments.py
```

## 📊 Monitoring Real-Time

### Automatic Monitoring
Ketika menggunakan `--monitor`, sistem akan:
- **Update setiap 30 detik** dengan progress terbaru
- **Show GPU/CPU usage** dan memory consumption
- **Display latest log entries** dari setiap eksperimen
- **Track model checkpoints** yang tersimpan
- **Estimate remaining time** berdasarkan progress

### Manual Monitoring
```bash
# Check sekali saja
python monitor_experiments.py --once

# Monitoring kontinyu (update setiap 60 detik)
python monitor_experiments.py --interval 60
```

## 📁 Output Structure

### Setelah Eksperimen Selesai
```
results/
├── automated_experiments_20250807_143022.json  # Main results
├── multi_architecture_ensemble_results.json    # Baseline results
├── extended_ensemble_results.json              # Extended results
└── augmentation_ensemble_results.json          # Augmented results

models/
├── ensemble_indobert/                          # Baseline models
├── ensemble_indobert_uncased/
├── ensemble_roberta_indo/
├── extended_ensemble_indobert/                 # Extended models
├── extended_ensemble_indobert_uncased/
├── extended_ensemble_roberta_indo/
├── augmented_ensemble_indobert/                # Augmented models
├── augmented_ensemble_indobert_uncased/
└── augmented_ensemble_roberta_indo/

logs/
├── experiment_launcher.log                     # Main launcher log
├── automated_experiments.log                   # Automated runner log
├── multi_architecture_ensemble.log             # Baseline experiment
├── extended_ensemble.log                       # Extended experiment
└── augmentation_ensemble.log                   # Augmented experiment
```

## 📈 Expected Results

### Target Performance
- **Individual Models**: 40-65% accuracy
- **Baseline Ensemble**: 60-70% accuracy
- **Extended Ensemble**: 65-75% accuracy (5 epochs)
- **Augmented Ensemble**: 65-75% accuracy (with augmentation)
- **Best Ensemble**: >75% accuracy

### Sample Results Format
```json
{
  "experiment_suite": "automated_ensemble_experiments",
  "total_duration_hours": 2.1,
  "summary": {
    "total_experiments": 3,
    "successful": 3,
    "failed": 0
  },
  "experiments": {
    "baseline_ensemble": {
      "status": "success",
      "duration_minutes": 28.5,
      "best_accuracy": 0.672
    },
    "extended_ensemble": {
      "status": "success", 
      "duration_minutes": 47.2,
      "best_accuracy": 0.738
    },
    "augmented_ensemble": {
      "status": "success",
      "duration_minutes": 39.8,
      "best_accuracy": 0.701
    }
  }
}
```

## 🛡️ Safety Features

### Auto-Recovery
- **Checkpoint saving** setiap 200 steps
- **Best model preservation** berdasarkan F1-score
- **Error logging** dengan full traceback
- **Resource monitoring** untuk prevent crashes

### Interruption Handling
- **Ctrl+C** akan stop gracefully
- **Power outage recovery** via checkpoints
- **Resume capability** dari checkpoint terakhir

### Resource Management
- **GPU memory optimization** dengan FP16
- **Batch size auto-adjustment** jika OOM
- **Disk space monitoring** 
- **Process isolation** antar eksperimen

## 🔧 Troubleshooting

### Jika Eksperimen Gagal
```bash
# Check logs untuk error details
tail -f logs/automated_experiments.log

# Check specific experiment logs
tail -f logs/multi_architecture_ensemble.log

# Check system resources
python monitor_experiments.py --once
```

### Common Issues

1. **GPU Out of Memory**
   - Script akan auto-reduce batch size
   - Atau gunakan CPU: set `CUDA_VISIBLE_DEVICES=""`

2. **Disk Space Full**
   - Monitor akan warn jika <5GB free
   - Clean old checkpoints: `rm -rf models/old_*`

3. **Network Issues (Model Download)**
   - Models akan di-cache setelah download pertama
   - Restart akan resume dari cache

4. **Process Hanging**
   - Check dengan: `python monitor_experiments.py --once`
   - Kill manual: `taskkill /f /im python.exe` (Windows)

## ⚡ Performance Tips

### Untuk Eksperimen Lebih Cepat
```bash
# Reduce epochs di script (edit training_args)
num_train_epochs=2  # instead of 3

# Increase batch size jika GPU memory cukup
per_device_train_batch_size=32  # instead of 16

# Use smaller subset
test_size=0.6  # instead of 0.4 (less training data)
```

### Untuk Hasil Lebih Baik
```bash
# Increase epochs
num_train_epochs=5

# Lower learning rate
learning_rate=1e-5  # instead of 2e-5

# More evaluation steps
eval_steps=100  # instead of 200
```

## 📱 Notifications

### Windows Notification
Sistem akan show popup notification ketika:
- ✅ Semua eksperimen selesai
- ❌ Ada eksperimen yang gagal
- ⏰ Timeout detected

### Email Notification (Optional)
Tambahkan di `monitor_experiments.py`:
```python
# Add email notification function
def send_email_notification(subject, message):
    # Implement email sending logic
    pass
```

## 🎯 Next Steps Setelah Eksperimen

### 1. Analyze Results
```bash
# Check main results
cat results/automated_experiments_*.json

# Compare all experiments
python analyze_experiment_results.py
```

### 2. Deploy Best Model
```bash
# Use best performing ensemble
python production_deployment.py --model extended_ensemble
```

### 3. Further Optimization
- **Hyperparameter tuning** pada best model
- **Cross-validation** untuk robust evaluation
- **Additional architectures** (XLM-RoBERTa, mBERT)
- **Advanced ensemble methods** (stacking, meta-learner)

## 🚀 Quick Start Commands

```bash
# 1. Pastikan environment ready
python check_gpu.py

# 2. Jalankan semua eksperimen (recommended)
python run_all_experiments.py --monitor

# 3. Tunggu hingga selesai (~2 jam)
# 4. Check results
cat results/automated_experiments_*.json

# 5. Deploy best model
python production_deployment.py
```

---

## 🎉 Success Criteria

### Minimum Success
- ✅ Semua 3 eksperimen completed
- ✅ At least 1 ensemble >70% accuracy
- ✅ All models properly saved

### Optimal Success  
- 🎯 Best ensemble >75% accuracy
- 🎯 Extended ensemble outperforms baseline
- 🎯 Consistent results across experiments

### Stretch Goal
- 🚀 Best ensemble >80% accuracy
- 🚀 Production-ready deployment
- 🚀 Research paper quality results

**Selamat menjalankan eksperimen! 🚀**