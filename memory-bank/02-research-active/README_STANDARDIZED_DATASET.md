# Dataset Terstandarisasi - Ujaran Kebencian Bahasa Jawa

## Overview

Dataset ujaran kebencian bahasa Jawa telah berhasil distandardisasi dan dibalance untuk eksperimen machine learning. Dataset asli yang memiliki ketidakseimbangan distribusi label telah diproses menjadi dataset yang seimbang dan siap untuk training model.

## Struktur Dataset

### Dataset Files
```
data/standardized/
├── balanced_dataset.csv     # Dataset lengkap yang sudah dibalance (24,964 sampel)
├── train_dataset.csv        # Dataset training (19,971 sampel - 80%)
└── test_dataset.csv         # Dataset testing (4,993 sampel - 20%)
```

### Format Dataset
Setiap file CSV memiliki kolom:
- `text`: Teks ujaran dalam bahasa Jawa
- `final_label`: Label kategorikal (4 kelas)
- `label_numeric`: Label numerik (0-3)
- `label_binary`: Label biner (0=bukan ujaran kebencian, 1=ujaran kebencian)

### Distribusi Label

| Label | Kategori | Jumlah | Persentase |
|-------|----------|--------|------------|
| 0 | Bukan Ujaran Kebencian | 6,241 | 25.0% |
| 1 | Ujaran Kebencian - Ringan | 6,241 | 25.0% |
| 2 | Ujaran Kebencian - Sedang | 6,241 | 25.0% |
| 3 | Ujaran Kebencian - Berat | 6,241 | 25.0% |

## Cara Menggunakan Dataset

### 1. Loading Dataset

```python
import pandas as pd

# Load training dataset
train_df = pd.read_csv('data/standardized/train_dataset.csv')
test_df = pd.read_csv('data/standardized/test_dataset.csv')

# Extract features and labels
X_train = train_df['text'].values
y_train = train_df['label_numeric'].values

X_test = test_df['text'].values
y_test = test_df['label_numeric'].values
```

### 2. Contoh Penggunaan dengan Transformers

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model dan tokenizer
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=4  # 4 kelas
)

# Tokenize data
train_encodings = tokenizer(
    list(X_train), 
    truncation=True, 
    padding=True, 
    max_length=512
)
```

## Scripts yang Tersedia

### 1. Analisis dan Balancing Dataset
```bash
python analyze_dataset_balance.py
```
- Menganalisis distribusi dataset asli
- Melakukan undersampling untuk balancing
- Membuat train-test split yang stratified
- Menghasilkan dataset terstandarisasi

### 2. Verifikasi Dataset
```bash
python verify_balanced_dataset.py
```
- Memverifikasi distribusi label
- Menampilkan statistik dataset
- Validasi konsistensi data
- Menampilkan contoh data per kategori

### 3. Eksperimen Baseline
```bash
python experiment_standardized_baseline.py
```
- Menjalankan eksperimen baseline dengan IndoBERT
- Menggunakan dataset terstandarisasi
- Menghasilkan evaluasi lengkap
- Menyimpan hasil dan visualisasi

## Eksperimen yang Dapat Dilakukan

### 1. Baseline Experiments
- **IndoBERT Base**: `experiment_standardized_baseline.py`
- **IndoBERT Large**: Modifikasi script untuk model large
- **XLM-RoBERTa**: Ganti model_name ke XLM-RoBERTa
- **mBERT**: Ganti model_name ke mBERT

### 2. Advanced Experiments
- **Multi-stage Fine-tuning**: Training bertahap
- **Custom Loss Functions**: Focal Loss, Class-weighted Loss
- **Data Augmentation**: Back-translation, paraphrasing
- **Ensemble Methods**: Voting, stacking

## Konfigurasi Training

### Recommended Settings
```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True
)
```

### Hardware Requirements
- **Minimum**: 8GB RAM, GPU dengan 6GB VRAM
- **Recommended**: 16GB RAM, GPU dengan 12GB VRAM
- **Training Time**: ~30-60 menit per eksperimen (tergantung hardware)

## Evaluasi Metrics

### Primary Metrics
- **F1-Score Macro**: Metric utama untuk dataset seimbang
- **Accuracy**: Akurasi keseluruhan
- **Precision/Recall Macro**: Per-class performance

### Target Performance
- **Baseline Target**: F1-Score Macro > 80%
- **Advanced Target**: F1-Score Macro > 85%
- **Publication Target**: F1-Score Macro > 90%

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Kurangi batch_size
   - Gunakan gradient_accumulation_steps
   - Kurangi max_length tokenizer

2. **Slow Training**
   - Gunakan mixed precision training
   - Optimalkan DataLoader num_workers
   - Gunakan gradient checkpointing

3. **Poor Performance**
   - Cek distribusi data
   - Adjust learning rate
   - Tambah warmup steps
   - Gunakan class weights

### Memory Optimization
```python
# Untuk GPU dengan memory terbatas
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # Kurangi batch size
    gradient_accumulation_steps=2,   # Simulasi batch size 16
    fp16=True,                      # Mixed precision
    dataloader_num_workers=0,       # Kurangi memory overhead
    gradient_checkpointing=True     # Trade compute for memory
)
```

## Next Steps

1. **Jalankan Baseline**: Mulai dengan `experiment_standardized_baseline.py`
2. **Analisis Results**: Review confusion matrix dan per-class metrics
3. **Optimize Hyperparameters**: Tuning learning rate, batch size, epochs
4. **Advanced Techniques**: Implement focal loss, data augmentation
5. **Ensemble Methods**: Combine multiple models

## Dokumentasi Lengkap

Untuk dokumentasi lengkap tentang proses standardisasi, lihat:
- `memory-bank/01-project-core/DATASET_STANDARDIZATION_COMPLETE.md`
- `memory-bank/02-research-active/NEXT_EXPERIMENTS_PLAN.md`

---

**Status**: ✅ Dataset siap untuk eksperimen  
**Last Updated**: 2024-12-19  
**Version**: 1.0