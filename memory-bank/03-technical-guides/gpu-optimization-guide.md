# Panduan Optimasi GPU untuk Training Model

## ✅ Status GPU

**GPU Terdeteksi**: NVIDIA GeForce RTX 4080  
**CUDA Version**: 11.8  
**PyTorch Version**: 2.7.1+cu118  
**Status**: ✅ Siap untuk training dengan GPU

## 🚀 Optimasi Training dengan GPU

### 1. Konfigurasi Otomatis

Kode training sudah dikonfigurasi untuk menggunakan GPU secara otomatis:

```python
# GPU Configuration (sudah ada di src/modelling/train_model.py)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Model dipindahkan ke GPU
```

### 2. Optimasi Batch Size untuk RTX 4080

Untuk RTX 4080 (16GB VRAM), batch size optimal:

```bash
# Training dengan batch size optimal
python train_model.py \
    --data_path ./hasil-labeling.csv \
    --output_dir ./models/hate_speech_model \
    --batch_size 32 \
    --eval_batch_size 64 \
    --epochs 3
```

### 3. Fitur GPU yang Diaktifkan

- **Mixed Precision (FP16)**: Mengurangi penggunaan memory hingga 50%
- **Pin Memory**: Mempercepat transfer data CPU-GPU
- **Parallel Data Loading**: 4 worker threads untuk loading data
- **Optimized Batch Size**: Otomatis disesuaikan berdasarkan GPU memory

### 4. Monitoring GPU

#### Monitoring Real-time
```bash
# Monitor penggunaan GPU selama training
nvidia-smi -l 1
```

#### Monitoring dalam Python
```python
import torch
print(f"GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
```

### 5. Estimasi Waktu Training

**Dengan RTX 4080**:
- Dataset 41,346 samples
- 3 epochs
- Batch size 32
- **Estimasi waktu**: 15-25 menit

**Perbandingan**:
- CPU Only: ~6-8 jam
- RTX 4080: ~15-25 menit
- **Speedup**: ~15-20x lebih cepat

### 6. Troubleshooting GPU

#### Out of Memory (OOM)
```bash
# Kurangi batch size
python train_model.py --batch_size 16 --eval_batch_size 32

# Atau gunakan gradient accumulation
python train_model.py --batch_size 8 --gradient_accumulation_steps 4
```

#### CUDA Error
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Restart training
python train_model.py --data_path ./hasil-labeling.csv
```

### 7. Optimasi Lanjutan

#### Gradient Checkpointing (untuk model besar)
```python
# Tambahkan di training arguments
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Mengurangi memory usage
    dataloader_pin_memory=True,
    fp16=True,
    ...
)
```

#### Dynamic Loss Scaling
```python
# Untuk mixed precision training yang lebih stabil
training_args = TrainingArguments(
    fp16=True,
    fp16_opt_level="O1",  # Optimasi level 1
    ...
)
```

## 🎯 Rekomendasi untuk RTX 4080

### Konfigurasi Optimal
```bash
python train_model.py \
    --data_path ./hasil-labeling.csv \
    --output_dir ./models/hate_speech_model \
    --epochs 3 \
    --batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_total_limit 2
```

### Memory Usage Estimation
- **Model Size**: ~440MB (IndoBERT-base)
- **Batch Size 32**: ~8-10GB VRAM
- **Available for RTX 4080**: 16GB
- **Safety Margin**: ~6GB tersisa

## 📊 Performance Monitoring

### GPU Utilization Target
- **GPU Utilization**: 85-95%
- **Memory Utilization**: 60-80%
- **Temperature**: <80°C

### Commands untuk Monitoring
```bash
# GPU stats
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

## ✅ Verifikasi Setup

Jalankan test berikut untuk memastikan GPU setup benar:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# Test GPU computation
device = torch.device('cuda')
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.mm(x, y)
print(f"GPU computation successful: {z.device}")
```

---

**Status**: ✅ GPU RTX 4080 siap untuk training  
**Estimasi Speedup**: 15-20x lebih cepat dari CPU  
**Rekomendasi**: Gunakan batch size 32 untuk performa optimal