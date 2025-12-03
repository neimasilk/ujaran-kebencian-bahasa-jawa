# Dokumentasi Setup GPU dan Eksperimen Training

## Ringkasan Eksekutif

**Tanggal:** 2025-01-02  
**Tim:** AI Assistant + User  
**Tujuan:** Setup GPU untuk training model hate speech detection Bahasa Jawa  
**Status:** ✅ Setup berhasil, eksperimen berjalan dengan GPU acceleration  
**Hardware:** NVIDIA GeForce RTX 3060 Ti (8GB VRAM)  
**Outcome:** GPU terdeteksi dan digunakan untuk training dengan mixed precision (FP16)

---

## 🎯 Konteks dan Tujuan

### Latar Belakang
- Proyek sistem deteksi ujaran kebencian Bahasa Jawa dalam fase eksperimen model training
- Baseline model IndoBERT telah dilatih sebelumnya dengan hasil accuracy 73.8%
- Diperlukan setup GPU untuk mempercepat eksperimen dan training model yang lebih besar
- Target: Meningkatkan performance model dari 73.8% ke >85% accuracy

### Hardware Environment
- **GPU:** NVIDIA GeForce RTX 3060 Ti
- **VRAM:** 8GB
- **OS:** Windows
- **Python:** 3.9+
- **CUDA:** 11.8 (setelah instalasi)

---

## 🔧 Proses Setup GPU

### 1. Deteksi Masalah Awal

**Masalah yang Ditemukan:**
```python
# Hasil check awal
CUDA available: False
CUDA device count: 0
No CUDA devices available
```

**Root Cause:** PyTorch terinstall tanpa CUDA support (CPU-only version)

### 2. Solusi: Instalasi PyTorch dengan CUDA Support

**Command yang Digunakan:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Packages yang Diinstall:**
- `torch-2.7.1+cu118`
- `torchaudio-2.7.1+cu118` 
- `torchvision-0.22.1+cu118`

**Hasil Setelah Instalasi:**
```python
# Hasil check setelah instalasi
CUDA available: True
CUDA device count: 1
Current CUDA device: 0
CUDA device name: NVIDIA GeForce RTX 3060 Ti
CUDA device memory: 8191 MB
```

### 3. Verifikasi GPU Detection

**Script Verifikasi (`check_gpu.py`):**
```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
    device = torch.cuda.get_device_properties(0)
    print(f"CUDA device memory: {device.total_memory // 1024 // 1024} MB")
else:
    print("No CUDA devices available")
```

**✅ Hasil:** GPU berhasil terdeteksi dan siap digunakan

---

## ⚡ Optimasi Eksperimen untuk GPU

### 1. Konfigurasi Training yang Dioptimasi

**File:** `experiment_1_simple.py`

**Parameter Optimasi:**
```python
# GPU-optimized configuration
BATCH_SIZE = 16                    # Increased from 8
GRADIENT_ACCUMULATION_STEPS = 1    # Decreased from 2  
LEARNING_RATE = 2e-5              # Increased from 1e-5
NUM_EPOCHS = 5                     # Increased from 3

# Training arguments optimizations
TrainingArguments(
    # ... other args ...
    dataloader_pin_memory=True,     # Enable pin memory for faster data transfer
    fp16=True,                      # Enable mixed precision training
    dataloader_num_workers=2,       # Parallel data loading
    # ... other args ...
)
```

### 2. GPU Detection dan Logging

**Implementasi Logging:**
```python
# GPU detection and logging
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
    print(f"🚀 GPU detected: {gpu_name} with {gpu_memory} MB memory")
    print(f"🔥 Mixed precision training: {'Enabled' if training_args.fp16 else 'Disabled'}")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected, using CPU")

# Move model to device
model = model.to(device)
print(f"📱 Model moved to: {device}")
```

### 3. Dependencies yang Diperlukan

**Instalasi Tambahan:**
```bash
# Accelerate untuk Trainer dengan PyTorch
pip install 'accelerate>=0.26.0'

# Transformers dengan torch support
pip install transformers[torch]
```

---

## 🧪 Hasil Eksperimen

### 1. Eksperimen Berhasil Dijalankan

**Status:** ✅ Eksperimen berjalan dengan GPU acceleration

**Log Output:**
```
🚀 GPU detected: NVIDIA GeForce RTX 3060 Ti with 8.0 GB memory
🔥 Mixed precision training: Enabled
📱 Model moved to: cuda:0

Loaded dataset with 41757 samples
Class distribution:
0: 8352 samples
1: 8352 samples  
2: 8352 samples
3: 16701 samples

Loading tokenizer and model: indobenchmark/indobert-large-p1
Model moved to CUDA device
Training started...
```

### 2. Konfigurasi Training yang Berhasil

**Model:** `indobenchmark/indobert-large-p1`  
**Dataset:** 41,757 samples (33,405 training, 8,352 test)  
**GPU Utilization:** NVIDIA GeForce RTX 3060 Ti dengan 8GB VRAM  
**Mixed Precision:** FP16 enabled  
**Batch Size:** 16  
**Workers:** 2 parallel data loading workers  

### 3. Performance Optimizations Applied

- ✅ **GPU Acceleration:** CUDA 11.8 support
- ✅ **Mixed Precision:** FP16 untuk memory efficiency
- ✅ **Optimized Batch Size:** 16 (optimal untuk 8GB VRAM)
- ✅ **Parallel Data Loading:** 2 workers
- ✅ **Pin Memory:** Enabled untuk faster data transfer
- ✅ **Gradient Accumulation:** Optimized untuk GPU memory

---

## 📊 Troubleshooting dan Lessons Learned

### 1. Masalah yang Ditemui dan Solusi

| Masalah | Penyebab | Solusi |
|---------|----------|--------|
| `CUDA available: False` | PyTorch CPU-only version | Install PyTorch dengan CUDA support |
| `ImportError: accelerate>=0.26.0` | Missing dependency | `pip install 'accelerate>=0.26.0'` |
| `SyntaxError: unterminated string` | Command syntax error | Create separate Python script |
| Model using CPU instead of GPU | Model not moved to device | Explicit `model.to(device)` |

### 2. Best Practices untuk Setup GPU

**✅ Do's:**
- Selalu verify CUDA availability sebelum training
- Install PyTorch dengan CUDA support yang sesuai
- Enable mixed precision (FP16) untuk memory efficiency
- Monitor GPU memory usage selama training
- Use appropriate batch size untuk GPU memory

**❌ Don'ts:**
- Jangan assume PyTorch sudah include CUDA support
- Jangan skip dependency installation (accelerate, etc.)
- Jangan use batch size terlalu besar untuk GPU memory
- Jangan lupa move model ke GPU device

### 3. Commands Reference

**Check GPU Status:**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Install PyTorch dengan CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Install Dependencies:**
```bash
pip install 'accelerate>=0.26.0'
pip install transformers[torch]
```

---

## 🎯 Rekomendasi untuk Tim

### 1. Untuk Setup GPU di Komputer Lain

**Checklist Setup:**
- [ ] Verify NVIDIA GPU dan driver terinstall
- [ ] Install PyTorch dengan CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- [ ] Install dependencies: `pip install 'accelerate>=0.26.0' transformers[torch]`
- [ ] Run verification script: `python check_gpu.py`
- [ ] Test dengan eksperimen kecil sebelum full training

### 2. Optimasi untuk Training Production

**Recommended Configuration:**
```python
# Untuk GPU 8GB (RTX 3060 Ti)
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5

# Untuk GPU 12GB+ (RTX 3080/4080)
BATCH_SIZE = 24
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
```

### 3. Monitoring dan Debugging

**GPU Memory Monitoring:**
```python
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
```

**Training Logs to Monitor:**
- GPU detection dan memory info
- Mixed precision status
- Batch size dan gradient accumulation
- Training progress dan loss
- Memory usage warnings

---

## 📝 Dokumentasi Terkait

**File Terkait:**
- `experiment_1_simple.py` - Script eksperimen dengan GPU optimization
- `check_gpu.py` - Script verifikasi GPU status
- `memory-bank/03-technical-guides/gpu-optimization-guide.md` - Panduan optimasi GPU
- `memory-bank/02-research-active/NEXT_EXPERIMENTS_PLAN.md` - Rencana eksperimen selanjutnya

**Dependencies:**
- PyTorch 2.7.1+cu118
- transformers[torch]
- accelerate>=0.26.0
- CUDA 11.8

---

**Status:** ✅ Dokumentasi lengkap  
**Next Steps:** Training di komputer lain dengan GPU yang lebih powerful  
**Contact:** Tim Development untuk setup assistance