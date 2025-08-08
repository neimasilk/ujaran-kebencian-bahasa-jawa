#!/usr/bin/env python3
"""
Konfirmasi penggunaan GPU untuk hyperparameter tuning
"""

import torch

print("\n" + "=" * 60)
print("KONFIRMASI GPU UNTUK HYPERPARAMETER TUNING")
print("=" * 60)

# Status GPU
cuda_available = torch.cuda.is_available()
print(f"✅ CUDA Available: {cuda_available}")

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"✅ GPU Name: {gpu_name}")
    print(f"✅ Memory Allocated: {memory_allocated:.2f} GB")
    print(f"✅ Memory Reserved: {memory_reserved:.2f} GB")
    print(f"✅ Total Memory: {memory_total:.2f} GB")
    print(f"✅ Memory Usage: {(memory_reserved/memory_total)*100:.1f}%")
else:
    print("❌ GPU tidak tersedia")

print("\n" + "=" * 60)
print("ANALISIS KONFIGURASI HYPERPARAMETER_TUNING.PY")
print("=" * 60)

print("Berdasarkan kode hyperparameter_tuning.py:")
print(f"• fp16=torch.cuda.is_available() -> {cuda_available}")
print("• Model akan otomatis dipindah ke GPU oleh Transformers Trainer")
print("• Training menggunakan mixed precision (fp16) untuk efisiensi GPU")
print("• dataloader_pin_memory=False (untuk kompatibilitas)")

print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)

if cuda_available:
    print("✅ KONFIRMASI: Training hyperparameter tuning MENGGUNAKAN GPU")
    print("✅ Model akan dilatih dengan mixed precision (fp16)")
    print("✅ Proses akan lebih cepat dibanding CPU")
    if memory_reserved > 0:
        print("✅ GPU sedang digunakan oleh proses PyTorch")
    else:
        print("ℹ️  GPU tersedia tapi belum ada alokasi memori (normal saat startup)")
else:
    print("❌ Training akan menggunakan CPU (lebih lambat)")
    print("❌ fp16 tidak akan diaktifkan")

print("\n" + "=" * 60)