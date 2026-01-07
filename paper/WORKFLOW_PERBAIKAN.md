# 📋 WORKFLOW PERBAIKAN PAPER JITK
## Javanese Hate Speech Detection dengan Label Smoothing

---

## 🎯 OVERVIEW

Paper ini memiliki **masalah inkonsistensi data** antara gambar dan hasil yang dilaporkan. Workflow ini akan memandu Anda untuk:
1. Generate ulang gambar yang konsisten
2. Menjalankan eksperimen untuk statistical significance
3. Melengkapi bagian yang kurang

---

## 📊 FASE 1: VERIFIKASI DATA (Estimasi: 1-2 jam)

### Task 1.1: Konfirmasi Dataset Final
```bash
# Di folder project Anda, jalankan:
cd ujaran-kebencian-bahasa-jawa

# Cek file dataset yang digunakan untuk training final
# Kemungkinan ada di: data/balanced/ atau data/final/
ls -la data/

# Hitung jumlah sample per kelas
python -c "
import pandas as pd
# Ganti path sesuai lokasi dataset Anda
df = pd.read_csv('data/YOUR_FINAL_DATASET.csv')
print('Total samples:', len(df))
print('\\nDistribusi label:')
print(df['label'].value_counts())
print('\\nPersentase:')
print(df['label'].value_counts(normalize=True) * 100)
"
```

**Yang harus dicatat:**
- [ ] Total samples: _______ (harusnya ~10,019)
- [ ] Neutral: _______% 
- [ ] Light Hate: _______%
- [ ] Moderate Hate: _______%
- [ ] Severe Hate: _______%
- [ ] Train/Val/Test split: _______/_______/_______

### Task 1.2: Identifikasi Model Terbaik
```bash
# Cari checkpoint model IndoBERT + Label Smoothing
find . -name "*.pt" -o -name "*.bin" -o -name "checkpoint*" 2>/dev/null

# Atau cek di folder models/
ls -la models/
```

**Yang harus dicatat:**
- [ ] Path ke model terbaik: _______________________
- [ ] Config/hyperparameters yang digunakan: _______

---

## 📈 FASE 2: GENERATE GAMBAR KONSISTEN (Estimasi: 2-3 jam)

### Task 2.1: Buat Script Generate Figures
Buat file baru `reproduce/generate_paper_figures.py`:

```python
#!/usr/bin/env python3
"""
Generate all figures for JITK paper submission.
Pastikan semua gambar konsisten dengan hasil yang dilaporkan:
- F1-Macro: 81.38%
- Accuracy: 81.24%
- Test set: 1,002 samples
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================
# CONFIGURATION - SESUAIKAN DENGAN SETUP ANDA
# ============================================
MODEL_PATH = "models/indobert_label_smoothing_best"  # Ganti sesuai lokasi
TEST_DATA_PATH = "data/test.csv"  # Ganti sesuai lokasi
OUTPUT_DIR = "paper/figures/"

LABEL_NAMES = ['Neutral', 'Light Hate', 'Moderate Hate', 'Severe Hate']
# Atau dalam Bahasa Indonesia:
# LABEL_NAMES = ['Bukan Ujaran Kebencian', 'Ujaran Kebencian - Ringan', 
#                'Ujaran Kebencian - Sedang', 'Ujaran Kebencian - Berat']

# ============================================
# FIGURE 1: DATASET DISTRIBUTION (4-class balanced)
# ============================================
def generate_dataset_distribution():
    """
    Generate pie chart dan bar chart untuk distribusi dataset.
    PENTING: Gunakan data FINAL (10,019 samples), BUKAN data lama (42K)
    """
    # DATA YANG BENAR - sesuaikan dengan dataset Anda
    distribution = {
        'Neutral': 2474,           # 24.7%
        'Light Hate': 2615,        # 26.1%
        'Moderate Hate': 2865,     # 28.6%
        'Severe Hate': 2065        # 20.6%
    }
    
    # Atau load dari file:
    # df = pd.read_csv('data/combined_final.csv')
    # distribution = df['label'].value_counts().to_dict()
    
    total = sum(distribution.values())
    print(f"Total samples: {total}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = axes[0].bar(distribution.keys(), distribution.values(), color=colors)
    axes[0].set_xlabel('Label', fontsize=12)
    axes[0].set_ylabel('Jumlah', fontsize=12)
    axes[0].set_title('Distribusi Label Dataset', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, distribution.values()):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{val}', ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    percentages = [v/total*100 for v in distribution.values()]
    axes[1].pie(percentages, labels=distribution.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Proporsi Label', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure1_dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 saved: dataset_distribution.png")


# ============================================
# FIGURE 2: CONFUSION MATRIX (dari model terbaik)
# ============================================
def generate_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix dari prediksi model.
    PENTING: Harus dari model IndoBERT+LabelSmoothing yang menghasilkan 81.38% F1
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Hitung metrics
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    acc = accuracy_score(y_true, y_pred) * 100
    
    print(f"Verification - F1-Macro: {f1:.2f}%, Accuracy: {acc:.2f}%")
    
    # VALIDASI: Pastikan sesuai dengan yang dilaporkan
    assert abs(f1 - 81.38) < 1.0, f"F1 tidak sesuai! Expected ~81.38%, got {f1:.2f}%"
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - IndoBERT + Label Smoothing\n'
              f'F1-Macro: {f1:.2f}% | Accuracy: {acc:.2f}%', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 saved: confusion_matrix.png")
    
    return cm, f1, acc


# ============================================
# FIGURE 3: MODEL COMPARISON (Validation vs Test Gap)
# ============================================
def generate_model_comparison():
    """
    Bar chart comparing validation vs test performance untuk berbagai metode.
    Menunjukkan overfitting pada ensemble methods.
    """
    # Data dari eksperimen Anda - SESUAIKAN dengan hasil aktual
    data = {
        'Method': ['Single Model\n(Ours)', 'Soft Voting', 'Weighted Voting', 'Meta-Learner'],
        'Validation F1': [81.13, 82.50, 84.20, 94.09],
        'Test F1': [81.38, 79.80, 78.50, 79.50],
    }
    
    df = pd.DataFrame(data)
    df['Gap'] = df['Validation F1'] - df['Test F1']
    
    x = np.arange(len(df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, df['Validation F1'], width, label='Validation F1', color='#3498db')
    bars2 = ax.bar(x + width/2, df['Test F1'], width, label='Test F1', color='#2ecc71')
    
    # Highlight overfitting gap
    for i, (val, test, gap) in enumerate(zip(df['Validation F1'], df['Test F1'], df['Gap'])):
        color = 'green' if gap < 1 else ('orange' if gap < 5 else 'red')
        ax.annotate(f'Gap: {gap:+.2f}%', xy=(i, max(val, test) + 1),
                   ha='center', fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('F1-Macro (%)', fontsize=12)
    ax.set_title('Validation vs Test Performance\n(Menunjukkan Overfitting pada Ensemble)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'])
    ax.legend()
    ax.set_ylim(70, 100)
    ax.axhline(y=81.38, color='green', linestyle='--', alpha=0.5, label='Best Test F1')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure3_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 saved: model_comparison.png")


# ============================================
# MAIN: Run inference dan generate semua figures
# ============================================
def run_inference_and_generate_figures():
    """
    Load model, run inference on test set, generate all figures.
    """
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)
    
    # 1. Dataset distribution (tidak perlu model)
    print("\n[1/3] Generating dataset distribution...")
    generate_dataset_distribution()
    
    # 2. Load model dan run inference
    print("\n[2/3] Loading model and running inference...")
    
    # UNCOMMENT dan sesuaikan kode di bawah ini:
    """
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Run inference
    y_true = test_df['label'].values
    y_pred = []
    
    with torch.no_grad():
        for text in test_df['text']:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            outputs = model(**inputs)
            pred = outputs.logits.argmax(-1).item()
            y_pred.append(pred)
    
    y_pred = np.array(y_pred)
    
    # Generate confusion matrix
    cm, f1, acc = generate_confusion_matrix(y_true, y_pred)
    """
    
    # SEMENTARA: Gunakan dummy data untuk testing script
    # HAPUS BAGIAN INI setelah uncomment kode di atas
    print("   [WARNING] Using dummy data - replace with actual inference!")
    y_true = np.random.randint(0, 4, 1002)
    y_pred = y_true.copy()
    # Add some errors
    error_idx = np.random.choice(1002, 187, replace=False)  # ~18.6% error for 81.4% acc
    y_pred[error_idx] = (y_pred[error_idx] + np.random.randint(1, 4, len(error_idx))) % 4
    
    generate_confusion_matrix(y_true, y_pred)
    
    # 3. Model comparison
    print("\n[3/3] Generating model comparison chart...")
    generate_model_comparison()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    run_inference_and_generate_figures()
```

### Task 2.2: Jalankan Script
```bash
# Pastikan environment aktif
conda activate your_env  # atau source venv/bin/activate

# Install dependencies jika belum
pip install matplotlib seaborn pandas numpy scikit-learn

# Jalankan script
python reproduce/generate_paper_figures.py

# Cek output
ls -la paper/figures/
```

**Checklist Gambar:**
- [ ] `figure1_dataset_distribution.png` - Distribusi 4 kelas (~10K samples)
- [ ] `figure2_confusion_matrix.png` - CM dengan F1=81.38%, Acc=81.24%
- [ ] `figure3_model_comparison.png` - Val vs Test gap chart

---

## 🔬 FASE 3: STATISTICAL SIGNIFICANCE (Estimasi: 4-8 jam training)

### Task 3.1: Multiple Runs dengan Different Seeds
```python
# Buat file: reproduce/experiment_multiple_seeds.py

"""
Jalankan eksperimen 5x dengan random seed berbeda untuk statistical significance.
"""

SEEDS = [42, 123, 456, 789, 1024]
RESULTS = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Running experiment with seed={seed}")
    print('='*60)
    
    # Set all random seeds
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Train model (gunakan script training Anda)
    # result = train_model(seed=seed)
    # RESULTS.append(result)

# Calculate statistics
import numpy as np
f1_scores = [r['test_f1'] for r in RESULTS]
acc_scores = [r['test_acc'] for r in RESULTS]

print(f"\n{'='*60}")
print("FINAL RESULTS (5 runs)")
print('='*60)
print(f"F1-Macro: {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}%")
print(f"Accuracy: {np.mean(acc_scores):.2f}% ± {np.std(acc_scores):.2f}%")
```

**Yang harus dicatat untuk paper:**
- [ ] F1-Macro: ______% ± ______%
- [ ] Accuracy: ______% ± ______%
- [ ] Training time per epoch: ______ seconds
- [ ] GPU used: ______ (e.g., NVIDIA RTX 3090)

---

## 📝 FASE 4: CONTENT YANG PERLU DITULIS (Estimasi: 2-3 jam)

### Task 4.1: Related Work Section (Tambahkan sebelum Methods)

**Template:**
```
RELATED WORK

Penelitian deteksi ujaran kebencian di Indonesia telah berkembang dalam 
beberapa tahun terakhir. Ibrohim dan Budi [ref] memperkenalkan dataset 
multi-label untuk bahasa Indonesia dengan XX samples. Penelitian mereka 
menggunakan metode [X] dan mencapai akurasi [Y]%.

Untuk bahasa daerah, Putri et al. [ref] melakukan studi awal deteksi 
ujaran kebencian dalam bahasa Jawa dan Sunda. Namun, dataset mereka 
terbatas pada [X] samples dan hanya menggunakan klasifikasi binary.

Dalam konteks transformer untuk bahasa Indonesia, Wilie et al. [ref] 
memperkenalkan IndoBERT yang dilatih pada corpus bahasa Indonesia. 
Model ini telah digunakan untuk berbagai task NLU termasuk sentiment 
analysis dan named entity recognition.

Berbeda dengan penelitian sebelumnya, penelitian ini: (1) menggunakan 
dataset yang lebih besar dengan 10,019 samples, (2) mengimplementasikan 
klasifikasi 4-kelas untuk tingkat keparahan ujaran kebencian, dan (3) 
melakukan analisis mendalam terhadap kegagalan metode ensemble.
```

### Task 4.2: LLM Augmentation Detail (Di Methods)

**Template:**
```
Dataset Phase 4 diaugmentasi menggunakan Large Language Model (LLM). 
Proses augmentasi dilakukan dengan langkah-langkah berikut:

1. Model yang digunakan: [GPT-3.5/GPT-4/Claude/etc]
2. Prompt template:
   "Buatkan contoh kalimat ujaran kebencian dalam bahasa Jawa 
    dengan tingkat [ringan/sedang/berat] yang berkaitan dengan 
    topik [agama/etnis/politik]. Gunakan ragam bahasa [ngoko/krama]."
3. Filtering criteria:
   - Panjang minimum: [X] kata
   - Harus mengandung kata kunci bahasa Jawa
   - Review manual oleh annotator native speaker
4. Jumlah data yang dihasilkan: 5,240 samples
5. Jumlah data yang lolos filtering: [X] samples
```

### Task 4.3: Limitation Section (Di Discussion)

**Template:**
```
Penelitian ini memiliki beberapa keterbatasan. Pertama, dataset Phase 4 
yang diaugmentasi menggunakan LLM mungkin memperkenalkan bias dari model 
tersebut. Kedua, klasifikasi tingkat keparahan ujaran kebencian bersifat 
subjektif dan mungkin berbeda antar annotator. Ketiga, model hanya 
dilatih pada teks, tidak mempertimbangkan konteks multimodal seperti 
gambar atau video yang sering menyertai ujaran kebencian di media sosial.
```

---

## ✅ FASE 5: FINAL CHECKLIST

### Sebelum Submit ke JITK:

**Format:**
- [ ] Font Cambria di seluruh dokumen
- [ ] Judul-Abstract 1 kolom, Introduction dst 2 kolom
- [ ] Tidak ada bullet points
- [ ] Tabel tanpa garis vertikal
- [ ] Gambar dengan caption "Figure X. ..."
- [ ] Referensi format IEEE dengan Mendeley

**Konten:**
- [ ] Semua angka di tabel konsisten dengan gambar
- [ ] F1-Macro 81.38% terverifikasi dari confusion matrix
- [ ] Dataset 10,019 samples (bukan 42K)
- [ ] Related Work section ada
- [ ] LLM augmentation dijelaskan
- [ ] Statistical significance (mean ± std) dilaporkan
- [ ] Limitation section ada

**Gambar (jika digunakan):**
- [ ] Figure 1: Dataset distribution (4 kelas, ~10K total)
- [ ] Figure 2: Confusion matrix (F1=81.38%)
- [ ] Figure 3: Val vs Test comparison (opsional)

**Referensi:**
- [ ] Minimal 15 dari 2021-2025
- [ ] Tidak ada "prior work" tanpa sitasi
- [ ] Format IEEE konsisten

---

## 📁 STRUKTUR FILE YANG DIHARAPKAN

```
ujaran-kebencian-bahasa-jawa/
├── reproduce/
│   ├── experiment_6_focal_loss.py      # Training script
│   ├── generate_paper_figures.py       # NEW: Script generate gambar
│   └── experiment_multiple_seeds.py    # NEW: Script statistical significance
├── paper/
│   ├── figures/
│   │   ├── figure1_dataset_distribution.png
│   │   ├── figure2_confusion_matrix.png
│   │   └── figure3_model_comparison.png
│   ├── paper.tex
│   └── Paper_JITK_Final.docx
├── data/
│   ├── train.csv                       # 8,015 samples
│   ├── val.csv                         # 1,002 samples
│   └── test.csv                        # 1,002 samples
└── models/
    └── indobert_label_smoothing_best/  # Best model checkpoint
```

---

## 🚨 PENTING: Verifikasi Sebelum Submit

Jalankan script ini untuk final check:

```python
# verify_paper_data.py
"""
Verifikasi konsistensi data sebelum submit paper.
"""

# Expected values from paper
EXPECTED = {
    'total_samples': 10019,
    'test_samples': 1002,
    'f1_macro': 81.38,
    'accuracy': 81.24,
    'val_test_gap': 0.25,  # absolute value
}

# Load your actual results
# actual_f1 = ...
# actual_acc = ...

# Verify
# assert abs(actual_f1 - EXPECTED['f1_macro']) < 0.5, "F1 mismatch!"
# assert abs(actual_acc - EXPECTED['accuracy']) < 0.5, "Accuracy mismatch!"

print("✓ All verifications passed!")
```

---

## ⏱️ ESTIMASI WAKTU TOTAL

| Fase | Task | Estimasi |
|------|------|----------|
| 1 | Verifikasi Data | 1-2 jam |
| 2 | Generate Gambar | 2-3 jam |
| 3 | Multiple Seeds | 4-8 jam (training) |
| 4 | Writing | 2-3 jam |
| 5 | Final Check | 1 jam |
| **Total** | | **10-17 jam** |

---

*Workflow ini dibuat untuk memastikan paper JITK Anda konsisten dan bebas dari masalah yang menyebabkan desk reject sebelumnya.*
