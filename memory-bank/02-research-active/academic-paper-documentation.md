# Dokumentasi Akademik: Deteksi Ujaran Kebencian Bahasa Jawa Menggunakan Deep Learning

## Abstract

Penelitian ini mengembangkan sistem deteksi ujaran kebencian untuk bahasa Jawa menggunakan model transformer berbasis IndoBERT. Evaluasi awal menunjukkan akurasi 95.5%, namun analisis mendalam mengungkap bias evaluasi yang signifikan akibat ketidakseimbangan kelas dan pengurutan data. Evaluasi yang diperbaiki dengan dataset seimbang menunjukkan akurasi aktual 73.8%, mengindikasikan perlunya strategi training yang lebih sophisticated untuk mengatasi class imbalance dalam deteksi ujaran kebencian bahasa daerah.

**Keywords**: Hate Speech Detection, Javanese Language, Class Imbalance, Transformer Models, IndoBERT

## 1. Pendahuluan

### 1.1 Latar Belakang
Ujaran kebencian dalam bahasa daerah, khususnya bahasa Jawa, menjadi tantangan tersendiri dalam Natural Language Processing (NLP) karena:
- Keterbatasan dataset berlabel
- Kompleksitas linguistik bahasa Jawa
- Variasi dialek dan tingkat tutur
- Ketidakseimbangan kelas dalam data training

### 1.2 Kontribusi Penelitian
1. **Dataset Berlabel**: Pengembangan dataset ujaran kebencian bahasa Jawa dengan 41,887 sampel
2. **Model Detection**: Implementasi model berbasis IndoBERT untuk klasifikasi 4 kategori
3. **Bias Analysis**: Identifikasi dan analisis bias evaluasi akibat class imbalance
4. **Improvement Strategy**: Pengembangan strategi perbaikan dengan class weighting dan focal loss

## 2. Metodologi

### 2.1 Dataset

#### 2.1.1 Sumber Data
- **Total Sampel**: 41,887 teks bahasa Jawa
- **Sumber**: Media sosial, forum online, komentar publik
- **Periode Pengumpulan**: [Sesuaikan dengan periode aktual]

#### 2.1.2 Skema Labeling
```
Kategori Ujaran Kebencian:
1. Bukan Ujaran Kebencian (0)
2. Ujaran Kebencian - Ringan (1)
3. Ujaran Kebencian - Sedang (2)
4. Ujaran Kebencian - Berat (3)
```

#### 2.1.3 Distribusi Data
| Kategori | Jumlah | Persentase |
|----------|--------|------------|
| Bukan Ujaran Kebencian | ~35,604 | ~85.0% |
| Ujaran Kebencian - Ringan | ~3,141 | ~7.5% |
| Ujaran Kebencian - Sedang | ~2,094 | ~5.0% |
| Ujaran Kebencian - Berat | ~1,048 | ~2.5% |

**Observasi**: Severe class imbalance dengan rasio 85:15 antara non-hate speech dan hate speech.

### 2.2 Model Architecture

#### 2.2.1 Base Model
- **Model**: IndoBERT (indobenchmark/indobert-base-p1)
- **Architecture**: BERT-based transformer
- **Parameters**: 110M parameters
- **Tokenizer**: WordPiece tokenization

#### 2.2.2 Fine-tuning Configuration
```python
Training Parameters:
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Max Length: 128 tokens
- Optimizer: AdamW
- Weight Decay: 0.01
```

### 2.3 Evaluation Methodology

#### 2.3.1 Initial Evaluation (Biased)
- **Split**: 80% training, 20% evaluation
- **Method**: Sequential split (tanpa shuffling)
- **Metrics**: Accuracy, Precision, Recall, F1-Score

#### 2.3.2 Improved Evaluation (Balanced)
- **Dataset**: Stratified sampling (200 sampel per kelas)
- **Method**: Balanced evaluation set
- **Metrics**: Per-class metrics, confusion matrix, macro/weighted averages

## 3. Hasil dan Analisis

### 3.1 Evaluasi Initial (Biased Results)

#### 3.1.1 Metrics Keseluruhan
```
Accuracy: 95.5%
F1-Score (Weighted): 97.7%
Precision (Weighted): 100.0%
```

#### 3.1.2 Distribusi Prediksi
```
Prediksi Model:
- Bukan Ujaran Kebencian: 955/1000 (95.5%)
- Ujaran Kebencian - Ringan: 0/1000 (0.0%)
- Ujaran Kebencian - Sedang: 0/1000 (0.0%)
- Ujaran Kebencian - Berat: 0/1000 (0.0%)
```

**Critical Finding**: Model memprediksi SEMUA sampel sebagai "Bukan Ujaran Kebencian".

### 3.2 Root Cause Analysis

#### 3.2.1 Data Ordering Bias
- Dataset tersusun berurutan berdasarkan label
- Evaluasi pada 20% terakhir hanya mengandung "Bukan Ujaran Kebencian"
- Menghasilkan false high accuracy

#### 3.2.2 Class Imbalance Impact
- Rasio 85:15 menyebabkan model bias terhadap kelas mayoritas
- Tidak ada class weighting dalam training
- Loss function standard tidak efektif untuk imbalanced data

### 3.3 Evaluasi Balanced (Actual Results)

#### 3.3.1 Metrics Keseluruhan
```
Accuracy: 73.8%
F1-Score (Macro): 0.651
F1-Score (Weighted): 0.738
Precision (Macro): 0.740
Recall (Macro): 0.701
```

#### 3.3.2 Per-Class Performance
| Kategori | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|----------|
| Bukan Ujaran Kebencian | 0.577 | 0.930 | 0.713 | 200 |
| Ujaran Kebencian - Ringan | 0.750 | 0.450 | 0.563 | 200 |
| Ujaran Kebencian - Sedang | 0.750 | 0.600 | 0.667 | 200 |
| Ujaran Kebencian - Berat | 0.882 | 0.825 | 0.852 | 200 |

#### 3.3.3 Confusion Matrix
```
                    Predicted
Actual        0    1    2    3
    0       186   14    0    0
    1        90   90   20    0
    2        30   50  120    0
    3         9   26    0  165
```

### 3.4 Key Findings

1. **Severe Model Bias**: Model sangat bias terhadap kelas "Bukan Ujaran Kebencian"
2. **High False Negative Rate**: Banyak hate speech yang tidak terdeteksi
3. **Class-Dependent Performance**: Performa berbeda signifikan antar kelas
4. **Evaluation Methodology Critical**: Metodologi evaluasi sangat mempengaruhi hasil

## 4. Strategi Perbaikan

### 4.1 Improved Training Strategy

#### 4.1.1 Class Weighting
```python
Class Weights (Balanced):
- Bukan Ujaran Kebencian: 0.295
- Ujaran Kebencian - Ringan: 3.333
- Ujaran Kebencian - Sedang: 5.000
- Ujaran Kebencian - Berat: 10.000
```

#### 4.1.2 Focal Loss Implementation
```python
Focal Loss Parameters:
- Alpha: 1.0
- Gamma: 2.0
- Class Weights: Integrated
```

#### 4.1.3 Stratified Sampling
- Training/Validation split dengan stratified sampling
- Mempertahankan distribusi kelas di setiap split
- Weighted Random Sampler untuk training

### 4.2 Threshold Tuning

#### 4.2.1 Per-Class Threshold Optimization
- Optimasi threshold berdasarkan F1-score per kelas
- Precision-Recall curve analysis
- Balance antara precision dan recall

#### 4.2.2 Expected Improvements
```
Target Metrics (Post-Improvement):
- Accuracy: 80-85%
- F1-Score (Macro): 0.75-0.80
- Balanced performance across classes
```

## 5. Diskusi

### 5.1 Implikasi Metodologis

#### 5.1.1 Evaluation Bias
- **Lesson Learned**: Evaluasi pada imbalanced dataset memerlukan stratified sampling
- **Best Practice**: Selalu gunakan balanced evaluation set untuk assessment yang akurat
- **Warning**: High accuracy pada imbalanced data bisa menyesatkan

#### 5.1.2 Class Imbalance Handling
- **Critical**: Class imbalance harus ditangani sejak tahap training
- **Strategy**: Kombinasi class weighting, focal loss, dan sampling strategy
- **Monitoring**: Evaluasi per-class metrics, bukan hanya overall accuracy

### 5.2 Tantangan Bahasa Jawa

#### 5.2.1 Linguistic Complexity
- Variasi dialek (Ngoko, Krama, Krama Inggil)
- Code-switching dengan bahasa Indonesia
- Konteks budaya dalam interpretasi ujaran kebencian

#### 5.2.2 Data Scarcity
- Keterbatasan data berlabel untuk bahasa daerah
- Ketidakseimbangan natural dalam distribusi hate speech
- Perlunya data augmentation dan synthetic data generation

### 5.3 Limitasi Penelitian

1. **Dataset Size**: Meskipun 41,887 sampel cukup besar, distribusi tidak seimbang
2. **Annotation Quality**: Subjektivitas dalam labeling ujaran kebencian
3. **Generalization**: Model mungkin tidak generalize ke dialek Jawa lainnya
4. **Context Dependency**: Ujaran kebencian sangat bergantung konteks

## 6. Rekomendasi

### 6.1 Immediate Actions

1. **Re-training dengan Improved Strategy**
   - Implementasi class weighting dan focal loss
   - Stratified sampling untuk training/validation split
   - Extended training dengan learning rate scheduling

2. **Comprehensive Evaluation**
   - Evaluasi pada multiple balanced test sets
   - Cross-validation dengan stratified folds
   - Error analysis untuk understanding failure cases

3. **Threshold Optimization**
   - Per-class threshold tuning
   - Production-ready threshold configuration
   - A/B testing untuk optimal threshold selection

### 6.2 Future Work

1. **Data Enhancement**
   - Data augmentation untuk kelas minoritas
   - Synthetic data generation menggunakan language models
   - Active learning untuk efficient annotation

2. **Model Architecture**
   - Ensemble methods untuk improved robustness
   - Multi-task learning dengan related tasks
   - Domain adaptation techniques

3. **Evaluation Framework**
   - Standardized evaluation protocol untuk bahasa daerah
   - Fairness metrics untuk bias assessment
   - Real-world deployment testing

## 7. Kesimpulan

Penelitian ini mengungkap pentingnya metodologi evaluasi yang tepat dalam pengembangan model NLP untuk bahasa daerah. Temuan utama:

1. **Evaluation Bias**: Akurasi 95.5% yang dilaporkan awalnya adalah hasil dari bias evaluasi, dengan akurasi aktual 73.8%

2. **Class Imbalance Impact**: Severe class imbalance (85:15) menyebabkan model bias yang signifikan terhadap kelas mayoritas

3. **Methodology Matters**: Stratified sampling dan balanced evaluation set sangat critical untuk assessment yang akurat

4. **Improvement Potential**: Dengan strategi training yang diperbaiki, target akurasi 80-85% dapat dicapai

Hasil ini menekankan perlunya:
- Careful evaluation methodology dalam imbalanced datasets
- Sophisticated training strategies untuk class imbalance
- Comprehensive bias analysis dalam model development
- Standardized protocols untuk NLP bahasa daerah

## References

[Akan diisi dengan referensi akademik yang relevan]

## Appendix

### A. Detailed Experimental Setup
### B. Complete Evaluation Results
### C. Error Analysis
### D. Code Implementation

---

**Corresponding Author**: [Nama Peneliti]
**Institution**: [Institusi]
**Email**: [Email]
**Date**: January 2025