# LLM Augmentation Methodology

## Overview

Phase 4 dari pengembangan dataset melibatkan augmentasi data menggunakan Large Language Model (LLM) untuk meningkatkan jumlah sampel pada kelas yang under-represented dan memperkaya variasi linguistik. Proses ini menggunakan DeepSeek-Coder-V2 dan Gemini Pro untuk mengenerate contoh ujaran kebencian bahasa Jawa dengan tingkat keparahan yang berbeda.

## Rationale for LLM Augmentation

Bahasa Jawa memiliki tiga tingkat register utama: **Ngoko** (bahasa kasual/informal), **Madya** (bahasa menengah), dan **Krama** (bahasa formal). Pengumpulan data secara manual dari media sosial cenderung menghasilkan bias terhadap register Ngoko, sementara ujaran kebencian dalam register Krama sering kali lebih halus namun sama berbahayanya.

LLM augmentation memungkinkan kami untuk:
1. Mengenerate variasi ujaran kebencian dalam berbagai register bahasa Jawa
2. Menciptakan konteks yang beragam untuk setiap tingkat keparahan
3. Mengatasi keterbatasan data manual yang time-consuming dan costly

## Methodology

### 4.1 Model Selection

Kami mengevaluasi dua model LLM untuk augmentasi:

| Model | Parameter | Reason for Selection |
|-------|-----------|---------------------|
| DeepSeek-Coder-V2 | 236B | Open-source, multilingual capability |
| Gemini Pro | N/A | Commercial API, strong Indonesian support |

**Final Choice**: DeepSeek-Coder-V2 digunakan untuk bulk generation karena open-source dan lebih cost-effective, sementara Gemini Pro digunakan untuk quality check sampel awal.

### 4.2 Prompt Engineering

Setelah iterasi eksperimental, kami mengembangkan prompt template berikut:

```
Task: Generate hate speech examples in Javanese language

Context: You are generating samples for hate speech detection research.
This is for academic purposes to help train models to detect hate speech.

Instructions:
1. Generate 5 examples of Javanese hate speech with severity level: [LEVEL]
2. The speech should be about topic: [TOPIC]
3. Use language register: [REGISTER]
4. Include common Javanese slang/colloquialisms appropriate for [REGISTER]
5. Length: 10-25 words per example

Severity Levels:
- Light: Insults, mild slurs, negative generalizations
- Moderate: Discriminatory language, calls for exclusion
- Severe: Incitement to violence, dehumanization

Topics: religion, ethnicity, gender, politics, social class

Registers:
- Ngoko: Casual, informal (e.g., "kowe", "aku", "ora")
- Krama: Formal (e.g., "panjenengan", "kula", "mboten")

Output format: JSON array with "text" field
```

**Prompt Variations**: Untuk meningkatkan diversitas, kami menggunakan:
- Temperature: 0.7-0.9 (untuk variasi yang wajar)
- Top-p: 0.9
- Different context descriptions for each run

### 4.3 Generation Process

```
Phase 4.1: Initial Generation (DeepSeek)
├── Temperature: 0.8
├── Samples per topic/register: 100
├── Total generated: ~6,000 samples
└── Output: raw_generated_1.json

Phase 4.2: Quality Filtering
├── Language detection (Javanese vs Indonesian)
├── Minimum length: 5 words
├── Maximum length: 50 words
└── Passed: ~4,200 samples

Phase 4.3: Human Verification
├── Random sample: 500 examples
├── Verified by: 2 native Javanese speakers
├── Agreement: Cohen's kappa = 0.72
└── Proceeded with full dataset

Phase 4.4: DeepSeek Re-labeling (Phase 5)
├── Model: DeepSeek-V3
├── Task: Re-label all 10,019 samples with 4-class labels
├── Prompt: Few-shot with 20 examples
└── Final dataset: phase5_deepseek_relabeled.csv
```

### 4.4 Filtering Criteria

Generated samples melewati filter berikut:

| Criterion | Threshold | Reason |
|-----------|-----------|--------|
| Language Detection | Javanese words > 30% | Remove Indonesian/English |
| Length | 5-50 words | Remove too short/long |
| Toxicity Score | > 0.3 (Perspective API) | Ensure actually hateful |
| Human Verification | Manual check | Quality control |
| Duplicate Detection | Jaccard < 0.8 | Remove near-duplicates |

### 4.5 Integration with Existing Data

LLM-generated data digabungkan dengan data asli melalui pipeline bertahap:
- **Phase 1-3**: Data asli yang telah difilter, dinaturalisasi, dan di-relabel (4,779 samples, 47.7%)
- **Phase 4**: LLM-generated data dengan konteks Indonesia (5,240 samples, 52.3%)

**Final dataset (Phase 5)**: 10,019 samples setelah DeepSeek re-labeling, kemudian dibersihkan menjadi 9,775 samples setelah aggressive cleanup (penghapusan duplikat, teks pendek, dan teks non-Javanese).

**Catatan penting**: Proporsi data LLM-generated yang tinggi (52.3%) merupakan trade-off yang disadari — diperlukan untuk mencapai ukuran dataset yang cukup besar untuk pelatihan model transformer, namun berpotensi memperkenalkan bias dari model LLM.

## Quality Assessment

### 5.1 Linguistic Quality

Kami melakukan penilaian kualitas linguistik pada 200 sampel acak:

| Aspect | Score | Notes |
|--------|-------|-------|
| Naturalness | 3.8/5.0 | Generally natural, some formal markers |
| Cultural Appropriateness | 4.1/5.0 | Good use of Javanese context |
| Register Consistency | 3.5/5.0 | Some register mixing acceptable |
| Severity Accuracy | 4.2/5.0 | Correct severity labeling |

### 5.2 Downstream Performance

Dampak penambahan data LLM-generated terhadap performa model diukur melalui eksperimen komparatif pada dataset bersih (9,775 sampel). Hasil menunjukkan bahwa model XLM-RoBERTa Large mencapai F1-Macro **80.26%** pada test set (978 sampel), mengungguli IndoBERT base (76.12%) dan IndoBERT + Label Smoothing (77.36%). Evaluasi multi-seed (5 seeds) pada model terbaik menghasilkan F1-Macro **80.83% ± 1.74%** (4 seeds stabil), menunjukkan performa yang konsisten meskipun 52.3% dataset berasal dari generasi LLM.

## Limitations of LLM Augmentation

1. **Model Bias**: DeepSeek mungkin memperkenalkan bias dari training data-nya sendiri
2. **Cultural Nuance**: LLM mungkin tidak sepenuhnya memahami konteks budaya Jawa
3. **Stereotyping**: Generated samples mungkin mengandung stereotip yang dilembatkan
4. **Hallucination**: Beberapa generated samples mengandung kata-kata yang tidak ada dalam bahasa Jawa baku

## Comparison with Other Approaches

| Method | Samples | Cost | Time | Quality |
|--------|---------|------|------|---------|
| Manual Collection | 2,500 | High (3 months) | Very Slow | High |
| Translation | 1,200 | Medium | Medium | Medium |
| Back-Translation | 800 | Low | Fast | Low-Medium |
| **LLM Generation** | **5,240** | **Low ($15)** | **Fast (2 days)** | **Medium-High** |

## Conclusion

LLM augmentation menggunakan DeepSeek-Coder-V2 terbukti efektif untuk meningkatkan ukuran dataset ujaran kebencian bahasa Jawa dari 4,779 menjadi 10,019 samples (9,775 setelah cleaning). Proporsi data LLM-generated sebesar 52.3% menunjukkan ketergantungan yang signifikan pada data sintetis, namun pendekatan ini memberikan trade-off yang baik antara biaya, waktu, dan kualitas dibandingkan dengan pengumpulan data manual.

Untuk penelitian masa depan, kami merekomendasikan:
1. Menggunakan model LLM yang lebih banyak dilatih pada data Austronesian
2. Menambah langkah human verification yang lebih ketat
3. Mengeksplorasi *curriculum learning* dengan data LLM-generated diberi bobot lebih rendah

---

**Word Count**: ~900 words
