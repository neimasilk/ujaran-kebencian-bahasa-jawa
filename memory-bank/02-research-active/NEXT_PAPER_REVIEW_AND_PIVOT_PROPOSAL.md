# Review Metodologi & Proposal Pivot untuk Next Paper
# Javanese Hate Speech Detection

**Tanggal:** 2025-01-08
**Tujuan:** Evaluasi kritis dan rekomendasi pivot untuk paper berikutnya

---

## 1. Kritik Terhadap Metodologi Saat Ini

### 1.1 Masalah Utama: Klasifikasi Subjektif

Saat ini menggunakan **4 kategori** yang sangat subjektif:

| Kategori | Definisi Saat Ini | Masalah |
|----------|-------------------|---------|
| Bukan Ujaran Kebencian | Teks netral/positif | Relatif jelas |
| Ringan | "Sindiran halus, ejekan terselubung" | **Sangat subjektif** |
| Sedang | "Hinaan langsung, cercaan" | **Boundary dengan Ringan kabur** |
| Berat | "Ancaman kekerasan, hasutan" | Relatif lebih jelas |

### 1.2 Statistik Inter-Annotator Agreement

- **Cohen's Kappa: 0.72** (substantial agreement)
- **~28% label bisa berubah** dengan annotator berbeda
- Boundary antara Ringan vs Sedang diakui "kabur"

### 1.3 Tidak Ada Dasar Framework Akademis

Klasifikasi dibuat secara **ad-hoc** tanpa merujuk:
- Taksonomi UNESCO
- Framework riset hate speech yang sudah established
- Konsensus linguistik komputasional

---

## 2. Framework Akademis yang Relevan

### 2.1 UNESCO Framework (2019)

UNESCO mendefinisikan hate speech sebagai:

> "Any kind of communication in speech, writing or behaviour, that attacks or uses pejorative or discriminatory language with reference to a person or a group on the basis of... religion, ethnicity, nationality, race, colour, descent, gender or other identity factor."

**Poin penting:**
- UNESCO **tidak membagi** menjadi ringan/sedang/berat
- Fokus pada **identitas target** (SARA)
- Biner: Hate Speech vs Non-Hate Speech

### 2.2 Cross-Taxonomy Integration (Fillies & Paschke, 2025)

Penelitian terbaru menunjukkan tren:
- **Multi-dimensional classification** bukan multi-level severity
- Dimensi yang terpisah:
  - Target identity (Siapa yang diserang?)
  - Type of attack (Insult, threat, dehumanization?)
  - Implicitness (Explicit vs Implicit)

### 2.3 Implicit Hate Speech Codetypes (2025)

Taksonomi baru untuk implicit hate speech dengan 6 **encoding strategies**:
1. **Stereotyping** - Generalisasi negatif
2. **Othering** - Memisahkan "kita" vs "mereka"
3. **Threatening** - Ancaman terselubung
4. **Dehumanizing** - Menyerang kemanusiaan
5. **Mocking** - Mengejek
6. **Blaming** - Menyalahkan

### 2.4 Multi-Task Learning Framework (MTLHateCorpus 2023)

Pendekatan multi-task dengan subtask terpisah:
- **Target Detection** - Apakah ada target?
- **Target Identification** - Target apa (etnis, gender, agama)?
- **Intensity Measurement** - Severity scoring (bukan kategori!)

---

## 3. Proposal Pivot untuk Next Paper

### Opsi 1: Binary Classification + Multi-Label Dimensions

**Ide:** Pisahkan menjadi dua task terpisah:

```
Task 1: Binary Classification
├── Hate Speech (0/1)

Task 2: Multi-Label Classification
├── Target: [Etnis, Agama, Gender, DLL]
├── Type: [Insult, Threat, Dehumanization, Stereotype]
├── Explicitness: [Explicit, Implicit]
└── Intensity Score: [0.0 - 1.0] ← continuous, bukan kategori
```

**Kelebihan:**
- Lebih objektif
- Sesuai framework akademis terkini
- Intensity score menghindari masalah kategori subjektif

**Contoh:**
| Teks | Hate | Target | Type | Explicit | Intensity |
|------|------|--------|------|----------|-----------|
| "Wong Jawa iku gela" | 1 | Etnis | Stereotype | Explicit | 0.6 |
| "Mesthine sah yen wong lanang ngenthu..." | 1 | Gender | Othering | Implicit | 0.4 |
| "Dalit minangka sampah masyarakat" | 1 | Kasta | Dehumanization | Explicit | 0.9 |

---

### Opsi 2: Focus pada Implicit Hate Speech Detection

**Problem Statement:**
> "Deteksi ujaran kebencian eksplisit sudah relatif mudah. Tantangan sebenarnya adalah mengenali ujaran kebencian **implisit** dalam Bahasa Jawa yang memanfaatkan konteks budaya, hierarki bahasa (ngoko/krama), dan metafora."

**Kontribusi Ilmiah:**
1. **Taksonomi Implicit Hate Speech Bahasa Jawa** - Adaptasi dari codetypes 2025
2. **Dataset Implicit Hate Speech** - Anotasi dengan penjelasan konteks budaya
3. **Model Deteksi Implicit** - Architecture khusus untuk implicit detection

**Framework yang Diusulkan:**

```
Javanese Implicit Hate Speech Taxonomy

1. Stereotyping Budaya
   - "Wong deso iku ora ngerti sopan santun"
   - Menggunakan generalisasi budaya

2. Othering Hierarkis
   - "Kowe ngoko wae, aku krama"
   - Mempermainkan tingkatan bahasa Jawa

3. Dehumanizing Kasta
   - Referensi hierarki sosial tradisional
   - "Wong cilik", "wong gedhe" dengan konotasi merendahkan

4. Mocking Aksara/Budaya
   - Mengejek aksara Jawa, budaya tradisional

5. Religious Syncretism Abuse
   - Memanfaatkan sinkretisme agama Jawa-Islam-Hindu

6. Dialectic Discrimination
   - Diskriminasi dialek (Banyumasan, Arekan, Mataraman)
```

---

### Opsi 3: Focus pada Sociolinguistic Dimensions

**Problem Statement:**
> "Bagaimana tingkatan bahasa Jawa (Ngoko, Madya, Krama) mempengaruhi perseksi ujaran kebencian?"

**Kontribusi Ilmiah:**
1. Analisis bagaimana **speech level** mempengaruhi severity
2. Model yang **aware of register** bahasa Jawa
3. Dataset dengan anotasi speech level

**Framework yang Diusulkan:**

```
Multi-Dimensional Annotation:

Dimensi 1: Hate Speech Detection (Binary)
├── Hate / Non-Hate

Dimensi 2: Speech Level Register
├── Ngoko (Kasar)
├── Madya (Tengah)
└── Krama (Halus)

Dimensi 3: Contextual Appropriateness
├── Appropriate (sesuai konteks)
├── Inappropriate (tidak sesuai konteks)
└── Offensive (menyinggung)

Dimensi 4: Target Identity
├── Etnis/Regional
├── Agama
├── Status Sosial
└── Gender
```

**Insight:**
- Ngoko yang "tidak pada tempatnya" bisa lebih menyinggung daripada Krama yang kasar
- Konteks sosial mempengaruhi severity

---

### Opsi 4: Code-Mixing Hate Speech Detection

**Problem Statement:**
> "Bagaimana mendeteksi ujaran kebencian dalam teks code-mixed Jawa-Indonesia-Inggris?"

**Kontribusi Ilmiah:**
1. **Dataset Code-Mixed Hate Speech** - Jawa + Indonesia + English
2. **Language-Aware Model** - Model yang tahu kapan ganti bahasa
3. **Analysis of Switching Patterns** - Pola code-switching dalam hate speech

**Framework yang Diusulkan:**

```
Code-Mixed Hate Speech Classification

Label Structure:
├── Overall: Hate / Non-Hate
├── Language Components:
│   ├── Jawa Only
│   ├── Indonesia Only
│   ├── Jawa-Indonesia Mixed
│   └── Jawa-Indonesia-English Mixed
└── Hate Location:
    ├── Hate in Jawa portion
    ├── Hate in Indonesia portion
    └── Hate in both
```

---

## 4. Rekomendasi untuk Paper Saat Ini vs Next Paper

### 4.1 Paper Saat Ini (Release)

Dengan sadar akan keterbatasan, dokumentasikan secara transparan:

```markdown
## Limitations & Future Work

1. **Subjektivitas Klasifikasi**
   - Kategori ringan/sedang/berat memiliki batas yang kabur
   - Inter-annotator agreement: 0.72
   - Memerlukan framework yang lebih objektif

2. **Rekomendasi Future Work**
   - Binary classification + continuous severity scoring
   - Multi-dimensional labeling (target, type, explicitness)
   - Focus pada implicit hate speech
   - Sociolinguistic dimension (speech level awareness)
```

### 4.2 Next Paper - Rekomendasi Utama

Berdasarkan tren akademis dan keunikan Bahasa Jawa, rekomendasi saya:

**🎯 Opsi Terbaik: Implicit Hate Speech Detection with Sociolinguistic Context**

**Alasan:**
1. **Novelty** - Implicit hate detection adalah frontier research
2. **Relevance to Javanese** - Bahasa Jawa kaya dengan konteks budaya implisit
3. **Scientific Contribution** - Bisa membuat taksonomi baru yang terpublish
4. **Practical Impact** - Lebih berguna untuk memoderasi hate speech yang halus

---

## 5. Action Plan untuk Next Paper

### Phase 1: Literature Review (2 minggu)
- [ ] Baca paper terbaru tentang implicit hate speech
- [ ] Studi taksonomi yang sudah ada
- [ ] Identifikasi gap untuk Javanese context

### Phase 2: Framework Development (2 minggu)
- [ ] Desain taksonomi implicit hate speech Bahasa Jawa
- [ ] Buat guidelines anotasi yang objektif
- [ ] Validasi dengan ahli bahasa/sosiolog

### Phase 3: Data Collection & Annotation (4 minggu)
- [ ] Kumpulkan data dengan implicit hate speech
- [ ] Anotasi dengan multi-label scheme
- [ ] Hitung inter-annotator agreement
- [ ] Publish dataset sebagai kontribusi

### Phase 4: Model Development (4 minggu)
- [ ] Baseline: BERT, RoBERTa
- [ ] Advanced: Context-aware model, sociolinguistic features
- [ ] Evaluation dengan metrics yang sesuai

### Phase 5: Paper Writing (4 minggu)
- [ ] Tulis paper dengan kontribusi jelas
- [ ] Submit ke conference/jurnal relevan

---

## 6. Referensi Penting untuk Next Paper

### Academic Frameworks
- [UNESCO Countering Hate Speech](https://www.unesco.org/en/countering-hate-speech/need-know)
- [Fillies & Paschke 2025: Cross-Taxonomy Dataset Integration](https://aclanthology.org/2025.latechclfl-1.14.pdf)
- [Implicit Hate Speech Detection through Coding](https://arxiv.org/html/2506.04693v1)
- [MTLHateCorpus 2023: Multi-task Learning](https://www.sciencedirect.com/science/article/pii/S0920548925000194)
- [Multi-Label Classification of Hate Speech Severity](https://www.researchgate.net/publication/362986863_Multi-Label_Classification_of_Hate_Speech_Severity_on_Social_Media_using_BERT_Model)

### Datasets for Comparison
- [Hate Speech Dataset Catalogue](https://hatespeechdata.com/)

---

## 7. Kesimpulan

Masalah utama dengan pendekatan saat ini adalah **subjektivitas** klasifikasi ringan/sedang/berat. Untuk next paper, direkomendasikan:

1. **Binary classification** sebagai base task
2. **Multi-dimensional annotation** untuk depth
3. **Focus pada implicit hate** dengan sociolinguistic context
4. **Framework based on established research**, bukan ad-hoc

Ini akan memberikan kontribusi ilmiah yang lebih kuat dan lebih mudah diterima di komunitas akademik.

---

**Document Status:** Draft untuk review dan diskusi
**Next Step:** Diskusikan pilihan pivot direction
