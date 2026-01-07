# Dataset Specification - Ujaran Kebencian Bahasa Jawa

## 1. Overview Dataset

### 1.1 Informasi Umum
- **Nama Dataset:** Javanese Hate Speech Detection Dataset
- **Bahasa:** Bahasa Jawa (dengan variasi dialek)
- **Domain:** Media sosial, forum online, komentar publik
- **Ukuran:** 41,887 sampel teks
- **Format:** CSV dengan kolom text, final_label, label_numeric
- **Lisensi:** Academic Research Only

### 1.2 Karakteristik Linguistik
- **Tingkat Tutur:** Ngoko (informal) dan Krama (formal)
- **Dialek:** Jawa Tengah, Jawa Timur, Yogyakarta
- **Script:** Latin (Aksara Jawa tidak digunakan)
- **Code-mixing:** Campuran Jawa-Indonesia-Inggris

## 2. Skema Klasifikasi

### 2.1 Kategori Ujaran Kebencian

| Label | Kategori | Deskripsi | Contoh Karakteristik |
|-------|----------|-----------|----------------------|
| 0 | Bukan Ujaran Kebencian | Teks normal, netral, atau positif | Diskusi biasa, informasi, humor positif |
| 1 | Ujaran Kebencian - Ringan | Sindiran halus, stereotyping ringan | Generalisasi negatif, sindiran tidak langsung |
| 2 | Ujaran Kebencian - Sedang | Penghinaan langsung, diskriminasi | Kata-kata kasar, penghinaan identitas |
| 3 | Ujaran Kebencian - Berat | Ancaman, hasutan kekerasan | Ancaman fisik, hasutan konflik |

### 2.2 Kriteria Labeling

#### 2.2.1 Bukan Ujaran Kebencian (Label 0)
- **Definisi:** Teks yang tidak mengandung unsur kebencian, diskriminasi, atau penghinaan
- **Karakteristik:**
  - Diskusi normal dan konstruktif
  - Informasi faktual
  - Humor yang tidak menyinggung
  - Ekspresi emosi positif atau netral
- **Contoh:** "Ayo padha gotong royong gawe desa sing apik" (Mari bersama-sama gotong royong untuk desa yang baik)

#### 2.2.2 Ujaran Kebencian - Ringan (Label 1)
- **Definisi:** Teks yang mengandung stereotyping atau sindiran halus terhadap kelompok tertentu
- **Karakteristik:**
  - Generalisasi negatif yang tidak eksplisit
  - Sindiran atau sarkasme yang menyinggung
  - Prasangka terselubung
- **Contoh:** "Wong kono ki pancen angel diajak mikir" (Orang sana memang sulit diajak berpikir)

#### 2.2.3 Ujaran Kebencian - Sedang (Label 2)
- **Definisi:** Teks yang mengandung penghinaan langsung atau diskriminasi eksplisit
- **Karakteristik:**
  - Penggunaan kata-kata kasar atau menghina
  - Diskriminasi berdasarkan identitas
  - Penghinaan langsung terhadap kelompok
- **Contoh:** "Kelompok X ki pancen bodho kabeh" (Kelompok X memang bodoh semua)

#### 2.2.4 Ujaran Kebencian - Berat (Label 3)
- **Definisi:** Teks yang mengandung ancaman, hasutan kekerasan, atau ujaran yang sangat berbahaya
- **Karakteristik:**
  - Ancaman kekerasan fisik
  - Hasutan untuk menyakiti orang lain
  - Ujaran yang dapat memicu konflik serius
- **Contoh:** "Kudu diusir kabeh wong kaya ngono" (Harus diusir semua orang seperti itu)

## 3. Distribusi Data

### 3.1 Distribusi Kelas Original

| Kategori | Jumlah Sampel | Persentase | Rasio |
|----------|---------------|------------|-------|
| Bukan Ujaran Kebencian | ~35,604 | ~85.0% | 1.00 |
| Ujaran Kebencian - Ringan | ~3,141 | ~7.5% | 0.09 |
| Ujaran Kebencian - Sedang | ~2,094 | ~5.0% | 0.06 |
| Ujaran Kebencian - Berat | ~1,048 | ~2.5% | 0.03 |

### 3.2 Class Imbalance Analysis
- **Imbalance Ratio:** 34:1 (Bukan vs Berat)
- **Majority Class:** 85% (Bukan Ujaran Kebencian)
- **Minority Classes:** 15% total (semua kategori ujaran kebencian)
- **Challenge:** Severe class imbalance memerlukan strategi khusus

### 3.3 Balanced Dataset untuk Evaluasi
- **Strategi:** Stratified sampling 200 sampel per kelas
- **Total Balanced:** 800 sampel
- **Tujuan:** Evaluasi yang fair tanpa bias kelas mayoritas

## 4. Preprocessing Pipeline

### 4.1 Text Cleaning
```python
# Langkah preprocessing yang diterapkan:
1. Normalisasi whitespace
2. Removal karakter non-printable
3. Standardisasi encoding (UTF-8)
4. Preservasi struktur kalimat Jawa
```

### 4.2 Tokenization Strategy
- **Tokenizer:** WordPiece (dari model pre-trained)
- **Max Length:** 128-256 tokens (tergantung model)
- **Handling:** Truncation untuk teks panjang
- **Special Tokens:** [CLS], [SEP], [PAD], [UNK]

### 4.3 Data Augmentation
- **Tidak diterapkan** pada dataset ini untuk menjaga autentisitas linguistik
- **Alasan:** Kompleksitas bahasa Jawa memerlukan augmentasi yang sangat hati-hati

## 5. Quality Assurance

### 5.1 Labeling Quality
- **Inter-annotator Agreement:** Belum diukur (future work)
- **Quality Control:** Manual review untuk sampel ambiguous
- **Consistency Check:** Validasi label dengan kriteria yang ditetapkan

### 5.2 Data Validation
- **Missing Values:** Dihapus dari dataset
- **Duplicate Detection:** Belum diimplementasikan
- **Outlier Detection:** Teks dengan panjang ekstrem dianalisis

### 5.3 Ethical Considerations
- **Privacy:** Data dari domain publik, identitas dianonimkan
- **Bias:** Acknowledged class imbalance dan regional bias
- **Harm Prevention:** Dataset hanya untuk penelitian akademik

## 6. Dataset Splits

### 6.1 Training/Validation Split
- **Training:** 80% dari dataset original
- **Validation:** 20% dari dataset original
- **Method:** Stratified split untuk menjaga distribusi kelas

### 6.2 Evaluation Dataset
- **Balanced Evaluation:** 200 sampel per kelas (800 total)
- **Purpose:** Evaluasi yang tidak bias oleh class imbalance
- **Selection:** Random sampling dari setiap kelas

## 7. Limitations dan Challenges

### 7.1 Dataset Limitations
- **Regional Bias:** Dominasi dialek tertentu
- **Temporal Bias:** Data dari periode waktu terbatas
- **Domain Bias:** Fokus pada media sosial
- **Class Imbalance:** Severe imbalance memerlukan strategi khusus

### 7.2 Linguistic Challenges
- **Code-mixing:** Campuran bahasa yang kompleks
- **Contextual Meaning:** Makna yang sangat bergantung konteks
- **Cultural Nuances:** Nuansa budaya yang sulit dikodifikasi
- **Dialectal Variations:** Variasi antar daerah yang signifikan

### 7.3 Technical Challenges
- **Tokenization:** Handling bahasa yang tidak standard
- **Encoding:** Konsistensi encoding untuk karakter khusus
- **Memory:** Dataset besar memerlukan optimasi memory

## 8. Future Improvements

### 8.1 Dataset Enhancement
- **Size Expansion:** Menambah jumlah sampel untuk kelas minoritas
- **Dialectal Coverage:** Memperluas cakupan dialek
- **Temporal Diversity:** Data dari berbagai periode waktu
- **Domain Expansion:** Sumber data yang lebih beragam

### 8.2 Quality Improvements
- **Inter-annotator Agreement:** Mengukur konsistensi labeling
- **Duplicate Removal:** Deteksi dan removal duplikasi
- **Bias Analysis:** Analisis mendalam terhadap berbagai bias
- **Validation Framework:** Framework validasi yang lebih robust

---

**Referensi:**
- Dataset original: `data/standardized/balanced_dataset.csv`
- Preprocessing code: `src/preprocessing/`
- Evaluation framework: `src/evaluation/`

**Metadata:**
- Created: 2025-01-06
- Version: 1.0
- Status: Production Ready
- Contact: Research Team