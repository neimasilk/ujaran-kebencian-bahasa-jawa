# Panduan Pekerjaan Manual - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Dokumentator:** AI Assistant  
**Update Terakhir:** 2024-12-29  
**Versi:** 2.0 (Konsolidasi)  
**Status:** Aktif  
**Mengikuti:** Vibe Coding Guide v1.4

---

## 🎯 Overview

Dokumen ini mengkonsolidasikan semua panduan untuk pekerjaan manual yang esensial dalam proyek "Sistem Deteksi Ujaran Kebencian Bahasa Jawa Menggunakan BERT". Menggabungkan tugas arsitek dan panduan pelabelan dalam satu dokumen komprehensif.

## 📋 Pekerjaan Manual Prioritas Tinggi

### 1. Data Labeling dan Quality Assurance (URGENT)
**Assignee:** Mukhlis Amien  
**Estimasi:** 2-3 minggu  
**Dependencies:** Pedoman pelabelan sudah tersedia

#### Tugas Spesifik:
- [ ] **Pelabelan Manual 500 Sampel Awal**
  - Gunakan pedoman di bagian "Panduan Pelabelan" dokumen ini
  - Fokus pada 4 kategori: Bukan Ujaran Kebencian, Ringan, Sedang, Berat
  - Dokumentasikan edge cases dan ambiguitas
  - Target: 125 sampel per kategori untuk balanced dataset

- [ ] **Quality Control**
  - Review konsistensi pelabelan setiap 100 sampel
  - Buat catatan untuk kasus-kasus sulit
  - Validasi dengan ahli bahasa Jawa (jika tersedia)

- [ ] **Inter-annotator Agreement**
  - Jika ada multiple labeler, hitung agreement score
  - Target minimum: Cohen's Kappa > 0.7
  - Resolusi disagreement melalui diskusi

### 2. Model Architecture Review (HIGH)
**Assignee:** Mukhlis Amien  
**Estimasi:** 1 minggu  
**Dependencies:** Literature review selesai

#### Tugas Spesifik:
- [ ] **Fine-tuning Strategy**
  - Review IndoBERT vs multilingual BERT
  - Tentukan layer mana yang akan di-freeze
  - Strategi learning rate scheduling

- [ ] **Evaluation Metrics**
  - Definisi metrics yang relevan untuk Bahasa Jawa
  - Consideration untuk class imbalance
  - Cross-validation strategy

### 3. Cultural Context Validation (MEDIUM)
**Assignee:** Mukhlis Amien + Expert Bahasa Jawa  
**Estimasi:** 2 minggu  
**Dependencies:** Dataset awal tersedia

#### Tugas Spesifik:
- [ ] **Cultural Sensitivity Review**
  - Review sampel untuk konteks budaya Jawa
  - Identifikasi bias dalam dataset
  - Dokumentasi nuansa linguistik khusus

- [ ] **Domain Expert Consultation**
  - Konsultasi dengan ahli bahasa Jawa
  - Validasi kategori ujaran kebencian
  - Review edge cases

## 📖 Panduan Pelabelan Data Ujaran Kebencian Bahasa Jawa

### Tujuan Pelabelan

* Menyediakan "ground truth" bagi model machine learning untuk belajar membedakan berbagai kategori ujaran kebencian dan teks netral/bukan ujaran kebencian
* Memastikan dataset mencerminkan nuansa linguistik dan budaya Bahasa Jawa yang relevan dengan ujaran kebencian
* Menghasilkan dataset yang dapat digunakan untuk penelitian selanjutnya

### Tim Pelabel

**Kualifikasi Pelabel:**
- Penutur asli atau fasih Bahasa Jawa
- Pemahaman konteks budaya Jawa
- Kemampuan membedakan tingkat kesopanan dalam Bahasa Jawa (ngoko, madya, krama)
- Objektivitas dalam menilai konten sensitif

**Struktur Tim:**
- **Lead Annotator**: Mukhlis Amien (koordinator dan quality control)
- **Primary Annotators**: 2-3 orang dengan kualifikasi di atas
- **Expert Reviewer**: Ahli bahasa Jawa (konsultasi untuk kasus sulit)

### Kategori Pelabelan

#### 1. Bukan Ujaran Kebencian (Label: 0)
**Definisi**: Teks yang tidak mengandung unsur kebencian, diskriminasi, atau serangan terhadap individu/kelompok.

**Karakteristik:**
- Netral atau positif
- Tidak menyerang identitas tertentu
- Kritik konstruktif tanpa unsur kebencian

**Contoh:**
- "Aku seneng banget karo budaya Jawa"
- "Perlu sinau luwih akeh babagan sejarah"
- "Ora setuju karo keputusan iki, nanging kudu dihormati"

#### 2. Ujaran Kebencian Ringan (Label: 1)
**Definisi**: Teks yang mengandung unsur negatif ringan, stereotip, atau generalisasi yang tidak tepat.

**Karakteristik:**
- Stereotip ringan
- Generalisasi negatif
- Sindiran halus
- Tidak mengandung ancaman

**Contoh:**
- "Wong Jawa iku angel diajak maju"
- "Tipikal wong kota, ora ngerti adat"
- "Biasane wong kono iku angel diatur"

#### 3. Ujaran Kebencian Sedang (Label: 2)
**Definisi**: Teks yang mengandung diskriminasi jelas, penghinaan, atau serangan verbal yang lebih eksplisit.

**Karakteristik:**
- Penghinaan langsung
- Diskriminasi berdasarkan identitas
- Bahasa kasar dengan target spesifik
- Dehumanisasi ringan

**Contoh:**
- "Wong [etnis] iku memang ora iso dipercaya"
- "Agama [nama] iku ngrusak moral bangsa"
- "Kelompok iki kudu diusir saka kene"

#### 4. Ujaran Kebencian Berat (Label: 3)
**Definisi**: Teks yang mengandung ancaman, hasutan kekerasan, atau ujaran yang sangat ekstrem.

**Karakteristik:**
- Ancaman kekerasan eksplisit
- Hasutan untuk menyakiti
- Dehumanisasi ekstrem
- Call to action untuk kekerasan

**Contoh:**
- "Wong [kelompok] iku kudu dipateni kabeh"
- "Ayo berantas [kelompok] nganti tuntas"
- "Ora ana gunane urip [kelompok] iki"

### Prosedur Pelabelan

#### Langkah-langkah:

1. **Persiapan**
   - Baca seluruh panduan ini
   - Pahami konteks budaya dan linguistik
   - Setup environment pelabelan

2. **Pelabelan**
   - Baca teks dengan seksama
   - Pertimbangkan konteks budaya
   - Tentukan kategori berdasarkan kriteria
   - Catat kasus yang meragukan

3. **Quality Control**
   - Review setiap 50 sampel
   - Diskusi kasus sulit dengan tim
   - Dokumentasi keputusan pelabelan

4. **Dokumentasi**
   - Catat edge cases
   - Dokumentasi reasoning untuk kasus sulit
   - Update guidelines jika diperlukan

### Kasus Khusus dan Edge Cases

#### 1. Konteks Humor/Satire
**Prinsip**: Humor tidak menghilangkan sifat ujaran kebencian jika mengandung diskriminasi atau stereotip berbahaya.

**Contoh**:
- Joke yang memperkuat stereotip negatif → Tetap dianggap ujaran kebencian
- Satire yang mengkritik tanpa menyerang identitas → Bukan ujaran kebencian

#### 2. Kritik vs Ujaran Kebencian
**Prinsip**: Kritik terhadap tindakan/kebijakan berbeda dengan serangan terhadap identitas.

**Kritik yang Valid**:
- "Kebijakan pemerintah iki ora adil"
- "Keputusan pemimpin iki salah"

**Ujaran Kebencian**:
- "Wong [etnis] ora layak dadi pemimpin"
- "[Kelompok] iku memang bodoh"

#### 3. Bahasa Kasar vs Ujaran Kebencian
**Prinsip**: Bahasa kasar tanpa target identitas spesifik bukan ujaran kebencian.

**Bahasa Kasar Biasa**:
- "Sialan, macet maneh"
- "Brengsek, telat maneh"

**Ujaran Kebencian**:
- "Wong [kelompok] iku sialan kabeh"
- "[Etnis] brengsek ora guna"

### Tools dan Environment

#### Setup Pelabelan
```bash
# Gunakan sistem pelabelan yang sudah ada
python src/google_drive_labeling.py

# Atau manual labeling untuk quality control
python src/manual_labeling_tool.py
```

#### File Management
- **Input**: `src/data_collection/raw-dataset.csv`
- **Output**: `hasil-labeling.csv`
- **Backup**: Google Drive sync otomatis
- **Checkpoint**: `src/checkpoints/`

### Quality Assurance

#### Metrics untuk Evaluasi
1. **Inter-annotator Agreement**
   - Cohen's Kappa > 0.7 (target)
   - Percentage agreement > 85%

2. **Consistency Check**
   - Review random sample 10%
   - Check for label drift over time

3. **Expert Validation**
   - Expert review untuk 5% sampel
   - Focus pada edge cases

#### Resolusi Konflik
1. **Disagreement antar Annotator**
   - Diskusi tim untuk mencapai konsensus
   - Konsultasi expert jika perlu
   - Dokumentasi reasoning

2. **Kasus Ambiguous**
   - Buat kategori "uncertain" sementara
   - Kumpulkan lebih banyak konteks
   - Expert consultation

### Timeline dan Milestone

#### Week 1-2: Setup dan Training
- Setup environment
- Training tim pelabel
- Pilot labeling 100 sampel
- Calibration dan adjustment

#### Week 3-4: Bulk Labeling
- Labeling 400 sampel utama
- Daily quality check
- Weekly team sync

#### Week 5: Quality Assurance
- Final review
- Inter-annotator agreement calculation
- Expert validation
- Documentation finalization

## 📊 Deliverables

### 1. Labeled Dataset
- 500 sampel berlabel dengan distribusi seimbang
- Format CSV dengan kolom: text, label, confidence, notes
- Backup di Google Drive

### 2. Documentation
- Labeling guidelines (dokumen ini)
- Edge cases documentation
- Quality metrics report
- Lessons learned

### 3. Quality Reports
- Inter-annotator agreement analysis
- Label distribution analysis
- Difficult cases analysis
- Recommendations untuk improvement

## 📚 Referensi

- **Spesifikasi Produk**: `spesifikasi-produk.md`
- **Architecture**: `architecture.md`
- **Team Manifest**: `../vibe-guide/team-manifest.md`
- **Vibe Coding Guide**: `../vibe-guide/VIBE_CODING_GUIDE.md`
- **Google Drive Integration**: `google-drive-integration.md`

---

*Dokumen ini mengkonsolidasikan informasi dari pekerjaan-manual-arsitek.md dan petunjuk-pekerjaan-manual.md untuk kemudahan maintenance dan referensi.*