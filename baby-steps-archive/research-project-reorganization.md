# Reorganisasi Dokumentasi Proyek Penelitian Ujaran Kebencian Bahasa Jawa

**Tanggal:** 2025-01-02  
**Arsitek:** AI Research Assistant  
**Status:** Proposal Reorganisasi  
**Tujuan:** Menata ulang dokumentasi untuk mendukung penulisan paper dan eksperimen lanjutan  

---

## 🎯 Analisis Situasi Saat Ini

### Masalah yang Diidentifikasi:
1. **Dokumentasi Terpecah-pecah**: Dokumentasi tersebar antara `/memory-bank/` dan root directory
2. **Duplikasi Konten**: Beberapa dokumen memiliki konten yang overlap
3. **Inkonsistensi Format**: Format dokumentasi tidak seragam
4. **Kesulitan Navigasi**: Sulit menemukan informasi spesifik
5. **Status Eksperimen Tidak Jelas**: Progress eksperimen ke-3 tidak terdokumentasi dengan baik

### Dokumentasi yang Ada:

#### Di Root Directory:
- `RESEARCH_PUBLICATION_STRATEGY.md` - Strategi publikasi paper
- `EXPERIMENT_DOCUMENTATION_FOR_PAPER.md` - Dokumentasi eksperimen untuk paper
- `ARCHITECTURAL_DOCUMENTATION_FOR_PAPER.md` - Dokumentasi arsitektur untuk paper
- `MODEL_DEVELOPMENT_ROADMAP.md` - Roadmap pengembangan model
- `TECHNICAL_IMPLEMENTATION_GUIDE.md` - Panduan implementasi teknis
- `readme.md` - Dokumentasi umum proyek
- Dan 10+ dokumen lainnya

#### Di `/memory-bank/`:
- `spesifikasi-produk.md` - Spesifikasi produk
- `architecture.md` - Arsitektur sistem
- `progress.md` - Progress proyek
- `project-summary.md` - Ringkasan proyek
- Dan 20+ dokumen lainnya

---

## 🏗️ Struktur Reorganisasi yang Diusulkan

### 1. Folder Utama Baru: `/docs/`

```
docs/
├── 01-project-overview/
│   ├── README.md                    # Overview proyek utama
│   ├── project-specification.md     # Spesifikasi lengkap (gabungan)
│   └── research-objectives.md       # Tujuan penelitian
│
├── 02-research-methodology/
│   ├── dataset-description.md       # Deskripsi dataset (raw + labeled)
│   ├── experimental-design.md       # Desain eksperimen
│   └── evaluation-framework.md      # Framework evaluasi
│
├── 03-experiments/
│   ├── experiment-01-baseline.md    # Eksperimen 1 (baseline)
│   ├── experiment-02-improved.md    # Eksperimen 2 (improved)
│   ├── experiment-03-indobert-large.md # Eksperimen 3 (sedang berjalan)
│   └── experiments-comparison.md    # Perbandingan semua eksperimen
│
├── 04-technical-documentation/
│   ├── system-architecture.md       # Arsitektur sistem
│   ├── implementation-guide.md      # Panduan implementasi
│   ├── api-documentation.md         # Dokumentasi API
│   └── deployment-guide.md          # Panduan deployment
│
├── 05-paper-preparation/
│   ├── paper-outline.md             # Outline paper
│   ├── literature-review.md         # Literature review
│   ├── results-analysis.md          # Analisis hasil
│   └── publication-strategy.md      # Strategi publikasi
│
└── 06-appendices/
    ├── code-documentation.md        # Dokumentasi kode
    ├── troubleshooting.md          # Troubleshooting
    └── references.md               # Referensi
```

### 2. Reorganisasi `/memory-bank/` → `/archive/`

```
archive/
├── historical-documents/           # Dokumen historis
├── deprecated-experiments/         # Eksperimen yang sudah tidak digunakan
├── old-implementations/           # Implementasi lama
└── meeting-notes/                 # Catatan meeting
```

### 3. Update Root Directory

```
Root/
├── README.md                      # Overview utama (simplified)
├── QUICK_START.md                 # Panduan cepat memulai
├── CURRENT_STATUS.md              # Status terkini proyek
└── docs/                          # Dokumentasi lengkap
```

---

## 📋 Rencana Implementasi

### Phase 1: Konsolidasi Dokumentasi (Hari 1-2)
1. **Buat struktur folder baru** `/docs/`
2. **Gabungkan dokumen yang overlap**:
   - Gabungkan `spesifikasi-produk.md` + `RESEARCH_PUBLICATION_STRATEGY.md` → `project-specification.md`
   - Gabungkan `architecture.md` + `ARCHITECTURAL_DOCUMENTATION_FOR_PAPER.md` → `system-architecture.md`
   - Gabungkan `progress.md` + `project-summary.md` → `CURRENT_STATUS.md`

### Phase 2: Dokumentasi Eksperimen (Hari 2-3)
1. **Dokumentasikan eksperimen yang sudah selesai**:
   - Eksperimen 1: Baseline IndoBERT
   - Eksperimen 2: Improved training strategy
2. **Dokumentasikan eksperimen yang sedang berjalan**:
   - Eksperimen 3: IndoBERT Large (fix error yang ada)
3. **Buat perbandingan hasil eksperimen**

### Phase 3: Persiapan Paper (Hari 3-4)
1. **Buat outline paper yang detail**
2. **Konsolidasi hasil eksperimen untuk paper**
3. **Persiapkan figures dan tables**
4. **Update strategi publikasi**

### Phase 4: Cleanup dan Archive (Hari 4-5)
1. **Pindahkan dokumen lama ke `/archive/`**
2. **Update semua referensi internal**
3. **Buat dokumentasi navigasi**
4. **Testing semua link dan referensi**

---

## 🎯 Manfaat Reorganisasi

### Untuk Penelitian:
- **Navigasi yang lebih mudah** untuk menemukan informasi
- **Dokumentasi eksperimen yang terstruktur** untuk analisis
- **Persiapan paper yang lebih sistematis**

### Untuk Pengembangan:
- **Dokumentasi teknis yang terpusat**
- **Panduan implementasi yang jelas**
- **Troubleshooting yang mudah diakses**

### Untuk Kolaborasi:
- **Onboarding yang lebih mudah** untuk anggota tim baru
- **Dokumentasi yang konsisten** untuk semua stakeholder
- **Version control yang lebih baik**

---

## 🚀 Next Steps

1. **Approval reorganisasi** dari tim
2. **Backup dokumentasi existing** sebelum reorganisasi
3. **Implementasi phase by phase** sesuai rencana
4. **Update semua referensi** di kode dan dokumentasi
5. **Training tim** untuk struktur baru

---

## 📊 Status Eksperimen Saat Ini

### Eksperimen 3: IndoBERT Large
- **Status**: Sedang berjalan dengan error
- **Error**: `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`
- **Action Required**: Fix parameter compatibility dengan Transformers library
- **Expected Fix**: Ganti `evaluation_strategy` dengan `eval_strategy`

### Dataset Status:
- **Raw Dataset**: `raw-dataset.csv` (41,759 samples)
- **Labeled Dataset**: `hasil-labeling.csv` (41,887 samples)
- **Quality**: High confidence labels dengan DeepSeek API
- **Categories**: 4-level classification (Bukan/Ringan/Sedang/Berat)

---

*Dokumen ini akan menjadi panduan untuk reorganisasi dokumentasi proyek penelitian ujaran kebencian bahasa Jawa.*