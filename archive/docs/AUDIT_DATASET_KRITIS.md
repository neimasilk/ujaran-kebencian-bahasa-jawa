# AUDIT KRITIS: KUALITAS DATASET
## Deteksi Ujaran Kebencian Bahasa Jawa

**Tanggal Audit**: 5 Januari 2026
**Status**: MASALAH KRITIS DITEMUKAN

---

## RINGKASAN EKSEKUTIF

### STATUS: DATASET BERMASALAH SERIUS

| Masalah | Tingkat Keparahan | Dampak |
|---------|-------------------|--------|
| **Dataset adalah TERJEMAHAN** | KRITIS | Model belajar dari teks tidak natural |
| **44% teks bukan Jawa asli** | KRITIS | Noise sangat tinggi |
| **Data Leak 4.9%** | TINGGI | Evaluasi tidak valid |
| **3.88% label noise** | SEDANG | Confusion pada training |
| **Konteks budaya salah** | KRITIS | Model tidak relevan untuk Indonesia |

---

## 1. TEMUAN UTAMA: DATASET ADALAH TERJEMAHAN

### Bukti Kuat

Dataset ini **BUKAN** data asli dari media sosial Indonesia/Jawa, melainkan **TERJEMAHAN** dari dataset hate speech bahasa Inggris.

#### 1.1 Perbandingan Target Hate Speech

| Target Budaya BARAT | Jumlah | Target Budaya INDONESIA | Jumlah |
|---------------------|--------|-------------------------|--------|
| Women/Feminist | 4,305 | Sundanese | 1 |
| LGBTQ | 2,555 | Madurese | 0 |
| Blacks | 2,161 | Papuan | 1 |
| Muslims/Islam | 1,760 | Batak | 12 |
| Immigrants | 1,295 | Chinese-Indonesian | 547 |
| Jews/Jewish | 1,219 | Pribumi | 170 |
| Gypsies | 179 | - | - |
| Mexicans | 125 | - | - |

**Jika data asli dari Indonesia**, seharusnya:
- Lebih banyak referensi ke isu Cina-Indonesia (sensitif di Indonesia)
- Ada referensi ke suku lokal (Sunda, Madura, Papua, Batak)
- Ada referensi politik Indonesia (bukan Trump/Biden)
- TIDAK ada referensi ke Gypsy, Mexican, atau slang Inggris

#### 1.2 Contoh Teks yang Jelas Terjemahan

```
"Aku ora kerja keras supaya para Hispanik bisa njupuk kabeh keuntungan"
→ Terjemahan dari: "I don't work hard so Hispanics can take all the benefits"

"Aku ora bisa iki hard supaya sing chavs bisa njupuk kabeh keuntungan"
→ Terjemahan tidak lengkap, "chavs" adalah slang Inggris (British)

"Dalit minangka sampah masyarakat sing asor"
→ "Dalit" adalah istilah kasta India, tidak relevan untuk Indonesia

"mateni kabeh gypsies"
→ Gypsy tidak ada di Indonesia
```

#### 1.3 Referensi Budaya Barat Lainnya

| Referensi | Jumlah |
|-----------|--------|
| Nazi/Hitler | 307 |
| N-word (nigger/nigga) | 295 |
| US Politics (Trump/Biden/Obama) | 332 |
| KKK | 22 |
| Holocaust | 56 |
| Brexit | 39 |
| SJW/Libtard | 35 |

---

## 2. MASALAH KUALITAS TEKS

### 2.1 Analisis Bahasa

| Metrik | Nilai | Masalah |
|--------|-------|---------|
| Teks tanpa kata Jawa (<2 kata) | 17,594 (44.16%) | SANGAT TINGGI |
| Teks dominan Indonesia | 209 (0.52%) | - |
| Teks mengandung Inggris | 123 (0.31%) | - |

**44% teks kemungkinan bukan bahasa Jawa asli!**

### 2.2 Statistik Teks

- Panjang rata-rata: 102.2 karakter
- Panjang min: 10 karakter
- Panjang max: 2,386 karakter
- Teks kosong: 0

---

## 3. MASALAH LABEL

### 3.1 Noise Label

| Masalah | Jumlah | Persentase |
|---------|--------|------------|
| "Bukan Ujaran" dengan kata kasar | 753 | 3.88% |
| "Ujaran Berat" dengan kata netral | 847 | 13.18% |

### 3.2 Contoh Label Salah

**Label 0 (Bukan Ujaran) tapi mengandung hate speech:**
```
"Muga-muga dheweke mati amarga covid"
→ Ini JELAS hate speech, bukan netral!

"Nelpon sedulurku bajingan cilik iku apik tenan!"
→ Mengandung kata kasar, dilabel netral

"Kebodohan kolosal iki kudu digabung"
→ Bahasa merendahkan
```

### 3.3 Metode Pelabelan

| Metode | Jumlah | Persentase |
|--------|--------|------------|
| DeepSeek API (AI) | 22,930 | 54.9% |
| Rule-based (Otomatis) | 18,827 | 45.1% |
| **Anotator Manusia** | **0** | **0%** |

**TIDAK ADA verifikasi manusia!**

---

## 4. DATA LEAK

### 4.1 Overlap Train-Test

| Metrik | Nilai |
|--------|-------|
| Teks di train set | 19,339 |
| Teks di test set | 4,936 |
| **Teks yang BOCOR** | **242** |
| **Persentase leak** | **4.90%** |

**Ini berarti evaluasi model TIDAK VALID!**

---

## 5. SUMBER DATASET ASLI (HIPOTESIS)

Berdasarkan analisis, dataset kemungkinan berasal dari:

1. **HateXplain** - Dataset hate speech Inggris dengan penjelasan
2. **Davidson Hate Speech** - Dataset Twitter hate speech
3. **Measuring Hate Speech** - Dataset dengan skala severity
4. **SemEval-2019 Task 5** - Hate against immigrants and women

Dataset asli (Inggris) kemudian **diterjemahkan ke bahasa Jawa** menggunakan:
- Google Translate atau layanan terjemahan serupa
- Tanpa review oleh native speaker

---

## 6. DAMPAK PADA MODEL

### Mengapa Model Hanya Mencapai ~60% F1?

1. **Teks tidak natural** - Model belajar pola terjemahan, bukan bahasa Jawa asli
2. **Label noise** - ~4-13% label salah
3. **Konteks salah** - Hate speech Barat ≠ Hate speech Indonesia
4. **Data leak** - Evaluasi over-optimistic
5. **Distribusi tidak representatif** - Target groups tidak sesuai konteks lokal

### Mengapa `improved_model` Mencapai 86%?

Kemungkinan:
- Overfitting pada pola terjemahan
- Data leak yang lebih parah pada split lama
- Evaluasi pada data yang "mudah" karena pola repetitif

---

## 7. REKOMENDASI

### Opsi A: Buat Dataset Baru (DIREKOMENDASIKAN)

1. **Kumpulkan data asli** dari:
   - Twitter/X Indonesia dengan hashtag Jawa
   - Komentar YouTube channel Jawa
   - Forum diskusi lokal
   - Komentar berita online regional

2. **Pelabelan manual** oleh:
   - Native speaker bahasa Jawa
   - Minimal 3 annotator per sampel
   - Inter-annotator agreement > 0.7

3. **Target yang relevan** untuk Indonesia:
   - Isu SARA Indonesia
   - Politik lokal/nasional
   - Isu antar-suku/etnis
   - Isu agama dalam konteks Indonesia

### Opsi B: Curate Dataset yang Ada

1. **Filter** teks yang mengandung referensi Barat
2. **Hapus** teks dengan kata Inggris yang tidak diterjemahkan
3. **Re-label** dengan annotator manusia
4. **Hapus** data leak antara train/test

### Opsi C: Hybrid Approach

1. Gunakan dataset terjemahan sebagai **pre-training only**
2. Fine-tune dengan **data asli** (meski sedikit)
3. Fokus pada **transfer learning** dari konteks umum ke spesifik

---

## 8. KESIMPULAN

### Status Dataset Saat Ini

- **TIDAK LAYAK** untuk publikasi akademik tanpa disclaimer
- **TIDAK REPRESENTATIF** untuk hate speech bahasa Jawa asli
- **TIDAK VALID** sebagai benchmark karena data leak

### Langkah Wajib Sebelum Publikasi

1. Tambahkan disclaimer di paper bahwa data adalah terjemahan
2. Akui limitasi konteks budaya
3. Atau buat dataset baru yang autentik

### Catatan Penting

> Dataset ini mungkin berguna untuk **memahami pola umum hate speech**,
> tetapi **TIDAK COCOK** untuk deteksi hate speech bahasa Jawa dalam
> konteks nyata media sosial Indonesia.

---

## LAMPIRAN: Script Analisis

Script yang digunakan untuk audit ini: `analyze_dataset_quality.py`

---

**Diaudit oleh**: Claude Code
**Tanggal**: 5 Januari 2026
**Status Dokumen**: FINAL - KRITIS
