# JIHST: Javanese Implicit Hate Speech Taxonomy (DRAFT)
## Taksonomi Implicit Hate Speech Bahasa Jawa

**Version:** 0.1 (Draft)
**Date:** 2025-01-08
**Status:** Initial Development

---

## I. Overview

**Definisi Implicit Hate Speech:**

> Ujaran kebencian yang disampaikan secara tidak langsung melalui bahasa halus, sindiran, metafora, atau kode budaya yang memerlukan pemahaman konteks sosial-budaya untuk menginterpretasikan maksud sebenarnya.

**Perbedaan dengan Explicit Hate Speech:**

| Aspek | Explicit | Implicit |
|-------|----------|----------|
| Bentuk | Langsung, terbuka | Halus, terselubung |
| Contoh | "Orang Jawa bodoh!" | "Budaya Jawa perlu modernisasi" |
| Deteksi | Relatif mudah | Memerlukan konteks |
| Plausible Deniability | Rendah | Tinggi |

---

## II. Codetypes (Adapted from Wei et al., 2025)

### II.1 Established Codetypes (Universal)

#### 1. Stereotyping
Penggunaan generalisasi negatif tentang kelompok tanpa menyebut secara eksplisit.

**Contoh Jawa:**
- "Wong Jawa iku sok alim ngemene dewe" (Orang Jawa suka pura-pula alim)
- "Arek-arek Suroboyo jago ngetawaine" (Anak Surabaya jago menertawakan)

#### 2. Othering
Membuat pemisahan antara "kita" vs "mereka" tanpa eksplisit.

**Contoh Jawa:**
- "Ing Jakarta wong kaya kita dianggap wong deso" (Di Jakarta orang seperti kita dianggap orang desa)
- Use of pronouns: "kowe" (kamu - kasar) vs "kita" (kita - inklusif)

#### 3. Threatening
Ancaman terselubung yang bisa disangkal.

**Contoh Jawa:**
- "Aja salah pithing, ngko ketularan" (Jangan salah pilih, nanti ketularan)
- "Ojo nganti lali sejarah" (Jangan sampai lupa sejarah - implying threat)

#### 4. Dehumanizing
Menyangkal kemanusiaan melalui bahasa halus.

**Contoh Jawa:**
- "Dadi wong kudu ngerti papan" (Jadi orang harus tahu tempat - implying some aren't fully human)
- "Sing ngerti adat jowo" (Yang mengerti adat Jawa - implying others don't count)

#### 5. Mocking
Ejekan melalui humor atau sindiran.

**Contoh Jawa:**
- "Wong pinter kalah karo wong bener" (Orang pintar kalah sama orang benar - sarcastic)
- "Mbok seneng-seneng wae" (Silakan senang saja - dismissive)

#### 6. Blaming
Menyalahkan kelompok untuk masalah sosial.

**Contoh Jawa:**
- "Yen nagih rezeki ki wong sekolah" (Kalau minta rezeki orang yang sekolah - blaming educated people)
- "Wong pinggiran kebawa arus" (Orang pinggiran terbawa arus - blaming marginalized groups)

---

### II.2 Javanese-Specific Codetypes (NOVEL)

#### 1. Hierarchical Inversion (Inversi Hierarki)

**Definisi:** Membalikkan hierarki sosial yang dipahami secara budaya untuk merendahkan atau mengkritik.

**Mekanisme:**
- Mengklaim kelompok yang secara tradisional "di bawah" sekarang menguasai
- Ironi sosial tentang status

**Contoh:**
| Teks | Makna Implicit |
|------|----------------|
| "Wong cilik iku sing nyekel negara kok, dadi wong gedhe ra kuat ngapa-ngapa" | Mengklaim "orang kecil" menguasai negara, mengkritik kekuasaan baru |
| "Jaman saiki wong deso luwih akeh dhuwite tinimbang wong kutho" | Ironi tentang perubahan ekonomi sosial |

**Target:** Kelompok sosial, kelas ekonomi, status tradisional

#### 2. Register Manipulation (Manipulasi Tingkatan Bahasa)

**Definisi:** Menggunakan tingkatan bahasa Jawa (Ngoko/Madya/Krama) secara tidak sesuai konteks untuk menyampaikan ketidaksukaan atau penghinaan.

**Mekanisme:**
- Menggunakan **Ngoko** kepada orang yang seharusnya dipakai **Krama**
- Menggunakan **Krama** secara sarcastic untuk mengejek
- Pelanggaran sopan santun sebagai bentuk hostility

**Contoh:**
| Teks | Register | Pelanggaran | Makna Implicit |
|------|----------|-------------|----------------|
| "Kowe oke wae" (to elder) | Ngoko | Harusnya Krama | Tidak menghormati, sengaja menghina |
| "Nderek menawa kersanipun" (sarcastic to peer) | Krama | Terlalu formal | Mengejek dengan bahasa berlebihan |

**Deteksi Technical:**
- Speech level classifier diperlukan
- Context: hubungan pembicara-pendengar

**Target:** Orang lebih tua, superior, orang yang dihormati

#### 3. Cultural Metaphor (Metafora Budaya)

**Definisi:** Menggunakan metafora budaya Jawa yang menyiratkan ketidaksukaan tanpa menyebut langsung.

**Mekanisme:**
- Metafora wayang, gamelan, adat
- Peribahasa Jawa dengan konotasi negatif
- Referensi budaya yang tidak dimengerti orang luar

**Contoh:**
| Teks | Metafora | Makna Implicit |
|------|----------|----------------|
| "Kaya kucing kepentok wedhus" | Kucing ketemu kambing | Mengejek ketidakmampuan beradaptasi |
| "Wong ngalegi nganti ora mangerteni awak dhewe" | Orang sombong sampai lupa diri | Mengkritik kesombongan |
| "Madhang arang-arang putunge" | Makan arang, kotoran nempel | Menuduh munafik |

**Catatan:** Perlu pengetahuan budaya Jawa untuk menginterpretasikan

**Target:** Sifat pribadi, perilaku sosial

#### 4. Religious Syncretism Abuse (Penyalahgunaan Sinkretisme Agama)

**Definisi:** Memanfaatkan sinkretisme agama Jawa-Islam-Hindu untuk menyindir atau menghina.

**Mekanisme:**
- Mengejek praktik kejawen
- Referensi negatif ke tradisi lama
- Mengkontraskan "Islam modern" vs "Jawa kuno"

**Contoh:**
| Teks | Makna Implicit |
|------|----------------|
| "Masih nyimpen banyangan ruwat" | Masih percaya hal takhayul (menghina tradisi Jawa) |
| "Wong jowo iku mbaurkene Islam karo kejawen" | Kritik terhadap praktik keagamaan Jawa |
| "Sing ngerti adat kuno" | Menyinggung orang yang masih memegang tradisi lama |

**Target:** Penganut Kejawen, praktik tradisional, orang desa

#### 5. Dialectic Othering (Othering Berbasis Dialek)

**Definisi:** Diskriminasi atau ejekan berdasarkan dialek Jawa (Banyumasan, Arekan, Mataraman).

**Mekanisme:**
- Ngejek logat atau kosakata daerah
- Stereotype regional

**Contoh:**
| Teks | Target | Makna Implicit |
|------|--------|----------------|
| "Arek-arek iku oake banget ngomonge" | Arekan (Surabaya) | Orang Surabaya kasar/berisik |
| "Wong Banyumas ra iso ngarteni basa sing bener" | Banyumasan | Orang Banyumas tidak mengerti bahasa yang benar |
| "Wong solo iku kaku krama-e" | Mataraman | Orang Solo kaku/kaku dalam bahasa halus |

**Regional Stereotypes:**
- **Banyumasan:** "Cemplung", "ora ngerti halus"
- **Arekan:** "Oake", "brutal", "terus terang"
- **Mataraman:** "kaku", "arrogant", "sombong"

**Target:** Orang dari daerah tertentu

#### 6. Historical Trauma Signaling (Sinyal Trauma Historis)

**Definisi:** Menggunakan referensi sejarah yang menyakitkan untuk menyampaikan kebencian tanpa menyebut eksplisit.

**Mekanisme:**
- Referensi konflik historis (1965, kolonial)
- Trigger memori kolektif
- Kode yang dimengerti kelompok tertentu

**Contoh:**
| Teks | Referensi | Makna Implicit |
|------|-----------|----------------|
| "Wong kang tau nggawé gawuk" | 1965 tragedy | Menuding anggota PKI/komunis tanpa menyebut |
| "Sing takluk karo tanah seberang" | Kolonial | Menghina yang pernah bekerja sama dengan penjajah |
| "Wong loro banyune" | Dukun santet | Menuduh praktik ilmu hitam |

**Catatan:** Sangat context-dependent, memerlukan pengetahuan sejarah

---

## III. Annotation Scheme

### III.1 Label Structure

Setiap sampel akan diberi label multi-dimensi:

```json
{
  "text": "contoh teks",
  "hate_binary": 0,           // 0=Non-Hate, 1=Hate
  "implicit_type": 0,         // 0=Explicit, 1=Implicit, 2=Borderline
  "codetypes": [1, 5],        // List of applicable codetypes
  "target": ["etnis", "sosial"], // Target identity
  "severity": 0.65,            // 0.0-1.0 continuous
  "context_required": true,    // Need cultural context?
  "explanation": "penjelasan..." // Why labeled this way
}
```

### III.2 Decision Tree

```
Apakah teks mengandung ujaran kebencian?
├── Tidak → Label: Non-Hate (0)
└── Ya →
    ├── Apakah eksplisit (kata kasar, ancaman langsung)?
    │   ├── Ya → Explicit Hate
    │   └── Tidak → Implicit Hate
    │       → Identifikasi Codetype(s)
    │       → Identifikasi Target
    │       → Severity Scoring
```

---

## IV. Examples for Annotation

### IV.1 Clear Examples

| # | Teks | Binary | Implicit | Codetypes | Target | Severity |
|---|------|--------|----------|-----------|--------|----------|
| 1 | "Wong cilik iku sing nyekel negara" | 1 | 1 | [1,7] | Sosial | 0.6 |
| 2 | "Kowe oke wae, pak" | 1 | 1 | [8] | Status | 0.4 |
| 3 | "Arek-arek iku oake banget" | 1 | 1 | [1,10] | Regional | 0.5 |
| 4 | "Masih nyimpen banyangan ruwat" | 1 | 1 | [2,9] | Agama | 0.7 |
| 5 | "Sugeng enjing, piye kabare?" | 0 | - | - | - | - |

### IV.2 Borderline Cases (Requires Discussion)

1. **"Wong Jawa iku alu wae"** - Stereotype positif atau negatif?
2. **"Ngoko karo wong tuo iku ora sopan"** - Pernyataan fakta atau implicit hate?
3. **"Aku seneng karo wong Suroboyo, jujur wae"** - Genuine preference atau Othering?

---

## V. Quality Control

### V.1 Inter-Annotator Agreement

- **Target:** Cohen's Kappa ≥ 0.80
- **Pilot:** 100 samples, 3 annotators
- **Final:** Full dataset dengan adjudication

### V.2 Expert Validation

- Ahli bahasa Jawa untuk validasi cultural context
- Sosiolog untuk validasi social hierarchy aspects

### V.3 Disagreement Resolution

- Round table discussion
- Document edge cases
- Update guidelines iteratively

---

## VI. Next Steps

1. **Pilot Study** - Anotasi 100 samples dengan 3 annotators
2. **Guidelines Refinement** - Perbaiki definisi berdasarkan pilot
3. **Expert Consultation** - Validasi dengan ahli bahasa/sosiolog
4. **Full Annotation** - Mulai anotasi dataset penuh

---

## VII. References

- Wei et al. (2025). "Cracking the Code: Enhancing Implicit Hate Speech Detection through Coding Classification"
- ElSherief et al. (2021). "Latent Hatred: A Benchmark for Understanding Implicit Hate Speech"
- Poedjosoedarmo (1979). "Javanese Speech Levels"
- Errington (1988). "Structure and Style in Javanese"

---

**Document Status:** DRAFT - Needs validation and refinement
**Next Review:** After pilot annotation
**Owner:** Research Team
