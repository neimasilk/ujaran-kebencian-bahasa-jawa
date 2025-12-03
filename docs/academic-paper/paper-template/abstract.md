# Abstract Template - Javanese Hate Speech Detection

## Abstract (English)

**Background:** Hate speech detection in regional languages poses unique challenges due to limited resources, cultural nuances, and linguistic complexity. Javanese, spoken by over 75 million people, lacks comprehensive hate speech detection systems despite increasing online presence.

**Objective:** This study develops and evaluates transformer-based models for detecting hate speech in Javanese text, comparing multiple architectures and optimization strategies to establish effective baselines for low-resource language hate speech detection.

**Methods:** We compiled a dataset of 41,887 Javanese text samples labeled into four categories: non-hate speech, mild hate speech, moderate hate speech, and severe hate speech. Six transformer models were evaluated: IndoBERT Base/Large, mBERT, and XLM-RoBERTa with various configurations. Models were assessed using balanced evaluation sets to mitigate class imbalance bias, with F1-Macro score as the primary metric.

**Results:** XLM-RoBERTa with optimized configuration achieved the best performance (F1-Macro: 61.86%, Accuracy: 61.95%), followed by IndoBERT Large v1.2 (F1-Macro: 60.75%, Accuracy: 63.05%). Configuration optimization proved more impactful than architecture selection, with improvements up to 25.47% through hyperparameter tuning. Cross-lingual models demonstrated superior transfer learning capabilities compared to Indonesian-specific models.

**Conclusions:** Transformer models can effectively detect Javanese hate speech with proper optimization. Cross-lingual pre-trained models show excellent transfer capabilities for low-resource languages. Configuration optimization is critical for achieving optimal performance, often more important than model architecture selection. The established framework provides a foundation for hate speech detection in other regional languages.

**Keywords:** Hate Speech Detection, Javanese Language, Transformer Models, Cross-lingual Transfer Learning, Low-resource NLP, Class Imbalance

---

## Abstrak (Bahasa Indonesia)

**Latar Belakang:** Deteksi ujaran kebencian dalam bahasa daerah menghadapi tantangan unik karena keterbatasan sumber daya, nuansa budaya, dan kompleksitas linguistik. Bahasa Jawa yang dituturkan oleh lebih dari 75 juta orang masih kekurangan sistem deteksi ujaran kebencian yang komprehensif meskipun kehadiran online yang meningkat.

**Tujuan:** Penelitian ini mengembangkan dan mengevaluasi model berbasis transformer untuk mendeteksi ujaran kebencian dalam teks bahasa Jawa, membandingkan berbagai arsitektur dan strategi optimisasi untuk menetapkan baseline yang efektif bagi deteksi ujaran kebencian bahasa dengan sumber daya terbatas.

**Metode:** Kami menyusun dataset berisi 41.887 sampel teks bahasa Jawa yang dilabeli ke dalam empat kategori: bukan ujaran kebencian, ujaran kebencian ringan, sedang, dan berat. Enam model transformer dievaluasi: IndoBERT Base/Large, mBERT, dan XLM-RoBERTa dengan berbagai konfigurasi. Model dinilai menggunakan set evaluasi seimbang untuk mengurangi bias ketidakseimbangan kelas, dengan skor F1-Macro sebagai metrik utama.

**Hasil:** XLM-RoBERTa dengan konfigurasi optimal mencapai performa terbaik (F1-Macro: 61,86%, Akurasi: 61,95%), diikuti oleh IndoBERT Large v1.2 (F1-Macro: 60,75%, Akurasi: 63,05%). Optimisasi konfigurasi terbukti lebih berpengaruh daripada pemilihan arsitektur, dengan peningkatan hingga 25,47% melalui penyetelan hiperparameter. Model lintas-bahasa menunjukkan kemampuan transfer learning yang superior dibandingkan model khusus Indonesia.

**Kesimpulan:** Model transformer dapat secara efektif mendeteksi ujaran kebencian bahasa Jawa dengan optimisasi yang tepat. Model pre-trained lintas-bahasa menunjukkan kemampuan transfer yang sangat baik untuk bahasa dengan sumber daya terbatas. Optimisasi konfigurasi sangat penting untuk mencapai performa optimal, seringkali lebih penting daripada pemilihan arsitektur model. Framework yang ditetapkan memberikan fondasi untuk deteksi ujaran kebencian dalam bahasa daerah lainnya.

**Kata Kunci:** Deteksi Ujaran Kebencian, Bahasa Jawa, Model Transformer, Transfer Learning Lintas-bahasa, NLP Sumber Daya Terbatas, Ketidakseimbangan Kelas

---

## Writing Guidelines

### Abstract Structure (150-250 words)
1. **Background (2-3 sentences):** Context dan problem statement
2. **Objective (1-2 sentences):** Research goals dan contributions
3. **Methods (3-4 sentences):** Dataset, models, evaluation methodology
4. **Results (3-4 sentences):** Key findings dengan specific numbers
5. **Conclusions (2-3 sentences):** Implications dan broader impact

### Key Numbers to Include
- **Dataset Size:** 41,887 samples
- **Models Evaluated:** 6 transformer variants
- **Best Performance:** 61.86% F1-Macro (XLM-RoBERTa)
- **Improvement Range:** Up to 25.47% dengan optimization
- **Categories:** 4-class classification

### Important Phrases
- "transformer-based models"
- "cross-lingual transfer learning"
- "low-resource language"
- "class imbalance mitigation"
- "configuration optimization"
- "balanced evaluation methodology"

### Avoid
- Technical jargon tanpa explanation
- Overly detailed methodology
- Speculation beyond results
- Redundant information
- Vague statements tanpa numbers

---

**Template Usage:**
1. Customize background untuk target venue
2. Adjust emphasis berdasarkan audience
3. Update numbers jika ada additional experiments
4. Tailor keywords untuk journal requirements
5. Ensure word count compliance

**Review Checklist:**
- [ ] Clear problem statement
- [ ] Specific methodology description
- [ ] Quantitative results included
- [ ] Broader implications stated
- [ ] Keywords relevant dan comprehensive
- [ ] Word count within limits
- [ ] No grammatical errors
- [ ] Consistent terminology