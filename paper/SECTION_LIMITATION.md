# Limitations and Future Work

## Limitations

### 6.1 Dataset Limitations

**Label Subjectivity**

Klasifikasi tingkat keparahan ujaran kebencian bersifat inherently subjektif. Batas antara "Light Hate" dan "Moderate Hate" sering kali kabur dan bergantung pada konteks serta interpretasi penilai. Dalam penelitian ini, kami menggunakan dua orang annotator native speaker bahasa Jawa dengan Cohen's kappa inter-annotator agreement of 0.72, yang menunjukkan "substantial agreement" namun bukan "near-perfect agreement". Hal ini menyiratkan bahwa sekitar 28% label mungkin berubah jika di-annotasi oleh orang yang berbeda.

**LLM-Generated Data Bias**

Sebanyak 52.3% (5,240 dari 10,019) dari dataset kami dihasilkan menggunakan LLM (DeepSeek-Coder-V2). Proporsi data sintetis yang tinggi ini merupakan limitasi signifikan. Meskipun kami melakukan quality check dan aggressive cleanup (menghasilkan 9,775 sampel bersih), data sintetis mungkin memperkenalkan bias yang spesifik untuk model LLM tersebut. Model seperti DeepSeek dilatih pada corpus internet yang mungkin mengandung stereotip dan bias budaya yang kemudian tertuang dalam generated text. Selain itu, LLM mungkin tidak sepenuhnya menangkap nuansa budaya dan bahasa Jawa yang halus. Ketergantungan tinggi pada data sintetis berarti performa model mungkin tidak sepenuhnya merepresentasikan kemampuan deteksi pada ujaran kebencian alami di media sosial.

**Domain Coverage**

Dataset kami dikumpulkan primarily dari Twitter/X dan Instagram. Hal ini berarti variasi ujaran kebencian di platform lain seperti Facebook, TikTok, atau forum online (Kaskus, Reddit) mungkin tidak terwakili secara memadai. Setiap platform memiliki culture, norm, dan style komunikasi yang berbeda yang mempengaruhi bagaimana ujaran kebencian diekspresikan.

**Temporal Stability**

Data dikumpulkan pada periode 2023-2024. Ujaran kebencian di media sosial berkembang dengan cepat - istilah, slang, dan meme baru muncul dan menjadi tidak relevan dalam hitungan bulan. Model yang dilatih pada data kami mungkin tidak efektif untuk mendeteksi ujaran kebencian "baru" yang menggunakan meme atau slang kontemporer.

### 6.2 Methodological Limitations

**Text-Only Analysis**

Penelitian ini hanya menganalisis teks. Dalam praktiknya, ujaran kebencian sering kali bersifat multimodal - menggabungkan teks dengan gambar, meme, atau video. Sebagai contoh, sebuah meme yang tampak tidak berbahaya secara visual dapat menjadi sangat ofensif ketika dipasangkan dengan caption tertentu. Model kami tidak dapat menangkap konteks multimodal ini.

**Lack of Context**

Kami menganalisis setiap posting secara terisolasi tanpa konteks percakapan sebelumnya. Dalam beberapa kasus, sebuah statement yang tampak seperti ujaran kebencian mungkin adalah quote, sarcasm, atau satire yang hanya dapat dipahami dengan konteks percakapan yang lebih luas. Sebaliknya, beberapa ujaran kebencian halus (dog-whistle) hanya dapat diidentifikasi dengan mengetahui sejarah interaksi antar pengguna.

**Single Language Focus**

Penelitian ini berfokus pada bahasa Jawa. Dalam praktiknya, media sosial Indonesia sering kali menampilkan *code-switching* dan *code-mixing* antara bahasa Indonesia, bahasa Jawa, bahasa Inggris, dan bahasa daerah lain. Model kami mungkin tidak bekerja dengan baik pada teks yang sangat mixed.

**Generalization to Other Malay Languages**

Meskipun IndoBERT dilatih pada corpus bahasa Indonesia, dan bahasa Jawa termasuk rumpun Austronesian seperti bahasa Indonesia, tidak dijamin bahwa pendekatan kami akan generalisasi dengan baik ke bahasa daerah Indonesia lainnya seperti Sundanese, Madurese, atau Minangkabau yang memiliki struktur gramatikal dan vocabulary yang berbeda secara signifikan.

### 6.3 Evaluation Limitations

**Single Test Set**

Kami menggunakan single test set dengan 978 samples (10% dari 9,775 sampel bersih, stratified split). Idealnya, kami melakukan evaluasi menggunakan multiple test sets dari berbagai sumber dan periode waktu untuk mengukur generalisasi yang lebih robust. Selain itu, test split berasal dari distribusi yang sama dengan training data, yang berarti evaluation kami mungkin overestimate performa sebenarnya pada data yang sangat berbeda.

**Statistical Significance**

Untuk mengatasi keterbatasan single-run evaluation, kami melakukan evaluasi dengan 5 random seeds berbeda (42, 123, 456, 789, 1024) pada model terbaik (XLM-RoBERTa Large) dan melaporkan mean ± standard deviation. Dari 5 seeds, 4 seeds menghasilkan performa konsisten (F1=80.83% ± 1.74%), namun 1 seed (1024) mengalami training collapse (F1=11.07%), menunjukkan instabilitas training yang mungkin terjadi pada model besar. Evaluasi pada external test set tetap diperlukan untuk validasi yang lebih kuat.

**No Human Evaluation**

Kami hanya melaporkan automatic metrics (F1, Precision, Recall). Kami tidak melakukan human evaluation dari model predictions - misalnya, dengan meminta expert untuk menilai apakah prediksi model masuk akal atau apakah error yang dibuat model dapat dimaafkan ("understandable mistakes").

### 6.4 Ethical Considerations

**Potential Misuse**

Model yang kami kembangkan secara teoritis dapat disalahgunakan - misalnya, oleh pemerintah atau otoritas untuk secara otomatis memantau dan membungkam kritik yang sah, atau oleh perusahaan untuk memfilter konten secara berlebihan. Kami berkomitmen pada prinsip bahwa model kami hanya untuk tujuan research dan tidak boleh digunakan untuk sensor otomatis tanpa human oversight.

**Bias in Training Data**

Seperti kebanyakan dataset hate speech, dataset kami mungkin mengandung bias. Sebagai contoh, ujaran kebencian terhadap kelompok tertentu mungkin over-represented atau under-represented dalam data. Ini dapat menyebabkan model menjadi lebih sensitif terhadap ujaran kebencian terhadap kelompok tertentu dibandingkan kelompok lain, yang secara tidak adil dapat mempengaruhi konten moderation decisions.

**Privacy Concerns**

Meskipun kami menghapus username dan identitas personal, data dikumpulkan dari public social media posts. Pengguna mungkin tidak menyadari atau menyetujui bahwa posting mereka digunakan untuk research training data. Kami mengikuti ethical guidelines dengan hanya menggunakan public data, namun concerns tentang consent tetap ada dalam domain social media research.

---

## Future Work

### 7.1 Multimodal Hate Speech Detection

Mengembangkan model yang dapat menganalisis ujaran kebencian multimodal yang menggabungkan teks, gambar, dan video. Ini akan memerlukan:
- Dataset multimodal bahasa Jawa
- Model vision-language seperti CLIP yang diadaptasi untuk bahasa Jawa
- Evaluasi pada meme dan video content

### 7.2 Context-Aware Detection

Mengembangkan model yang mempertimbangkan konteks percakapan dan sejarah interaksi antar pengguna. Pendekatan yang mungkin:
- Thread-level modeling dengan transformer models
- Incorporating user embeddings dan social graph information
- Temporal modeling untuk mendeteksi escalating behavior

### 7.3 Cross-Lingual Transfer

Mengeksplorasi apakah model yang dilatih pada bahasa Jawa dapat transfer ke bahasa daerah Indonesia lainnya:
- Fine-tuning experiments pada Sundanese, Madurese, Minangkabau
- Multilingual models untuk bahasa daerah Indonesia
- Analysis of linguistic similarities dan differences

### 7.4 Explainability dan Interpretability

Mengembangkan methods untuk menjelaskan mengapa model mengklasifikasikan sesuatu sebagai ujaran kebencian:
- Attention visualization untuk mengidentifikasi kata-kata kunci
- LIME/SHAP explanations untuk individual predictions
- Rule-based extraction untuk human-understandable patterns

### 7.5 Continual Learning

Mengembangkan sistem yang dapat belajar dari ujaran kebencian "baru" tanpa forgetting patterns lama:
- Incremental learning untuk new slang dan meme
- Disaster learning methods untuk unlearning harmful patterns
- Active learning untuk human-in-the-loop continuous improvement

### 7.6 Fairness Auditing

Melakukan audit fairness yang lebih komprehensif:
- Evaluate model performance across different demographic groups
- Test for bias terhadap kelompok tertentu
- Develop bias mitigation techniques spesifik untuk bahasa Jawa

---

**Word Count**: ~1,300 words
