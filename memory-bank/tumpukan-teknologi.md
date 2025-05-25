# Tumpukan Teknologi (Tech Stack) - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

**Versi:** 0.1
**Tanggal:** 26 Mei 2025

Proyek ini akan memanfaatkan kombinasi teknologi yang sudah teruji untuk Pemrosesan Bahasa Alami (NLP) dan pengembangan aplikasi web.

## 1. Komponen Inti (Machine Learning & NLP)

* **Model Dasar:**
    * **BERT (Bidirectional Encoder Representations from Transformers)** [cite: 19, 22]
        * Akan menggunakan model *pre-trained* **IndoBERT** sebagai titik awal karena kedekatan leksikal antara Bahasa Indonesia dan Bahasa Jawa. [cite: 58, 100]
        * Selanjutnya akan dilakukan *fine-tuning* dengan dataset Bahasa Jawa yang spesifik. [cite: 100, 101]
* **Bahasa Pemrograman (ML/NLP):**
    * **Python (versi 3.8+)**: Bahasa utama untuk pengembangan model, preprocessing data, dan scripting.
* **Frameworks & Libraries (ML/NLP):**
    * **Hugging Face Transformers:** Untuk mengakses, memodifikasi, dan melatih model BERT (IndoBERT).
    * **PyTorch** atau **TensorFlow/Keras:** Sebagai backend untuk Hugging Face Transformers dan untuk membangun lapisan kustom jika diperlukan. (Pilihan akan ditentukan berdasarkan kemudahan penggunaan IndoBERT dan preferensi tim).
    * **Scikit-learn:** Untuk metrik evaluasi (akurasi, presisi, recall, F1-score [cite: 54, 106]), pembagian dataset, dan mungkin beberapa model baseline untuk perbandingan.
    * **Pandas:** Untuk manipulasi dan analisis data tabular (dataset).
    * **NumPy:** Untuk operasi numerik.
    * **NLTK (Natural Language Toolkit) / SpaCy / Sastrawi (atau library sejenis):** Untuk tugas preprocessing teks seperti tokenisasi, stemming (jika relevan untuk Bahasa Jawa), stopword removal (dengan daftar stopword Bahasa Jawa kustom). [cite: 24, 93, 94]
    * **Matplotlib / Seaborn:** Untuk visualisasi data dan hasil analisis model.

## 2. Pengembangan Prototipe API & Web

* **Bahasa Pemrograman (Backend API):**
    * **Python**
* **Framework API:**
    * **Flask** atau **FastAPI**[cite: 108]: Dipilih karena ringan, cepat, dan mudah diintegrasikan dengan model Python ML. FastAPI dipertimbangkan untuk performa dan validasi data otomatis.
* **Antarmuka Pengguna (Frontend - Sederhana):**
    * **HTML, CSS, JavaScript (Vanilla JS atau framework ringan seperti Vue.js/React jika diperlukan untuk interaktivitas lebih):** Untuk membangun antarmuka pengguna berbasis web yang sederhana untuk input teks dan menampilkan hasil klasifikasi. [cite: 109]

## 3. Alat Pendukung & Lingkungan

* **Lingkungan Pengembangan:**
    * **IDE:** VS Code (dengan ekstensi Python, Pylance, Jupyter), atau Cursor (seperti yang disebutkan dalam Vibe Coding).
    * **Jupyter Notebooks / Google Colab:** Untuk eksperimentasi model, analisis data, dan visualisasi awal.
* **Manajemen Versi:**
    * **Git**
    * **GitHub:** Sebagai repositori kode utama.
* **Manajemen Ketergantungan (Python):**
    * **pip** dengan `requirements.txt` atau virtual environment tools seperti `venv` atau `conda`.
* **Server (untuk prototipe API):**
    * Server pengembangan Flask/FastAPI bawaan untuk development lokal.
    * Platform seperti Heroku, PythonAnywhere, atau Docker container di cloud service (AWS, GCP, Azure) untuk deployment prototipe jika diperlukan untuk pengujian publik.

## 4. Pengumpulan Data

* Akses ke API media sosial (Twitter API, Facebook Graph API - jika memungkinkan dan sesuai kebijakan).
* Web scraping tools (misalnya, BeautifulSoup, Scrapy dengan Python) untuk mengumpulkan data dari forum online jika API tidak tersedia (dengan memperhatikan etika dan file `robots.txt`). [cite: 56]

## 5. Pertimbangan

* **Skalabilitas:** Meskipun MVP mungkin tidak memerlukan skalabilitas tinggi, pemilihan FastAPI dapat membantu jika di masa depan ada kebutuhan untuk menangani lebih banyak request.
* **Kemudahan Integrasi:** Pemilihan stack Python secara keseluruhan (ML dan API) akan memudahkan integrasi antar komponen.
* **Sumber Daya Komputasi:** Pelatihan/fine-tuning model BERT memerlukan GPU. Google Colab Pro, Kaggle Kernels, atau akses ke layanan cloud dengan GPU akan dipertimbangkan. [cite: 81] (Meskipun TKT 3 lebih ke formulasi konsep, persiapan sumber daya komputasi penting untuk implementasi aktual).

--- 