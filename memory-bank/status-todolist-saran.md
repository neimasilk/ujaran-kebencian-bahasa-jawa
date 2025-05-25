# Status, To-Do List & Saran - Deteksi Ujaran Kebencian Bahasa Jawa

**Update Terakhir:** 26 Mei 2025, 04:00 WIB

## 1. Status Proyek Saat Ini:

* Tahap 0 (Pengaturan Awal Proyek) selesai.
* Dokumen perencanaan awal (`dokumen-desain-produk.md`, `tumpukan-teknologi.md`, `rencana-implementasi.md`, `architecture.md`) telah dibuat (versi 0.1).
* Proyek siap untuk memasuki Tahap 1: Setup Environment & Klarifikasi Awal.

## 2. To-Do List (Masa Depan - Berdasarkan `rencana-implementasi.md` MVP):

* **Fase 0: Setup Proyek & Lingkungan (Lanjutan)**
    * [ ] Inisialisasi repositori Git & GitHub.
    * [ ] Buat struktur folder proyek dasar.
    * [ ] Setup virtual environment (Python) dan install library awal.
    * [ ] Buat file `requirements.txt` awal.
* **Fase 1: Pengumpulan & Preprocessing Data Awal**
    * [ ] Identifikasi sumber data awal.
    * [ ] Lakukan pengumpulan data manual/script sederhana (~500-1000 sampel).
    * [ ] Kembangkan script dasar untuk preprocessing data.
    * [ ] Implementasikan fungsi filtering dasar untuk konten duplikat.
* **Fase 2: Pelabelan Data Awal & Persiapan Dataset**
    * [ ] Definisikan pedoman pelabelan awal.
    * [ ] Lakukan pelabelan manual pada sampel data awal (~200-500 sampel).
    * [ ] Bagi dataset berlabel menjadi set pelatihan dan validasi.
* **Fase 3: Pengembangan Model Dasar (Fine-tuning IndoBERT)**
    * [ ] Setup environment untuk Hugging Face Transformers dan PyTorch/TensorFlow.
    * [ ] Kembangkan script untuk tokenisasi data.
    * [ ] Kembangkan script untuk fine-tuning IndoBERT.
    * [ ] Simpan model yang sudah di-fine-tune.
* **Fase 4: Evaluasi Model Dasar**
    * [ ] Kembangkan script untuk prediksi pada set validasi.
    * [ ] Implementasikan perhitungan metrik evaluasi dasar (Akurasi).
    * [ ] (Opsional MVP) Implementasikan confusion matrix.
* **Fase 5: Pengembangan Prototipe API Sederhana**
    * [ ] Pilih framework API dan setup struktur dasar.
    * [ ] Buat endpoint API `/detect`.
    * [ ] (Opsional MVP) Buat antarmuka web sangat sederhana.

## 3. Saran "Baby-Step To-Do List" untuk Langkah Berikutnya:

* **Baby Step 1: Inisialisasi Proyek & Lingkungan Dasar**
    1.  Buat repositori baru di GitHub untuk proyek ini.
        * **Tes/Validasi:** Repositori berhasil dibuat dan dapat diakses.
    2.  Clone repositori ke lingkungan pengembangan lokal.
        * **Tes/Validasi:** Folder proyek ada di lokal dengan file `.git`.
    3.  Buat struktur folder proyek dasar:
        * `data/raw` (untuk data mentah yang dikumpulkan)
        * `data/processed` (untuk data yang sudah dibersihkan dan dilabeli)
        * `notebooks` (untuk eksperimen dan analisis)
        * `src` (untuk kode sumber Python)
            * `src/data_collection`
            * `src/preprocessing`
            * `src/modelling`
            * `src/api`
        * `models` (untuk menyimpan model yang sudah dilatih)
        * `tests` (untuk unit test)
        * `docs` (jika diperlukan selain memory-bank)
        * **Tes/Validasi:** Struktur folder sesuai dengan yang didefinisikan.
    4.  Inisialisasi virtual environment Python (misalnya, menggunakan `venv`:
        * Pastikan Anda memiliki Python versi 3.8+ terinstall di sistem Anda (sesuai `tumpukan-teknologi.md`). Anda bisa memeriksanya dengan `python --version` atau `python3 --version`.
        * Jalankan `python -m venv .venv` atau `python3 -m venv .venv` jika perintah `python` Anda merujuk ke versi Python 2.x, dan aktivasi).
        * **Tes/Validasi:** Virtual environment aktif, prompt terminal berubah. Versi Python di dalam venv sesuai.
    5.  Install library Python dasar: `pandas`, `numpy`, `jupyterlab`, `scikit-learn`.
        * **Tes/Validasi:** `pip freeze` menunjukkan library terinstall. Dapat di-import di sesi Python.
    6.  Buat file `requirements.txt` awal dari library yang terinstall (`pip freeze > requirements.txt`).
        * **Tes/Validasi:** File `requirements.txt` berisi daftar library.
    7.  Buat file `.gitignore` dasar (misalnya, untuk `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`, `data/raw/*` jika data mentah besar dan tidak di-commit).
        * **Tes/Validasi:** File `.gitignore` ada.
    8.  Commit semua perubahan awal ini ke Git dan push ke GitHub.
        * **Tes/Validasi:** Perubahan terlihat di repositori GitHub.

--- 