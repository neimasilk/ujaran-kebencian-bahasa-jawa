# ☁️ Panduan Eksperimen Cloud-Based (Google Colab + Drive)

Panduan ini dirancang agar Anda dapat menjalankan ulang eksperimen (Re-experiment) proyek Deteksi Ujaran Kebencian Bahasa Jawa dari komputer manapun (rumah, laptop, warnet) tanpa perlu memindahkan data menggunakan flashdisk.

## 📋 Konsep Dasar
*   **Kode Program:** Diambil otomatis dari GitHub (`neimasilk/ujaran-kebencian-bahasa-jawa`).
*   **Dataset:** Disimpan di Google Drive Anda pribadi.
*   **Komputasi (GPU):** Dijalankan menggunakan Google Colab (Gratis).

---

## 🚀 Langkah 1: Persiapan Data (Lakukan di Komputer Kantor)
Karena dataset asli ada di komputer kantor, Anda perlu mengunggahnya sekali saja ke Cloud.

1.  Buka **Google Drive** (drive.google.com) login dengan akun Google Anda.
2.  Buat folder baru dengan nama: `Javanese_Hate_Speech_Project`.
3.  Di dalam folder tersebut, buat folder `data`.
4.  **Upload file dataset Anda** (`balanced_dataset.csv`) ke dalam folder `data` tersebut.
    *   *Struktur di Drive:* `My Drive/Javanese_Hate_Speech_Project/data/balanced_dataset.csv`

---

## 💻 Langkah 2: Menjalankan Eksperimen (Di Komputer Manapun)

1.  Buka **[Google Colab](https://colab.research.google.com/)**.
2.  Pilih menu **File** > **Open Notebook**.
3.  Pilih tab **GitHub**.
4.  Di kolom pencarian, ketik username GitHub Anda: `neimasilk`.
5.  Pilih repositori: `ujaran-kebencian-bahasa-jawa`.
6.  Klik pada file: **`run_experiment_colab.ipynb`**.

---

## ▶️ Langkah 3: Eksekusi di Colab

Setelah notebook terbuka:

1.  **Connect Runtime:** Klik tombol "Connect" di pojok kanan atas. Pastikan terhubung ke GPU (Runtime > Change runtime type > T4 GPU).
2.  **Jalankan Cell 1 (Mount Drive):**
    *   Klik tombol Play.
    *   Akan muncul *pop-up* meminta izin akses Google Drive. Klik **Allow/Izinkan**. Ini penting agar Colab bisa membaca dataset yang Anda upload di Langkah 1.
3.  **Jalankan Cell 2 (Setup Project):**
    *   Ini akan men-download kode terbaru dari GitHub secara otomatis.
4.  **Jalankan Cell 3 (Link Dataset):**
    *   Pastikan path file di script sesuai dengan lokasi Anda upload di Drive. Script ini akan menyalin data dari Drive ke mesin Colab.
5.  **Jalankan Cell 4 (Training):**
    *   Ini akan melatih model dasar (*Improved Model*) menggunakan data Anda.
6.  **Jalankan Cell 5 (Ensemble Experiment):**
    *   Ini adalah eksperimen utamanya. Skrip akan berjalan otomatis menggunakan GPU Colab.

---

## 💾 Langkah 4: Menyimpan Hasil

Secara default, file di Colab akan hilang saat sesi ditutup. Gunakan **Cell 6** di notebook tersebut untuk menyalin kembali hasil eksperimen (`results/` dan `models/`) dari Colab ke Google Drive Anda (`results_backup`).

---

## 💡 Tips Tambahan
*   **Jika ada update kode:** Cukup push perubahan ke GitHub dari komputer manapun. Saat Anda membuka Colab lagi dan menjalankan Cell 2, ia akan otomatis mengambil kode versi terbaru.
*   **GPU:** Google Colab menyediakan GPU gratis (biasanya Tesla T4) yang jauh lebih cepat daripada CPU laptop biasa.

*Selamat bereksperimen!*
