# Papan Proyek - Sistem Deteksi Ujaran Kebencian Bahasa Jawa

### STATUS [Update: 2024-12-29]
- ✅ Proyek telah disiapkan sesuai dengan Vibe Coding Guide v1.4
- ✅ Spesifikasi produk telah diperbaiki dan disesuaikan dengan template PRD
- ✅ Tim manifest telah disiapkan dengan peran yang jelas
- ✅ Environment setup dan modular code structure telah diimplementasi
- ✅ Dataset inspection dan basic testing infrastructure telah selesai
- 🔄 Dokumentasi sedang disesuaikan dengan panduan terbaru

### REFERENSI ARSIP
- Baby-step sebelumnya: Implementasi Testing dan Dokumentasi API (selesai)
- Arsip lengkap tersedia di: `baby-steps-archive/`

### BABY-STEP SAAT INI

**"Persiapan Data Labeling dan Model Development"** 📋 SIAP DIMULAI
- **Tujuan:** Mempersiapkan dataset berlabel dan infrastruktur untuk pengembangan model machine learning.
- **Tugas:**
     - [ ] **T1:** Setup data labeling workflow | **File:** `src/data/labeling_workflow.py` | **Tes:** Script dapat memuat data dan menyediakan interface labeling | **Assignee:** Mukhlis Amien, jules_dev1
     - [ ] **T2:** Implementasi data validation dan quality checks | **File:** `src/data/validation.py` | **Tes:** Validasi kualitas label dengan metrics yang jelas | **Assignee:** jules_dev1
     - [ ] **T3:** Setup model training infrastructure | **File:** `src/models/` | **Tes:** Environment siap untuk fine-tuning IndoBERT | **Assignee:** jules_dev2
     - [ ] **T4:** Dokumentasi proses labeling dan training | **File:** `memory-bank/model-development.md` | **Tes:** Dokumentasi lengkap untuk reproduksi | **Assignee:** jules_dokumen

### SARAN & RISIKO
- **Saran:** Implementasikan auto-generation untuk summary proyek menggunakan script
- **Saran:** Pastikan semua referensi antar dokumen menggunakan path yang benar
- **Risiko:** Inkonsistensi dokumentasi dapat menyebabkan kebingungan tim di tahap selanjutnya
- **Risiko:** Perubahan struktur mungkin memerlukan update pada script dan konfigurasi yang ada