# Historical Results

Folder ini berisi hasil eksperimen **historis** yang **tidak dapat direproduksi** dengan pipeline saat ini.

## Hasil yang Dipindahkan

| File | Hasil Klaim | Masalah |
|------|-------------|---------|
| `ensemble_advanced_results.json` | 94.09% val, 86.86% test | Self-ensemble dengan 1 model 3x |
| `improved_model_evaluation.json` | 86.88% F1-Macro | Model training tidak terdokumentasi |
| `improved_model_threshold_tuning.json` | - | Bergantung pada improved_model |
| `final_90_percent_results.json` | - | Eksperimen 90% tidak reproducible |
| `ultimate_90_percent_results.json` | - | Eksperimen 90% tidak reproducible |
| `advanced_ensemble_90_percent_results.json` | - | Eksperimen 90% tidak reproducible |

## Catatan Penting

Hasil-hasil ini mungkin valid pada saat eksperimen dilakukan, tetapi:
1. Tidak ada dokumentasi training yang memadai
2. Tidak dapat direproduksi dengan script yang ada
3. Mungkin menggunakan dataset atau konfigurasi berbeda

## Untuk Referensi Saja

Gunakan folder `/results/reproducible/` untuk hasil yang dapat dipercaya.
