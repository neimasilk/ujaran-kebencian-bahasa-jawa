# WORKFLOW KOMPUTER BIASA (CPU-ONLY)

<!-- TRIGGER: "ini pakai komputer biasa, lanjutkan dan lakukan untuk komputer biasa" -->

## Status Hari Ini (6 Jan 2026)

| Task | Status | Progress |
|------|--------|----------|
| **DAPT Training** | 🔄 Running | **~58%** (Epoch 5.78/10) - checkpoint tersimpan |
| **LLM Re-labeling** | ⏳ Ready | Menunggu di komputer biasa |

---

Dokumentasi ini untuk task yang bisa dijalankan di **komputer biasa (tanpa GPU)** saat GPU tidak tersedia.

---

## Available Tasks untuk Komputer Biasa

### 1. LLM-as-Judge Re-labeling (Experiment 11) ⭐ RECOMMENDED

**Deskripsi:** Gunakan LLM API (Claude/GPT) untuk re-label samples yang ambigu/uncertain.

**Kelebihan:**
- Tidak butuh GPU (pakai API)
- Bisa jalan di background
- Expected improvement: +0.5-1% F1-Macro
- Cost: ~$5-20 (tergantung jumlah samples)

**File:** `experiments/experiment_11_llm_relabeled.py`

---

## Cara Pakai (Step-by-Step)

### STEP 1: Setup API Key

Pilih salah satu provider:

#### Option A: DeepSeek ⭐ PALING MURAH
```bash
# Windows Command Prompt
set DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-your-deepseek-key-here"

# Linux/Mac
export DEEPSEEK_API_KEY=sk-your-deepseek-key-here
```

#### Option B: Anthropic (Claude)
```bash
# Windows Command Prompt
set ANTHROPIC_API_KEY=sk-ant-your-key-here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

#### Option C: OpenAI (GPT)
```bash
# Windows Command Prompt
set OPENAI_API_KEY=sk-your-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"

# Linux/Mac
export OPENAI_API_KEY=sk-your-key-here
```

---

### STEP 2: Jalankan Re-labeling

**Perintah:**

```bash
# Dengan DeepSeek (default - PALING MURAH)
python experiments/experiment_11_llm_relabeled.py --provider deepseek --max-samples 500

# Atau dengan Claude
python experiments/experiment_11_llm_relabeled.py --provider anthropic --max-samples 500

# Atau dengan GPT-4
python experiments/experiment_11_llm_relabeled.py --provider openai --max-samples 500
```

**Parameter:**
- `--provider`: `deepseek` (default), `anthropic`, atau `openai`
- `--max-samples`: Jumlah maksimal samples untuk di-relabel (default: 500)
- `--threshold`: Threshold confidence untuk uncertain samples (default: 0.6)

**Output:**
- `data/improved/phase5_llm_relabeled.csv` - Dataset baru dengan labels yang diperbaiki
- `results/experiment_11_llm_relabeled/relabeled_details.json` - Detail re-labeling

---

### STEP 3: Cek Hasil

```bash
# Lihat detail re-labeling
cat results/experiment_11_llm_relabeled/relabeled_details.json

# Atau
type results\experiment_11_llm_relabeled\relabeled_details.json
```

**Expected Output:**
```json
{
  "total_samples": 10019,
  "uncertain_samples": 500,
  "relabeled_samples": 350,
  ...
}
```

---

### STEP 4: Push ke GitHub

Setelah selesai, push hasilnya:

```bash
git add data/improved/phase5_llm_relabeled.csv
git add results/experiment_11_llm_relabeled/
git commit -m "feat: add LLM-re-labeled dataset (Phase 5)"
git push origin main
```

---

## Workflow Lengkap

### Di Komputer Biasa (Hari Ini)
1. Setup API key
2. Jalankan `experiment_11_llm_relabeled.py`
3. Tunggu sampai selesai (~30-60 menit untuk 500 samples)
4. Push ke GitHub

### Di Komputer GPU (Besok/Lanjutan)
1. Pull dari GitHub
2. Jalankan training dengan Phase 5 dataset:
   ```bash
   python experiments/experiment_11_train_relabel.py
   ```
3. Evaluate improvement

---

## Estimasi Biaya

| Provider | Model | 500 Samples | Estimasi |
|----------|-------|-------------|----------|
| **DeepSeek** | deepseek-chat | **~$0.10-0.50** | **PALING MURAH** |
| OpenAI | gpt-4o-mini | ~$0.50-1 | Murah |
| Anthropic | claude-3-haiku | ~$5-10 | Paling mahal |

**Rekomendasi:** Pakai **DeepSeek** (paling murah, cukup akurat).

---

## Troubleshooting

### "API key not found"
Pastikan environment variable sudah di-set dengan benar. Cek:
```bash
# Windows
echo %ANTHROPIC_API_KEY%

# Linux/Mac
echo $ANTHROPIC_API_KEY
```

### "Rate limit exceeded"
Tambah delay antara requests:
```bash
python experiments/experiment_11_llm_relabeled.py --max-samples 100
```
Lalu jalankan beberapa kali dengan batch kecil.

### "Out of memory" (tidak mungkin di CPU)
Script ini tidak menggunakan GPU, jadi tidak akan OOM.

---

## Command Siap Pakai

Copy-paste command ini:

```bash
# 1. Set API key (ganti dengan key kamu)
set DEEPSEEK_API_KEY=sk-your-deepseek-key-here

# 2. Jalankan re-labeling
cd D:\documents\ujaran-kebencian-bahasa-jawa
python experiments/experiment_11_llm_relabeled.py --provider deepseek --max-samples 500

# 3. Setelah selesai, push ke GitHub
git add data/improved/phase5_llm_relabeled.csv results/experiment_11_llm_relabeled/
git commit -m "feat: add LLM-re-labeled dataset"
git push origin main
```

---

*Created: 6 Januari 2026*
*Status: Ready to run on CPU*
