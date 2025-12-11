# Javanese Hate Speech Detection: AI-Augmented Ensemble

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Status](https://img.shields.io/badge/Status-Experimental-orange)

This repository contains the official implementation of experiments focusing on **Human-and-Model-in-the-Loop Ensemble Learning** for Javanese Hate Speech Detection. It features a novel **Dual-Engine AI Augmentation** strategy using DeepSeek and Gemini to create a massive sociolinguistically diverse dataset.

## 🌟 Key Features
*   **Dual-Engine Data Generation:** Uses DeepSeek (for Ngoko/Slang) and Gemini (for Code-Switching/Krama) to generate synthetic training data.
*   **Domain-Adaptive Pre-Training (DAPT):** Creates a "Custom Javanese BERT" by training on Wikipedia + Hate Speech Dataset + AI Synthetic Data.
*   **Advanced Ensembling:** Implements Multi-Architecture and Multi-Granularity Stacking ensembles.

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
pip install zai-sdk google-genai openai
```

### 2. Generate Synthetic Data (Optional)
If you have API keys for DeepSeek and Gemini:
```bash
export DEEPSEEK_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
python massive_data_generator.py
```

### 3. Download Wikipedia & Create Corpus
```bash
python download_wiki_javanese.py
```

### 4. Train Custom Model (The "Super Model")
This will combine all data sources and fine-tune a RoBERTa model.
```bash
python train_custom_bert_v2.py
```

### 5. Run Experiments
Run the ensemble experiments to verify performance.
```bash
# Experiment 1: Multi-Architecture (High Diversity)
python final_meta_ensemble_90_percent.py

# Experiment 2: Multi-Granularity with Custom Model
python super_meta_ensemble_v2.py
```

## 📊 Results

| Model | F1-Macro | Accuracy |
|-------|----------|----------|
| Baseline (IndoRoberta) | 56.32% | 62.39% |
| **Meta-Ensemble (Our Best)** | **72.29%** | **73.93%** |
| Custom Javanese BERT v2 | 62.55% | 65.44% |

See `Technical_Report_Javanese_AI_Augmentation.md` for detailed analysis.

## 📂 Project Structure
*   `data/`: Datasets and corpora.
*   `models/`: Saved model checkpoints.
*   `results/`: JSON logs of experiment results.
*   `massive_data_generator.py`: The AI agent script.
*   `train_custom_bert_v2.py`: The DAPT training script.

## 📜 License
This project is licensed under the MIT License.