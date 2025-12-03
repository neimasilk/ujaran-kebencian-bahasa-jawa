# Deteksi Ujaran Kebencian Bahasa Jawa

**Sistem Deteksi Ujaran Kebencian berbasis Transformer untuk Bahasa Jawa**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.20%2B-yellow)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Overview

Proyek ini mengembangkan sistem deteksi ujaran kebencian untuk bahasa Jawa menggunakan model Transformer (IndoBERT, XLM-RoBERTa, mBERT). Sistem ini dapat mengklasifikasikan teks bahasa Jawa ke dalam kategori: **Normal**, **Abusive**, dan **Hate Speech**.

### 🏆 Pencapaian Utama
- **Model Terbaik:** IndoBERT Base dengan F1-Score Macro **65.80%** (Hyperparameter Optimized)
- **Hyperparameter Tuning:** 72 eksperimen lengkap dengan konfigurasi optimal
- **Dataset:** 41,887 samples berlabel berkualitas tinggi
- **Infrastruktur:** Production-ready dengan GPU acceleration
- **API:** FastAPI server untuk deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation
```bash
# Clone repository
git clone <repository-url>
cd ujaran-kebencian-bahasa-jawa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.modelling.train_model import HateSpeechClassifier

# Load trained model
classifier = HateSpeechClassifier()
model = classifier.load_model('models/best_model')

# Predict
text = "Contoh teks bahasa Jawa"
prediction = classifier.predict(text)
print(f"Prediction: {prediction}")
```

---

## 📊 Model Performance

| Model | F1-Score Macro | Accuracy | Configuration | Status |
|-------|----------------|----------|---------------|--------|
| **IndoBERT Base (Optimized)** | **65.80%** | **65.79%** | LR=5e-05, BS=32, EP=3 | ✅ **Best** |
| IndoBERT Base (Previous) | 80.36% | 73.8% | Default config | ✅ Complete |
| IndoBERT Large | ~75% | ~70% | Default config | ✅ Complete |
| mBERT | ~65% | ~60% | Default config | ✅ Complete |
| XLM-RoBERTa Improved | 61.86% | ~58% | Default config | ✅ Complete |

### 🔬 Hyperparameter Tuning Results
- **Total Experiments:** 72 kombinasi hyperparameter
- **Optimal Configuration:** Learning Rate 5e-05, Batch Size 32, Epochs 3, Warmup Ratio 0.05
- **Training Efficiency:** 133.56 detik (~2.2 menit) dengan mixed precision
- **GPU Acceleration:** NVIDIA GeForce RTX 4080 dengan FP16

---

## 📁 Project Structure

```
📦 ujaran-kebencian-bahasa-jawa/
├── 📁 src/                    # Source code
│   ├── 📁 api/               # FastAPI server
│   ├── 📁 modelling/         # Model training & evaluation
│   ├── 📁 data_collection/   # Data labeling pipeline
│   └── 📁 preprocessing/     # Text preprocessing
├── 📁 data/                  # Datasets
│   ├── 📁 raw/              # Raw unlabeled data
│   ├── 📁 processed/        # Processed datasets
│   └── 📁 standardized/     # Final standardized dataset
├── 📁 experiments/           # Experiment scripts & results
├── 📁 models/               # Trained models
├── 📁 docs/                 # Academic paper documentation
├── 📁 memory-bank/          # Project documentation
│   ├── 📁 01-project-core/  # Core project docs
│   ├── 📁 02-research-active/ # Research & experiments
│   ├── 📁 03-technical-guides/ # Technical documentation
│   └── 📁 04-archive-ready/ # Completed documentation
└── 📁 tests/               # Unit & integration tests
```

---

## 📚 Documentation

### 🎯 For Different Roles

**🔬 Researchers & Data Scientists:**
- [Hyperparameter Tuning Results](HYPERPARAMETER_TUNING_RESULTS.md)
- [Experiment Results](memory-bank/02-research-active/consolidated-experiments/)
- [Model Comparison Report](memory-bank/02-research-active/IMPROVED_MODEL_COMPARISON_REPORT.md)
- [Academic Paper Docs](docs/academic-paper/)

**👨‍💻 Developers:**
- [Technical Implementation Guide](memory-bank/03-technical-guides/)
- [API Documentation](src/api/README.md)
- [GPU Setup Guide](memory-bank/03-technical-guides/GPU_SETUP_DOCUMENTATION.md)
- [Architecture Overview](memory-bank/03-technical-guides/architecture.md)

**📋 Project Managers:**
- [Project Status](memory-bank/01-project-core/papan-proyek.md)
- [Progress Timeline](memory-bank/01-project-core/progress.md)
- [Project Summary](memory-bank/01-project-core/project-summary.md)

### 🧭 Quick Navigation
- **📖 Documentation Summary:** [DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md) 🆕
- **🎯 Project Core:** [memory-bank/01-project-core/](memory-bank/01-project-core/)
- **🔬 Active Research:** [memory-bank/02-research-active/](memory-bank/02-research-active/)
- **🛠️ Technical Guides:** [memory-bank/03-technical-guides/](memory-bank/03-technical-guides/)

---

## 🔧 Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest src/tests/test_training.py
pytest src/tests/test_api_unit.py
```

### Starting API Server
```bash
# Development server
cd src/api
python run_server.py

# Production server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Training New Model
```bash
# Train IndoBERT baseline
python experiments/experiment_0_baseline_indobert.py

# Train with custom configuration
python src/modelling/train_model.py --config custom_config.json
```

---

## 📈 Current Status

**🎯 Phase:** Production Deployment Preparation (Phase 4)  
**📊 Progress:** 90% Complete  
**🏆 Best Model:** IndoBERT Base Optimized (F1-Score: 65.80%)  
**📅 Last Updated:** August 6, 2025  

### ✅ Completed
- [x] Data Collection & Labeling (41,887 samples)
- [x] Model Training & Evaluation (7/9 experiments successful)
- [x] **Comprehensive Hyperparameter Tuning (72 experiments)**
- [x] **Optimal Configuration Discovery (LR=5e-05, BS=32, EP=3, WR=0.05)**
- [x] Performance Optimization (GPU acceleration, mixed precision)
- [x] Class Imbalance Solutions (stratified sampling, focal loss)
- [x] Documentation & Academic Paper Preparation
- [x] **Resume-capable Training Pipeline**

### 🚧 In Progress
- [ ] Final Model Training dengan Optimal Configuration
- [ ] API Deployment & Production Setup
- [ ] Performance Monitoring & Logging
- [ ] Academic Paper Writing

### 🎯 Next Steps
- Train production model dengan konfigurasi optimal
- Production deployment dengan monitoring
- Model serving optimization
- Academic publication preparation

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Development Guidelines
- Follow [VIBE Coding Guide v1.4](vibe-guide/) for project structure
- Write tests for new features
- Update documentation for significant changes
- Use meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Project Team:** AI Research Lab  
**Email:** [contact@example.com](mailto:contact@example.com)  
**Documentation:** [memory-bank/QUICK_NAVIGATION.md](memory-bank/QUICK_NAVIGATION.md)  

---

## 🙏 Acknowledgments

- **Hugging Face** for Transformers library
- **IndoBERT** team for Indonesian language model
- **PyTorch** community for deep learning framework
- **FastAPI** for modern web framework

---

*Proyek ini mengikuti standar [VIBE Coding Guide v1.4](vibe-guide/) untuk pengembangan kolaboratif yang efisien.*