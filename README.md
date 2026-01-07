# Javanese Hate Speech Detection: Label-Smoothing Optimized BERT Approach

This repository contains the source code, dataset, and LaTeX source for the paper **"Label-Smoothing Optimized BERT Approach for Javanese Hate Speech Detection: A Comprehensive Analysis with Hard Negative Mining"**.

## 📂 Repository Structure

- **`paper/`**: Contains the LaTeX source code and the compiled PDF of the final paper.
- **`reproduce/`**: Scripts to reproduce the experiments and figures reported in the paper.
    - `experiment_6_focal_loss.py`: The training script for the best performing model (IndoBERT + Label Smoothing).
    - `generate_figures.py`: Script to generate all figures used in the paper.
- **`data/`**: Dataset files used for training and evaluation.
- **`models/`**: Directory for saving/loading model checkpoints.
- **`archive/`**: Contains legacy experiments, logs, and historical artifacts.

## 🚀 Reproducing Results

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training the Best Model

To train the IndoBERT model with Label Smoothing (epsilon=0.1) as reported in the paper:

```bash
python reproduce/experiment_6_focal_loss.py
```

### Generating Figures

To generate the confusion matrix, architecture diagram, and other plots:

```bash
python reproduce/generate_figures.py
```

## 📄 Citation

If you use this code or dataset, please cite our paper:

```bibtex
@article{amien2026javanese,
  title={Label-Smoothing Optimized BERT Approach for Javanese Hate Speech Detection},
  author={Amien, Mukhlis and Sijabat, Daniel Rudiaman and Kanthi, Yekti Asmoro},
  journal={National Conference on Computer Science},
  year={2026}
}
```