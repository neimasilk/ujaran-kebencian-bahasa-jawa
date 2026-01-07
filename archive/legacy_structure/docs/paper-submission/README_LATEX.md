# LaTeX Paper for Javanese Hate Speech Detection

## Files

- **paper.tex** - Main LaTeX source file (529 lines)
- **figures/** - Directory containing 8 PNG figures

## Figures Included

1. `architecture_diagram.png` - Model architecture visualization
2. `confusion_matrix.png` - Normalized confusion matrix
3. `dataset_comparison.png` - Dataset variant comparison
4. `ensemble_overfitting.png` - Ensemble overfitting analysis
5. `hard_negative_analysis.png` - Hard negative analysis by class
6. `loss_function_comparison.png` - Loss function ablation study
7. `model_comparison.png` - Baseline model comparison
8. `per_class_f1.png` - Per-class F1-score performance

## Compilation Options

### Option 1: Overleaf (Recommended - No Installation)

1. Go to https://overleaf.com
2. Create a new project
3. Upload `paper.tex`
4. Create a folder named `figures` and upload all 8 PNG files
5. Click "Recompile"

### Option 2: Local MiKTeX Installation

If you have MiKTeX installed:

```bash
cd docs/paper-submission
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

### Option 3: TeX Live

```bash
cd docs/paper-submission
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Option 4: VS Code with LaTeX Workshop

1. Install LaTeX Workshop extension
2. Open `paper.tex`
3. Click "Build LaTeX project"

## Paper Structure

- **Title**: Label-Smoothing Optimized BERT Approach for Javanese Hate Speech Detection: A Comprehensive Analysis with Hard Negative Mining
- **Authors**: Mukhlis Amien, Daniel Rudiaman Sijabat, Yekti Asmoro Kanthi
- **Sections**:
  1. Introduction
  2. Materials and Methods
  3. Results
  4. Hard Negative Analysis
  5. Discussion
  6. Conclusion
  7. References

## Key Results Reported

- **Best Model**: IndoBERT + Label Smoothing (epsilon=0.1)
- **F1-Macro**: 81.38%
- **Validation-Test Gap**: 0.25% (genuine generalization)
- **Hard Negatives**: 59 samples (5.9% of test set)

## Tables Included

1. Dataset Statistics
2. Label Distribution
3. Optimal Hyperparameters
4. Baseline Model Comparison
5. Loss Function Ablation Study
6. Dataset Variant Comparison
7. Ensemble Overfitting Analysis
8. Hard Negative Statistics
9. Custom BERT v3 Comparison
10. Ensemble Method Failure Analysis

## Notes

- Uses IEEEtran document class
- All figures are PNG format at 300 DPI
- Mathematical formulas use LaTeX math mode
- References are manually formatted (no .bib file needed)
