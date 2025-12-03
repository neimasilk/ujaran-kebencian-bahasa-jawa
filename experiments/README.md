# Enhanced Experiments for Javanese Hate Speech Detection

This directory contains enhanced experiments designed to improve the performance of the Javanese hate speech detection system beyond the current baseline.

## Overview

The current best model achieves an F1-macro score of 51.67%, significantly below the target of 80.36%. These enhanced experiments implement advanced techniques to bridge this performance gap.

## Experiments

### 1. Enhanced IndoBERT Experiment (`enhanced_indobert_experiment.py`)

Implements several enhancements to the baseline IndoBERT model:

- **Label Smoothing**: Improves generalization by softening the target distribution
- **Data Augmentation**: Simple synonym replacement to increase dataset diversity
- **Progressive Training**: More epochs with careful hyperparameter tuning
- **Mixed Precision Training**: Uses FP16 for faster training on compatible hardware
- **Enhanced Evaluation**: Detailed metrics and confusion matrix analysis

### 2. Hyperparameter Tuning (`hyperparameter_tuning.py`)

Systematically searches for optimal hyperparameters:

- **Grid Search**: Tests combinations of learning rates, batch sizes, epochs, and warmup ratios
- **Comprehensive Evaluation**: Tracks F1-macro, accuracy, precision, and recall for each configuration
- **Best Configuration Identification**: Automatically identifies the best hyperparameter combination

### 3. Ensemble Method (`ensemble_method.py`)

Combines predictions from multiple models for improved performance:

- **Weighted Average**: Combines predictions using configurable weights
- **Model Comparison**: Shows performance improvement over individual models
- **Flexible Configuration**: Easy to add/remove models and adjust weights

### 4. Data Augmentation (`data_augmentation.py`)

Increases dataset diversity to improve model generalization:

- **Multiple Techniques**: Synonym replacement, random insertion, swapping, and deletion
- **Javanese-Specific**: Uses a synonym dictionary tailored to Javanese/Indonesian language
- **Controlled Augmentation**: Configurable augmentation rate and sample count

## Usage

1. **Enhanced IndoBERT Experiment**:
   ```bash
   python experiments/enhanced_indobert_experiment.py
   ```

2. **Hyperparameter Tuning**:
   ```bash
   python experiments/hyperparameter_tuning.py
   ```

3. **Ensemble Method**:
   ```bash
   python experiments/ensemble_method.py
   ```

4. **Data Augmentation**:
   ```bash
   python experiments/data_augmentation.py
   ```

## Expected Improvements

These experiments are expected to provide the following improvements:

1. **Enhanced IndoBERT**: 5-10% improvement in F1-macro score
2. **Hyperparameter Tuning**: 3-7% improvement through optimal configuration
3. **Ensemble Method**: 5-15% improvement by combining model strengths
4. **Data Augmentation**: 3-8% improvement through increased data diversity

## Configuration

Each experiment can be configured by modifying the respective configuration classes:

- `EnhancedConfig` in `enhanced_indobert_experiment.py`
- `HyperparameterConfig` in `hyperparameter_tuning.py`
- `EnsembleConfig` in `ensemble_method.py`
- `DataAugmentationConfig` in `data_augmentation.py`

## Results

Results are automatically saved in experiment-specific directories with detailed metrics and logs.

## Next Steps

1. Run the hyperparameter tuning experiment to find optimal configurations
2. Use the best hyperparameters in the enhanced IndoBERT experiment
3. Apply data augmentation to increase training data diversity
4. Combine the best individual models using the ensemble method
5. Iterate and refine based on results