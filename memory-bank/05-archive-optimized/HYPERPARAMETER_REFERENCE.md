# Hyperparameter Reference Guide

## Experiment 0: Baseline IndoBERT Balanced

### Model Architecture
- **Base Model**: `indobenchmark/indobert-base-p1`
- **Model Type**: BERT-based transformer
- **Number of Labels**: 4 (multi-class classification)
- **Max Sequence Length**: 128 tokens

### Training Hyperparameters

#### Core Training Parameters
```yaml
learning_rate: 2e-5          # AdamW learning rate
num_epochs: 10               # Maximum training epochs
batch_size: 16               # Per-device batch size
weight_decay: 0.01           # L2 regularization
warmup_steps: 100            # Linear warmup steps
```

#### Evaluation & Logging
```yaml
eval_strategy: "steps"        # Evaluation strategy
eval_steps: 200              # Evaluate every N steps
logging_steps: 50            # Log every N steps
save_steps: 400              # Save checkpoint every N steps (disabled)
metric_for_best_model: "eval_f1_macro"  # Primary metric
```

#### Early Stopping
```yaml
early_stopping_patience: 3   # Stop if no improvement for N evaluations
load_best_model_at_end: false # Disabled due to JSON serialization issues
```

#### Data Splitting
```yaml
test_size: 0.2               # 20% for test set
validation_size: 0.2         # 20% of remaining for validation
random_state: 42             # Reproducibility seed
stratify: true               # Maintain class distribution
```

#### Class Balancing
```yaml
use_class_weights: true      # Enable balanced class weights
class_weight_method: "balanced"  # Sklearn balanced method
```

### Computed Class Weights
```python
# Automatically computed based on class frequencies
class_weights = {
    0: 0.5167,  # Bukan Ujaran Kebencian (majority)
    1: 1.6728,  # Ujaran Kebencian - Ringan (minority)
    2: 1.2138,  # Ujaran Kebencian - Sedang
    3: 1.5555   # Ujaran Kebencian - Berat
}
```

### Hardware & Environment
```yaml
dataloader_num_workers: 0    # Single-threaded data loading
remove_unused_columns: false # Keep all dataset columns
push_to_hub: false          # Don't upload to HuggingFace Hub
report_to: null             # Disable wandb/tensorboard
seed: 42                    # Global random seed
```

### Performance Results
```yaml
training_time: 458.55       # seconds (~7.6 minutes)
test_accuracy: 0.6651       # 66.51%
test_f1_macro: 0.6236       # 62.36%
test_precision_macro: 0.6192 # 61.92%
test_recall_macro: 0.6288   # 62.88%
```

## Hyperparameter Tuning Guidelines

### Learning Rate
- **Current**: 2e-5 (standard for BERT fine-tuning)
- **Range to try**: [1e-5, 2e-5, 3e-5, 5e-5]
- **Notes**: Higher LR may cause instability, lower LR may be too slow

### Batch Size
- **Current**: 16 (limited by memory)
- **Alternatives**: 8, 32 (if memory allows)
- **Notes**: Larger batch size may improve stability but requires more memory

### Max Length
- **Current**: 128 tokens
- **Considerations**: 
  - Increase to 256 or 512 for longer texts
  - Check actual text length distribution first
  - Longer sequences = more memory usage

### Epochs
- **Current**: 10 epochs
- **Observations**: Training completed without early stopping
- **Recommendations**: 
  - Try 5-15 epochs
  - Monitor validation metrics closely
  - Use early stopping effectively

### Weight Decay
- **Current**: 0.01
- **Range**: [0.001, 0.01, 0.1]
- **Purpose**: Prevents overfitting

### Warmup Steps
- **Current**: 100 steps
- **Formula**: Usually 10% of total training steps
- **Total steps**: ~1,670 (26,724 samples / 16 batch size * 1 epoch)
- **Recommended**: 167 steps (10% of 1,670)

## Advanced Hyperparameter Strategies

### Learning Rate Scheduling
```python
# Current: Linear warmup + linear decay (default)
# Alternatives to try:
scheduler_type: "cosine"           # Cosine annealing
scheduler_type: "polynomial"       # Polynomial decay
scheduler_type: "constant_with_warmup"  # Constant after warmup
```

### Gradient Clipping
```python
max_grad_norm: 1.0  # Default gradient clipping
# Try: 0.5, 1.0, 2.0
```

### Optimizer Settings
```python
adam_beta1: 0.9     # Default
adam_beta2: 0.999   # Default
adam_epsilon: 1e-8  # Default
# Consider trying different beta values for different convergence behavior
```

## Class Imbalance Strategies

### Current: Class Weighting
- **Method**: Sklearn's "balanced" approach
- **Formula**: n_samples / (n_classes * np.bincount(y))
- **Pros**: Simple, effective for moderate imbalance
- **Cons**: May not work well for extreme imbalance

### Alternative Strategies
1. **Focal Loss**: Focus on hard examples
2. **SMOTE**: Synthetic minority oversampling
3. **Cost-sensitive learning**: Custom loss functions
4. **Ensemble methods**: Combine multiple models

## Hyperparameter Search Strategies

### Grid Search (Systematic)
```python
param_grid = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [8, 16, 32],
    'weight_decay': [0.001, 0.01, 0.1],
    'warmup_ratio': [0.06, 0.1, 0.2]
}
```

### Random Search (Efficient)
```python
# Sample randomly from distributions
learning_rate: uniform(1e-5, 5e-5)
weight_decay: loguniform(1e-4, 1e-1)
warmup_ratio: uniform(0.05, 0.2)
```

### Bayesian Optimization (Advanced)
- Use tools like Optuna or Hyperopt
- More efficient than grid/random search
- Good for expensive evaluations

## Monitoring & Validation

### Key Metrics to Track
1. **Training Loss**: Should decrease steadily
2. **Validation F1-Macro**: Primary metric for class imbalance
3. **Per-class F1**: Monitor minority class performance
4. **Learning Rate**: Track scheduler behavior
5. **Gradient Norm**: Check for gradient explosion/vanishing

### Early Stopping Configuration
```python
early_stopping_patience: 3      # Current setting
metric_for_best_model: "eval_f1_macro"  # Focus on balanced performance
greater_is_better: true         # Higher F1 is better
```

## Best Practices

1. **Start Simple**: Use proven hyperparameters first
2. **One Change at a Time**: Isolate the effect of each parameter
3. **Monitor Closely**: Watch for overfitting signs
4. **Document Everything**: Keep detailed logs of all experiments
5. **Reproducibility**: Always set random seeds
6. **Validation Strategy**: Use proper train/val/test splits
7. **Resource Management**: Consider computational costs

## Next Experiment Suggestions

### Experiment 1: Learning Rate Tuning
- Try: [1e-5, 3e-5, 5e-5]
- Keep other parameters constant
- Expected impact: Convergence speed and final performance

### Experiment 2: Batch Size Impact
- Try: [8, 32] (if memory allows)
- Adjust learning rate accordingly (larger batch → higher LR)
- Expected impact: Training stability and generalization

### Experiment 3: Sequence Length
- Analyze actual text lengths in dataset
- Try: 256 tokens if many texts are truncated
- Expected impact: Better context understanding

### Experiment 4: Advanced Balancing
- Implement focal loss
- Try SMOTE oversampling
- Expected impact: Better minority class performance

## Computational Considerations

### Memory Usage
- **Current**: ~4-6GB GPU memory (estimated)
- **Factors**: Batch size, sequence length, model size
- **Optimization**: Gradient checkpointing, mixed precision

### Training Time
- **Current**: ~7.6 minutes for 10 epochs
- **Scaling**: Linear with epochs, batch size affects steps
- **Optimization**: Larger batch size (fewer steps), better hardware

### Cost-Performance Trade-offs
- **Quick experiments**: Smaller models, fewer epochs
- **Final models**: Full hyperparameter search, longer training
- **Resource allocation**: Balance exploration vs exploitation