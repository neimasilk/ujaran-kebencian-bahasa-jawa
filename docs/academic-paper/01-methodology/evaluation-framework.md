# Evaluation Framework - Robust Assessment for Imbalanced Classification

## 1. Overview Framework Evaluasi

### 1.1 Tujuan Evaluasi
- **Primary Goal:** Mengukur performa model pada deteksi ujaran kebencian bahasa Jawa
- **Secondary Goal:** Memastikan evaluasi yang fair dan tidak bias oleh class imbalance
- **Tertiary Goal:** Memberikan insights untuk improvement model

### 1.2 Challenges dalam Evaluasi
- **Class Imbalance:** 85% non-hate speech vs 15% hate speech
- **Multi-class Classification:** 4 kategori dengan distribusi yang sangat tidak seimbang
- **Language Specificity:** Evaluasi untuk bahasa dengan resource terbatas
- **Cultural Context:** Mempertimbangkan nuansa budaya dalam penilaian

## 2. Evaluation Methodology

### 2.1 Dataset Splits Strategy

#### 2.1.1 Original Split (Biased)
```python
# Split yang mengikuti distribusi original
train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Mempertahankan proporsi kelas
)
```

**Karakteristik:**
- Training: 80% (33,509 samples)
- Testing: 20% (8,378 samples)
- Distribusi: Mengikuti class imbalance original
- **Problem:** Bias terhadap majority class dalam evaluasi

#### 2.1.2 Balanced Evaluation Set
```python
# Balanced sampling untuk evaluasi yang fair
balanced_samples = []
for label in [0, 1, 2, 3]:
    class_samples = df[df['label'] == label].sample(n=200, random_state=42)
    balanced_samples.append(class_samples)

balanced_eval_set = pd.concat(balanced_samples)
```

**Karakteristik:**
- Total: 800 samples (200 per class)
- Distribusi: 25% per class (perfectly balanced)
- **Advantage:** Evaluasi yang tidak bias oleh majority class
- **Usage:** Primary evaluation untuk model comparison

### 2.2 Metrics Selection

#### 2.2.1 Primary Metrics

**F1-Score Macro**
```python
f1_macro = f1_score(y_true, y_pred, average='macro')
```
- **Rationale:** Memberikan bobot yang sama untuk semua kelas
- **Advantage:** Tidak bias terhadap majority class
- **Range:** 0.0 - 1.0 (higher is better)
- **Interpretation:** Average F1-score across all classes

**Per-Class F1-Score**
```python
f1_per_class = f1_score(y_true, y_pred, average=None)
```
- **Purpose:** Detailed analysis per kategori ujaran kebencian
- **Insight:** Identifikasi kelas yang sulit diprediksi

#### 2.2.2 Secondary Metrics

**Accuracy**
```python
accuracy = accuracy_score(y_true, y_pred)
```
- **Usage:** Overall correctness measure
- **Limitation:** Dapat misleading pada imbalanced data
- **Context:** Dilaporkan untuk completeness

**Precision dan Recall per Class**
```python
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)
```
- **Precision:** Proportion of positive predictions yang benar
- **Recall:** Proportion of actual positives yang terdeteksi
- **Support:** Jumlah true instances per class

#### 2.2.3 Advanced Metrics

**Weighted F1-Score**
```python
f1_weighted = f1_score(y_true, y_pred, average='weighted')
```
- **Purpose:** F1-score yang mempertimbangkan class distribution
- **Usage:** Comparison dengan baseline methods

**Cohen's Kappa**
```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred)
```
- **Purpose:** Agreement measure yang mempertimbangkan chance agreement
- **Range:** -1 to 1 (higher is better)
- **Interpretation:** > 0.8 (excellent), 0.6-0.8 (good), 0.4-0.6 (moderate)

### 2.3 Confusion Matrix Analysis

#### 2.3.1 Standard Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

#### 2.3.2 Normalized Confusion Matrix
```python
# Normalisasi per true class (recall perspective)
cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

# Normalisasi per predicted class (precision perspective)
cm_pred_normalized = confusion_matrix(y_true, y_pred, normalize='pred')
```

#### 2.3.3 Error Analysis dari Confusion Matrix
- **True Positives (TP):** Correctly identified hate speech
- **False Positives (FP):** Non-hate speech classified as hate speech
- **False Negatives (FN):** Hate speech missed by model
- **True Negatives (TN):** Correctly identified non-hate speech

## 3. Evaluation Protocols

### 3.1 Standard Evaluation Protocol

#### 3.1.1 Single Run Evaluation
```python
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return compute_metrics(all_labels, all_predictions)
```

#### 3.1.2 Multiple Runs Protocol
```python
def evaluate_multiple_runs(model_class, train_data, test_data, n_runs=5):
    results = []
    
    for run in range(n_runs):
        # Set different random seed for each run
        set_seed(42 + run)
        
        # Train model
        model = train_model(model_class, train_data)
        
        # Evaluate
        metrics = evaluate_model(model, test_data)
        results.append(metrics)
    
    return aggregate_results(results)
```

### 3.2 Cross-Validation Protocol (Future Work)

#### 3.2.1 Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train and evaluate
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val)
    results.append(metrics)
```

### 3.3 Statistical Significance Testing

#### 3.3.1 Paired t-test untuk Model Comparison
```python
from scipy.stats import ttest_rel

def compare_models(results_model_a, results_model_b, metric='f1_macro'):
    scores_a = [r[metric] for r in results_model_a]
    scores_b = [r[metric] for r in results_model_b]
    
    statistic, p_value = ttest_rel(scores_a, scores_b)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(scores_a) - np.mean(scores_b)
    }
```

#### 3.3.2 Confidence Intervals
```python
from scipy.stats import t

def compute_confidence_interval(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    std_err = np.std(scores, ddof=1) / np.sqrt(n)
    
    # t-distribution critical value
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha/2, df=n-1)
    
    margin_error = t_critical * std_err
    
    return {
        'mean': mean,
        'ci_lower': mean - margin_error,
        'ci_upper': mean + margin_error,
        'std_err': std_err
    }
```

## 4. Bias Detection dan Mitigation

### 4.1 Class Imbalance Bias

#### 4.1.1 Detection
```python
def detect_class_bias(y_true, y_pred):
    # Hitung distribusi prediksi
    pred_dist = np.bincount(y_pred) / len(y_pred)
    true_dist = np.bincount(y_true) / len(y_true)
    
    # Hitung bias score
    bias_scores = abs(pred_dist - true_dist)
    
    return {
        'prediction_distribution': pred_dist,
        'true_distribution': true_dist,
        'bias_scores': bias_scores,
        'total_bias': np.sum(bias_scores)
    }
```

#### 4.1.2 Mitigation Strategies
- **Balanced Evaluation Set:** Primary mitigation
- **Stratified Sampling:** Untuk train/validation splits
- **Class Weighting:** Dalam loss function
- **Threshold Tuning:** Per-class threshold optimization

### 4.2 Evaluation Set Bias

#### 4.2.1 Temporal Bias
- **Problem:** Data dari periode waktu terbatas
- **Detection:** Analisis distribusi temporal
- **Mitigation:** Sampling dari berbagai periode (future work)

#### 4.2.2 Domain Bias
- **Problem:** Dominasi sumber data tertentu
- **Detection:** Analisis distribusi sumber
- **Mitigation:** Diversifikasi sumber data

## 5. Error Analysis Framework

### 5.1 Qualitative Error Analysis

#### 5.1.1 False Positive Analysis
```python
def analyze_false_positives(texts, y_true, y_pred, class_names):
    fp_indices = np.where((y_true == 0) & (y_pred != 0))[0]
    
    fp_analysis = []
    for idx in fp_indices:
        fp_analysis.append({
            'text': texts[idx],
            'true_label': class_names[y_true[idx]],
            'predicted_label': class_names[y_pred[idx]],
            'error_type': 'false_positive'
        })
    
    return fp_analysis
```

#### 5.1.2 False Negative Analysis
```python
def analyze_false_negatives(texts, y_true, y_pred, class_names):
    fn_indices = np.where((y_true != 0) & (y_pred == 0))[0]
    
    fn_analysis = []
    for idx in fn_indices:
        fn_analysis.append({
            'text': texts[idx],
            'true_label': class_names[y_true[idx]],
            'predicted_label': class_names[y_pred[idx]],
            'error_type': 'false_negative'
        })
    
    return fn_analysis
```

### 5.2 Quantitative Error Patterns

#### 5.2.1 Error Distribution Analysis
```python
def analyze_error_patterns(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    # Hitung error rates per class
    error_rates = []
    for i in range(len(cm)):
        total_true = np.sum(cm[i, :])
        correct = cm[i, i]
        error_rate = (total_true - correct) / total_true
        error_rates.append(error_rate)
    
    return {
        'confusion_matrix': cm,
        'error_rates_per_class': error_rates,
        'most_confused_classes': find_most_confused_pairs(cm)
    }
```

#### 5.2.2 Confidence Score Analysis
```python
def analyze_confidence_scores(model, test_loader, device):
    model.eval()
    confidences = []
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            
            max_probs, preds = torch.max(probs, dim=-1)
            
            confidences.extend(max_probs.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    return analyze_confidence_patterns(confidences, predictions, true_labels)
```

## 6. Benchmark Comparison

### 6.1 Baseline Comparisons

#### 6.1.1 Random Baseline
```python
def random_baseline(y_true, class_distribution=None):
    if class_distribution is None:
        # Uniform random
        y_pred = np.random.randint(0, 4, size=len(y_true))
    else:
        # Weighted random based on class distribution
        y_pred = np.random.choice(4, size=len(y_true), p=class_distribution)
    
    return compute_metrics(y_true, y_pred)
```

#### 6.1.2 Majority Class Baseline
```python
def majority_baseline(y_true):
    majority_class = np.bincount(y_true).argmax()
    y_pred = np.full_like(y_true, majority_class)
    
    return compute_metrics(y_true, y_pred)
```

#### 6.1.3 Stratified Baseline
```python
def stratified_baseline(y_true):
    class_probs = np.bincount(y_true) / len(y_true)
    y_pred = np.random.choice(len(class_probs), size=len(y_true), p=class_probs)
    
    return compute_metrics(y_true, y_pred)
```

### 6.2 Model Comparison Framework

#### 6.2.1 Performance Ranking
```python
def rank_models(model_results, primary_metric='f1_macro'):
    rankings = []
    
    for model_name, results in model_results.items():
        mean_score = np.mean([r[primary_metric] for r in results])
        std_score = np.std([r[primary_metric] for r in results])
        
        rankings.append({
            'model': model_name,
            'mean_score': mean_score,
            'std_score': std_score,
            'ci': compute_confidence_interval([r[primary_metric] for r in results])
        })
    
    # Sort by mean score (descending)
    rankings.sort(key=lambda x: x['mean_score'], reverse=True)
    
    return rankings
```

#### 6.2.2 Statistical Significance Matrix
```python
def compute_significance_matrix(model_results, metric='f1_macro'):
    models = list(model_results.keys())
    n_models = len(models)
    
    significance_matrix = np.zeros((n_models, n_models))
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i != j:
                comparison = compare_models(
                    model_results[model_a], 
                    model_results[model_b], 
                    metric
                )
                significance_matrix[i, j] = comparison['p_value']
    
    return significance_matrix, models
```

## 7. Reporting Framework

### 7.1 Standard Report Template

```python
def generate_evaluation_report(model_name, results, error_analysis):
    report = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(results['y_true']),
            'class_distribution': np.bincount(results['y_true']).tolist()
        },
        'performance_metrics': {
            'f1_macro': results['f1_macro'],
            'accuracy': results['accuracy'],
            'per_class_f1': results['per_class_f1'].tolist(),
            'per_class_precision': results['per_class_precision'].tolist(),
            'per_class_recall': results['per_class_recall'].tolist()
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'error_analysis': error_analysis,
        'statistical_info': {
            'confidence_interval': results['confidence_interval'],
            'standard_error': results['standard_error']
        }
    }
    
    return report
```

### 7.2 Visualization Framework

#### 7.2.1 Performance Comparison Plot
```python
def plot_model_comparison(model_results, metric='f1_macro'):
    models = list(model_results.keys())
    means = [np.mean([r[metric] for r in results]) for results in model_results.values()]
    stds = [np.std([r[metric] for r in results]) for results in model_results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, means, yerr=stds, capsize=5)
    plt.ylabel(f'{metric.replace("_", " ").title()}')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

#### 7.2.2 Confusion Matrix Heatmap
```python
def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
```

## 8. Limitations dan Future Work

### 8.1 Current Limitations
- **Single Dataset:** Evaluasi hanya pada satu dataset
- **Limited Cross-validation:** Belum implementasi full cross-validation
- **No Inter-annotator Agreement:** Belum ada measurement untuk label quality
- **Limited Error Analysis:** Analisis error masih basic

### 8.2 Future Improvements
- **Multi-dataset Evaluation:** Evaluasi pada multiple datasets
- **Cross-lingual Evaluation:** Testing pada bahasa daerah lain
- **Human Evaluation:** Comparison dengan human annotators
- **Adversarial Testing:** Robustness testing dengan adversarial examples

---

**Implementation References:**
- Evaluation utilities: `src/evaluation/`
- Metrics computation: `src/utils/metrics.py`
- Visualization tools: `src/utils/visualization.py`

**Experimental Results:**
- [Experiment Summary](../02-experiments/experiment-summary.md)
- [Statistical Analysis](../03-results/statistical-significance.md)

**Metadata:**
- Created: 2025-01-06
- Version: 1.0
- Status: Production Ready
- Contact: Research Team