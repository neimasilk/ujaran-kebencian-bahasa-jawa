import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import json
from tqdm import tqdm

def load_and_preprocess_data(file_path):
    """Load and preprocess the test data"""
    df = pd.read_csv(file_path)
    
    # Remove rows with missing text or labels
    df = df.dropna(subset=['text', 'final_label'])
    
    # Remove empty text
    df = df[df['text'].str.strip() != '']
    
    # Map labels to integers
    label_mapping = {
        'Bukan Ujaran Kebencian': 0,
        'Ujaran Kebencian - Ringan': 1,
        'Ujaran Kebencian - Sedang': 1,
        'Ujaran Kebencian - Berat': 1
    }
    
    df['label'] = df['final_label'].map(label_mapping)
    df = df.dropna(subset=['label'])  # Remove unmapped labels
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df['text'].tolist(), df['label'].tolist()

def evaluate_model(model_path, texts, labels, batch_size=16):
    """Evaluate the model on given texts and labels"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_predictions)
    
    return predictions

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'confusion_matrix': cm.tolist()
    }
    
    return results

def main():
    model_path = 'models/retrained_best'
    test_data_path = 'src/data_collection/test-hasil.csv'
    
    print("Loading test data...")
    texts, labels = load_and_preprocess_data(test_data_path)
    
    print("\nEvaluating model...")
    predictions = evaluate_model(model_path, texts, labels)
    
    print("\nCalculating metrics...")
    results = calculate_metrics(labels, predictions)
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Macro: {results['f1_macro']:.4f}")
    print(f"F1-Weighted: {results['f1_weighted']:.4f}")
    
    print("\n=== PER-CLASS METRICS ===")
    class_names = ['Non-Hate', 'Hate']
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {results['per_class_metrics']['precision'][i]:.4f}")
        print(f"  Recall: {results['per_class_metrics']['recall'][i]:.4f}")
        print(f"  F1: {results['per_class_metrics']['f1'][i]:.4f}")
        print(f"  Support: {results['per_class_metrics']['support'][i]}")
    
    print("\n=== CONFUSION MATRIX ===")
    print("Predicted ->")
    print("Actual |  Non-Hate  Hate")
    print(f"Non-Hate | {results['confusion_matrix'][0][0]:8d} {results['confusion_matrix'][0][1]:5d}")
    print(f"Hate     | {results['confusion_matrix'][1][0]:8d} {results['confusion_matrix'][1][1]:5d}")
    
    # Save results
    output_path = 'models/retrained_best/test_evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()