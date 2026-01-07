"""
EXPERIMENT 7: Simple Ensemble - Voting dari Best Models
======================================================

Metode: Majority voting dari predictions yang ada
Target: F1-Macro > 82%
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

# Suppress warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model_from_checkpoint(checkpoint_path):
    """Load model dari checkpoint folder yang benar"""
    # Cari checkpoint folder
    if os.path.isdir(checkpoint_path):
        # Cari subfolder checkpoint-*
        subdirs = [d for d in os.listdir(checkpoint_path) if d.startswith('checkpoint-')]
        if subdirs:
            # Ambil checkpoint terakhir (nomor terbesar)
            subdirs.sort(key=lambda x: int(x.split('-')[1]))
            checkpoint_path = os.path.join(checkpoint_path, subdirs[-1])

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        return model, tokenizer, checkpoint_path
    except Exception as e:
        return None, None, None


def get_predictions(model, tokenizer, texts, device, batch_size=32):
    """Get predictions from model"""
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def main():
    print("="*50)
    print("EXPERIMENT 7: SIMPLE ENSEMBLE")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1/3] Loading dataset...")
    phase3 = pd.read_csv("data/improved/phase3_relabeled.csv")
    phase4 = pd.read_csv("data/improved/phase4_generated.csv")

    phase3 = phase3[['text', 'new_label']].rename(columns={'new_label': 'label'})
    phase4 = phase4[['text', 'label']]

    df = pd.concat([phase3, phase4], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    temp_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    print(f"      Test samples: {len(test_df)}")

    # Model configurations
    model_configs = [
        ("IndoBERT (Exp 5)", "models/experiment_5_improved", "indobenchmark/indobert-base-p1"),
        ("Label Smoothing (Exp 6C)", "models/experiment_6c_hyperparam_tuning/lr_2e-05_bs_16_wd_0.01_wr_0.0_ep_5_eps_0.1", "indobenchmark/indobert-base-p1"),
        ("Focal Loss (Exp 6A)", "models/experiment_6a_focal_loss", "indobenchmark/indobert-base-p1"),
    ]

    # Collect predictions
    print("\n[2/3] Collecting predictions...")
    all_preds = []
    all_probs = []
    model_names = []

    for name, checkpoint, base_model in model_configs:
        print(f"      Loading {name}...")
        model, tokenizer, actual_path = load_model_from_checkpoint(checkpoint)

        if model is None:
            # Try base model
            try:
                print(f"        Checkpoint not found, using base model {base_model}")
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=4)
                actual_path = base_model
            except:
                print(f"        Skipping {name}")
                continue

        model.to(device)
        texts = test_df['text'].tolist()
        labels = test_df['label'].values

        preds, probs = get_predictions(model, tokenizer, texts, device)

        f1 = f1_score(labels, preds, average='macro')
        print(f"        F1-Macro: {f1:.4f}")

        all_preds.append(preds)
        all_probs.append(probs)
        model_names.append(name)

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

    if len(model_names) == 0:
        print("ERROR: No models loaded!")
        return

    print(f"\n      Loaded {len(model_names)} models")

    # Ensemble methods
    print("\n[3/3] Testing ensemble methods...")
    labels = test_df['label'].values

    results = {}

    # 1. Hard Voting (Majority)
    from scipy.stats import mode
    stacked_preds = np.stack(all_preds, axis=0)
    vote_result = mode(stacked_preds, axis=0)
    vote_preds = vote_result.mode.flatten()
    vote_f1 = f1_score(labels, vote_preds, average='macro')
    results['hard_voting'] = {'f1': vote_f1, 'preds': vote_preds}
    print(f"      Hard Voting: {vote_f1:.4f}")

    # 2. Soft Voting (Average Probabilities)
    avg_probs = np.mean(all_probs, axis=0)
    soft_preds = np.argmax(avg_probs, axis=1)
    soft_f1 = f1_score(labels, soft_preds, average='macro')
    results['soft_voting'] = {'f1': soft_f1, 'preds': soft_preds}
    print(f"      Soft Voting: {soft_f1:.4f}")

    # 3. Weighted Voting (by individual F1)
    weights = np.array([f1_score(labels, p, average='macro') for p in all_preds])
    weights = weights / weights.sum()
    weighted_probs = np.average(all_probs, axis=0, weights=weights)
    weighted_preds = np.argmax(weighted_probs, axis=1)
    weighted_f1 = f1_score(labels, weighted_preds, average='macro')
    results['weighted_voting'] = {'f1': weighted_f1, 'preds': weighted_preds, 'weights': weights}
    print(f"      Weighted Voting: {weighted_f1:.4f}")

    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_f1 = results[best_name]['f1']
    best_preds = results[best_name]['preds']

    # Detailed report for best
    report = classification_report(
        labels, best_preds,
        target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
        output_dict=True
    )

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"\nModels used: {model_names}")
    print(f"\nBest Method: {best_name}")
    print(f"F1-Macro: {best_f1:.2%}")
    print(f"Accuracy: {accuracy_score(labels, best_preds):.2%}")
    print("\nPer-Class F1:")
    for cls in ['Neutral', 'Light', 'Moderate', 'Severe']:
        print(f"  {cls}: {report[cls]['f1-score']:.2%}")

    if best_name == 'weighted_voting':
        print("\nWeights:")
        for i, name in enumerate(model_names):
            print(f"  {name}: {results['weighted_voting']['weights'][i]:.3f}")

    # Save results
    os.makedirs("results/experiment_7", exist_ok=True)
    with open("results/experiment_7/results.json", 'w') as f:
        json.dump({
            'models_used': model_names,
            'best_method': best_name,
            'best_f1_macro': float(best_f1),
            'best_accuracy': float(accuracy_score(labels, best_preds)),
            'per_class_f1': {cls: report[cls]['f1-score'] for cls in ['Neutral', 'Light', 'Moderate', 'Severe']},
            'all_methods': {k: {'f1_macro': float(v['f1'])} for k, v in results.items()}
        }, f, indent=2)

    print("\n[OK] Experiment 7 complete!")
    print(f"Results saved to: results/experiment_7/results.json")

    return results


if __name__ == "__main__":
    main()
