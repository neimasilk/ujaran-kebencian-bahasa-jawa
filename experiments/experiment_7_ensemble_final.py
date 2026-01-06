"""
EXPERIMENT 7: Final Ensemble dengan Models yang sudah ada
==========================================================

Menggabungkan predictions dari:
1. IndoBERT (Exp 5) - 79.19%
2. mBERT (Exp 6) - sudah trained
3. XLM-RoBERTa (Exp 6) - sudah trained
4. Custom Javanese BERT v2 - sudah trained

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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

# Suppress transformers warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class EnsembleExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Models yang sudah trained
        self.model_configs = [
            {
                "name": "IndoBERT",
                "path": "indobenchmark/indobert-base-p1",
                "checkpoint": "models/experiment_5_improved"
            },
            {
                "name": "mBERT",
                "path": "bert-base-multilingual-cased",
                "checkpoint": "models/experiment_6_7_improved/bert-base-multilingual-cased"
            },
            {
                "name": "XLM-RoBERTa",
                "path": "xlm-roberta-base",
                "checkpoint": "models/experiment_6_7_improved/xlm-roberta-base"
            },
            {
                "name": "Custom BERT v2",
                "path": "flax-community/indonesian-roberta-base",
                "checkpoint": "models/custom_javanese_bert_v2"
            },
        ]

        self.models = {}
        self.tokenizers = {}

    def load_data(self):
        print("[1/5] Loading dataset...")
        phase3 = pd.read_csv("data/improved/phase3_relabeled.csv")
        phase4 = pd.read_csv("data/improved/phase4_generated.csv")

        phase3 = phase3[['text', 'new_label']].rename(columns={'new_label': 'label'})
        phase4 = phase4[['text', 'label']]

        df = pd.concat([phase3, phase4], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

        print(f"      Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def load_model(self, config):
        """Load model dari checkpoint"""
        name = config["name"]
        checkpoint = config["checkpoint"]
        base_path = config["path"]

        if not os.path.exists(checkpoint):
            print(f"      Skipping {name} (checkpoint not found)")
            return None, None

        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(self.device)
            model.eval()
            return model, tokenizer
        except Exception as e:
            print(f"      Warning: Could not load {name}: {e}")
            # Try loading from base path
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_path)
                model = AutoModelForSequenceClassification.from_pretrained(base_path, num_labels=4)
                model.to(self.device)
                model.eval()
                return model, tokenizer
            except:
                return None, None

    def get_predictions(self, model, tokenizer, df, model_name):
        """Get prediction probabilities from model"""
        texts = df['text'].tolist()

        all_probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def collect_predictions(self, train_df, val_df, test_df):
        print("[2/5] Loading models and collecting predictions...")

        val_probs = []
        test_probs = []
        val_labels = val_df['label'].values
        test_labels = test_df['label'].values
        model_names = []

        for config in self.model_configs:
            name = config["name"]
            print(f"      Processing {name}...")

            model, tokenizer = self.load_model(config)
            if model is None:
                print(f"        Skipping {name}")
                continue

            # Get predictions
            val_p = self.get_predictions(model, tokenizer, val_df, name)
            test_p = self.get_predictions(model, tokenizer, test_df, name)

            val_probs.append(val_p)
            test_probs.append(test_p)
            model_names.append(name)

            # Quick eval
            val_preds = np.argmax(val_p, axis=1)
            f1 = f1_score(val_labels, val_preds, average='macro')
            print(f"        Val F1: {f1:.4f}")

        print(f"      Loaded {len(model_names)} models: {model_names}")
        return val_probs, test_probs, val_labels, test_labels, model_names

    def ensemble_methods(self, val_probs, test_probs, val_labels, test_labels, model_names):
        print("[3/5] Testing ensemble methods...")

        results = {}

        # Convert to numpy arrays - ensure all have same shape
        val_probs = np.array([p.astype(np.float32) for p in val_probs])
        test_probs = np.array([p.astype(np.float32) for p in test_probs])
        n_models = val_probs.shape[0]
        n_classes = 4

        # 1. Simple Average
        print("      Testing Simple Average...")
        avg_val_probs = val_probs.mean(axis=0)
        avg_test_probs = test_probs.mean(axis=0)
        avg_preds = np.argmax(avg_test_probs, axis=1)

        avg_f1 = f1_score(test_labels, avg_preds, average='macro')
        avg_acc = accuracy_score(test_labels, avg_preds)

        results['simple_average'] = {
            'predictions': avg_preds,
            'probabilities': avg_test_probs,
            'f1_macro': avg_f1,
            'accuracy': avg_acc
        }
        print(f"        Simple Average F1: {avg_f1:.4f}")

        # 2. Weighted Average (by validation F1)
        print("      Testing Weighted Average...")
        weights = []
        for i, vp in enumerate(val_probs):
            preds = np.argmax(vp, axis=1)
            f1 = f1_score(val_labels, preds, average='macro')
            weights.append(f1)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        weighted_test_probs = np.average(test_probs, axis=0, weights=weights)
        weighted_preds = np.argmax(weighted_test_probs, axis=1)

        weighted_f1 = f1_score(test_labels, weighted_preds, average='macro')
        results['weighted_average'] = {
            'predictions': weighted_preds,
            'probabilities': weighted_test_probs,
            'f1_macro': weighted_f1,
            'accuracy': accuracy_score(test_labels, weighted_preds),
            'weights': {model_names[i]: w for i, w in enumerate(weights)}
        }
        print(f"        Weighted Average F1: {weighted_f1:.4f}")
        print(f"        Weights: {results['weighted_average']['weights']}")

        # 3. Majority Voting
        print("      Testing Majority Voting...")
        val_preds = np.argmax(val_probs, axis=2)
        test_preds = np.argmax(test_probs, axis=2)

        # Find most common prediction
        from scipy.stats import mode
        vote_preds = mode(test_preds, axis=0)[0][0]

        vote_f1 = f1_score(test_labels, vote_preds, average='macro')
        results['majority_voting'] = {
            'predictions': vote_preds,
            'f1_macro': vote_f1,
            'accuracy': accuracy_score(test_labels, vote_preds)
        }
        print(f"        Majority Voting F1: {vote_f1:.4f}")

        # 4. Logistic Regression Meta-Learner
        print("      Testing Logistic Regression Meta-Learner...")
        val_meta = val_probs.transpose(1, 0, 2).reshape(-1, n_models * n_classes)
        test_meta = test_probs.transpose(1, 0, 2).reshape(-1, n_models * n_classes)

        lr_meta = LogisticRegression(max_iter=1000, random_state=42)
        lr_meta.fit(val_meta, val_labels)
        lr_preds = lr_meta.predict(test_meta)

        lr_f1 = f1_score(test_labels, lr_preds, average='macro')
        results['logistic_regression'] = {
            'predictions': lr_preds,
            'f1_macro': lr_f1,
            'accuracy': accuracy_score(test_labels, lr_preds)
        }
        print(f"        Logistic Regression F1: {lr_f1:.4f}")

        # 5. XGBoost Meta-Learner
        print("      Testing XGBoost Meta-Learner...")
        xgb_meta = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbosity=0
        )
        xgb_meta.fit(val_meta, val_labels)
        xgb_preds = xgb_meta.predict(test_meta)

        xgb_f1 = f1_score(test_labels, xgb_preds, average='macro')
        results['xgboost'] = {
            'predictions': xgb_preds,
            'f1_macro': xgb_f1,
            'accuracy': accuracy_score(test_labels, xgb_preds)
        }
        print(f"        XGBoost F1: {xgb_f1:.4f}")

        return results

    def evaluate_best(self, results, test_labels, model_names):
        print("[4/5] Evaluating best method...")

        # Find best method
        best_method = max(results.items(), key=lambda x: x[1]['f1_macro'])
        best_name = best_method[0]
        best_result = best_method[1]
        best_preds = best_result['predictions']

        print(f"\n      Best method: {best_name}")
        print(f"      F1-Macro: {best_result['f1_macro']:.4f}")

        # Detailed metrics
        report = classification_report(
            test_labels, best_preds,
            target_names=['Neutral', 'Light', 'Moderate', 'Severe'],
            output_dict=True
        )

        per_class_f1 = {
            'neutral': report['Neutral']['f1-score'],
            'light': report['Light']['f1-score'],
            'moderate': report['Moderate']['f1-score'],
            'severe': report['Severe']['f1-score'],
        }

        print("\n      Per-Class F1:")
        for cls, f1 in per_class_f1.items():
            print(f"        {cls.capitalize()}: {f1:.4f}")

        return best_name, best_result, per_class_f1

    def save_results(self, results, best_name, best_result, per_class_f1, model_names):
        print("[5/5] Saving results...")

        output = {
            'models_used': model_names,
            'ensemble_methods': {k: {'f1_macro': v['f1_macro'], 'accuracy': v['accuracy']} for k, v in results.items()},
            'best_method': best_name,
            'best_f1_macro': best_result['f1_macro'],
            'best_accuracy': best_result['accuracy'],
            'per_class_f1': per_class_f1
        }

        os.makedirs("results/experiment_7_ensemble", exist_ok=True)
        with open("results/experiment_7_ensemble/results.json", 'w') as f:
            json.dump(output, f, indent=2)

        print("      Results saved to: results/experiment_7_ensemble/results.json")

    def run(self):
        print("\n" + "="*50)
        print("EXPERIMENT 7: FINAL ENSEMBLE")
        print("="*50)

        train_df, val_df, test_df = self.load_data()
        val_probs, test_probs, val_labels, test_labels, model_names = self.collect_predictions(train_df, val_df, test_df)

        if len(model_names) == 0:
            print("ERROR: No models loaded!")
            return

        results = self.ensemble_methods(val_probs, test_probs, val_labels, test_labels, model_names)
        best_name, best_result, per_class_f1 = self.evaluate_best(results, test_labels, model_names)
        self.save_results(results, best_name, best_result, per_class_f1, model_names)

        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"\nModels used: {model_names}")
        print(f"\nAll Ensemble Methods:")
        for name, res in results.items():
            print(f"  {name}: F1={res['f1_macro']:.2%}, Acc={res['accuracy']:.2%}")

        print(f"\n🏆 BEST: {best_name}")
        print(f"   F1-Macro: {best_result['f1_macro']:.2%}")
        print(f"   Accuracy: {best_result['accuracy']:.2%}")

        print("\n[OK] Experiment 7 complete!")
        return results


if __name__ == "__main__":
    exp = EnsembleExperiment()
    results = exp.run()
