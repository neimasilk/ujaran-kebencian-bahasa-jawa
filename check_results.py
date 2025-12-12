import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    try:
        with open('results/integrated_custom_ensemble_results.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded results from: results/integrated_custom_ensemble_results.json")
        print(f"Best Method: {data['best_method']['method']}")
        print(f"Accuracy: {data['best_method']['test_accuracy']:.4f}")
        print(f"F1-Macro: {data['best_method']['test_f1_macro']:.4f}")

        # Note: The JSON doesn't store per-instance predictions, so we can't generate
        # a confusion matrix from the JSON alone unless we saved predictions.
        # However, we can visualize the model comparison bar chart.
        
        models = [res['model'] for res in data['individual_models']]
        f1_scores = [res['f1_macro'] for res in data['individual_models']]
        
        # Add ensemble to the list
        models.append(f"Ensemble ({data['best_method']['method']})")
        f1_scores.append(data['best_method']['test_f1_macro'])
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=f1_scores)
        plt.title('Model Performance Comparison (F1-Macro)')
        plt.ylabel('F1-Macro Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
            
        plt.tight_layout()
        plt.savefig('ensemble_comparison_chart.png')
        print("Chart saved to ensemble_comparison_chart.png")

    except FileNotFoundError:
        print("Results file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()