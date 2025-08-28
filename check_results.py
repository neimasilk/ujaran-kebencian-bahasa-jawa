import json

with open('models/retrained_best/training_results.json', 'r') as f:
    data = json.load(f)

print('=== FINAL TRAINING RESULTS ===')
print(f'F1-Macro: {data["final_metrics"]["eval_f1_macro"]:.4f}')
print(f'Accuracy: {data["final_metrics"]["eval_accuracy"]:.4f}')
print(f'F1-Weighted: {data["final_metrics"]["eval_f1_weighted"]:.4f}')
print(f'Precision-Macro: {data["final_metrics"]["eval_precision_macro"]:.4f}')
print(f'Recall-Macro: {data["final_metrics"]["eval_recall_macro"]:.4f}')

print('\n=== BEST HYPERPARAMETERS USED ===')
for key, value in data['best_hyperparameters'].items():
    print(f'{key}: {value}')