"""
Generate figures for the Javanese Hate Speech Detection paper
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.font_manager as fm

# Set style for paper-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def create_architecture_diagram():
    """Create model architecture diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'IndoBERT + Label Smoothing Architecture',
            ha='center', fontsize=14, weight='bold')

    # Define box positions
    boxes = [
        (5, 8.2, 'Input Text\n(Javanese)', 2.5, 0.8, '#E8F4FD'),
        (5, 7.0, 'IndoBERT Base\n(110M parameters)', 3.0, 1.0, '#D6E3BC'),
        (5, 5.6, 'Dropout\n(0.1)', 2.0, 0.6, '#FDE4D0'),
        (5, 4.4, 'Classification Layer\n(768 -> 4)', 2.5, 0.8, '#E8F4FD'),
        (5, 3.0, 'Label Smoothing\n(epsilon = 0.1)', 2.5, 0.8, '#FDE4D0'),
        (5, 1.8, 'Softmax', 2.0, 0.6, '#D6E3BC'),
        (5, 0.6, 'Output: P(class|input)', 2.5, 0.8, '#E8F4FD'),
    ]

    # Draw boxes
    for x, y, text, w, h, color in boxes:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)

    # Draw arrows
    for i in range(len(boxes) - 1):
        x1, y1 = boxes[i][0], boxes[i][1] - boxes[i][4]/2
        x2, y2 = boxes[i+1][0], boxes[i+1][1] + boxes[i+1][4]/2
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='#333333')
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/architecture_diagram.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/architecture_diagram.pdf', bbox_inches='tight', dpi=300)
    print("Architecture diagram saved!")
    plt.close()

def create_confusion_matrix():
    """Create confusion matrix visualization"""
    # Normalized confusion matrix
    cm = np.array([
        [0.803, 0.137, 0.040, 0.020],
        [0.125, 0.762, 0.075, 0.038],
        [0.024, 0.080, 0.840, 0.056],
        [0.023, 0.030, 0.045, 0.902]
    ])

    labels = ['Neutral', 'Light\nHate', 'Moderate\nHate', 'Severe\nHate']

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, f'{cm[i, j]:.3f}',
                          ha="center", va="center",
                          color="white" if cm[i, j] > 0.5 else "black",
                          fontsize=11, weight='bold')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
    ax.set_ylabel('True Label', fontsize=12, weight='bold')
    ax.set_title('Confusion Matrix (Normalized)\nIndoBERT + Label Smoothing (epsilon=0.1)',
                 fontsize=13, weight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/confusion_matrix.pdf', bbox_inches='tight', dpi=300)
    print("Confusion matrix saved!")
    plt.close()

def create_per_class_performance():
    """Create per-class F1-score bar chart"""
    classes = ['Neutral', 'Light\nHate', 'Moderate\nHate', 'Severe\nHate']
    f1_scores = [79.83, 74.77, 85.09, 85.84]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#4CAF50', '#FF9800', '#2196F3', '#F44336']
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{score}%', ha='center', va='bottom',
                fontsize=11, weight='bold')

    ax.set_ylabel('F1-Score (%)', fontsize=12, weight='bold')
    ax.set_xlabel('Class', fontsize=12, weight='bold')
    ax.set_title('Per-Class F1-Score Performance\nIndoBERT + Label Smoothing (epsilon=0.1)',
                 fontsize=13, weight='bold', pad=15)
    ax.set_ylim(70, 90)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=81.38, color='red', linestyle='--', linewidth=2, label='Macro Average (81.38%)')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/per_class_f1.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/per_class_f1.pdf', bbox_inches='tight', dpi=300)
    print("Per-class F1 chart saved!")
    plt.close()

def create_loss_function_comparison():
    """Create loss function comparison chart"""
    loss_functions = ['Cross-\nEntropy', 'Focal\nLoss', 'Label\nSmoothing', 'Focal +\nLabel\nSmooth']
    neutral = [76.21, 76.50, 79.83, 79.45]
    light = [72.73, 72.50, 74.77, 74.20]
    moderate = [83.67, 83.45, 85.09, 85.25]
    severe = [84.14, 83.95, 85.84, 85.98]

    x = np.arange(len(loss_functions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - 1.5*width, neutral, width, label='Neutral', color='#9E9E9E', edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, light, width, label='Light Hate', color='#FF9800', edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, moderate, width, label='Moderate Hate', color='#2196F3', edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, severe, width, label='Severe Hate', color='#F44336', edgecolor='black')

    ax.set_ylabel('F1-Score (%)', fontsize=12, weight='bold')
    ax.set_xlabel('Loss Function', fontsize=12, weight='bold')
    ax.set_title('Loss Function Ablation Study: Per-Class F1-Scores',
                 fontsize=13, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(loss_functions)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(70, 90)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/loss_function_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/loss_function_comparison.pdf', bbox_inches='tight', dpi=300)
    print("Loss function comparison saved!")
    plt.close()

def create_model_comparison():
    """Create model comparison chart"""
    models = ['mBERT', 'XLM-R\nBase', 'IndoBERT\nBase', 'IndoBERT\n+ LS', 'Custom\nBERT v3', 'XLM-R\nLarge']
    f1_scores = [77.93, 78.38, 79.19, 81.38, 78.26, 81.11]

    # Highlight the best model
    colors = ['#9E9E9E'] * 6
    colors[3] = '#4CAF50'  # IndoBERT + Label Smoothing

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        color = 'white' if score > 80 else 'black'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{score}%', ha='center', va='bottom',
                fontsize=10, weight='bold', color=color)

    ax.set_ylabel('F1-Macro Score (%)', fontsize=12, weight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, weight='bold')
    ax.set_title('Baseline Model Comparison\n(F1-Macro on Test Set)',
                 fontsize=13, weight='bold', pad=15)
    ax.set_ylim(75, 85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation for best model
    ax.annotate('Best Model\n(+2.19% vs baseline)',
                xy=(3, 81.38), xytext=(4.5, 83),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', weight='bold',
                ha='center')

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/model_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/model_comparison.pdf', bbox_inches='tight', dpi=300)
    print("Model comparison saved!")
    plt.close()

def create_ensemble_overfitting():
    """Create ensemble overfitting visualization"""
    methods = ['Single\nModel\n(IndoBERT+LS)', 'Simple\nSoft\nVoting', 'Weighted\nVoting', 'Meta-Learner\nStacking']
    val_scores = [81.13, 82.50, 84.20, 94.09]
    test_scores = [81.38, 79.80, 78.50, 79.50]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, val_scores, width, label='Validation F1',
                   color='#2196F3', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, test_scores, width, label='Test F1',
                   color='#FF9800', edgecolor='black', linewidth=1.5)

    # Add gap indicators
    for i, (val, test) in enumerate(zip(val_scores, test_scores)):
        gap = val - test
        if gap > 0:
            ax.text(i, max(val, test) + 1, f'+{gap:.1f}%',
                   ha='center', fontsize=9, color='red', weight='bold')

    ax.set_ylabel('F1-Macro Score (%)', fontsize=12, weight='bold')
    ax.set_xlabel('Ensemble Method', fontsize=12, weight='bold')
    ax.set_title('Ensemble Overfitting Analysis\n(Validation vs Test Performance)',
                 fontsize=13, weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(75, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation for overfitting
    ax.annotate('Severe Overfitting!\n14.59% validation-test gap',
                xy=(3, 94.09), xytext=(2, 97),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', weight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/ensemble_overfitting.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/ensemble_overfitting.pdf', bbox_inches='tight', dpi=300)
    print("Ensemble overfitting chart saved!")
    plt.close()

def create_hard_negative_analysis():
    """Create hard negative analysis chart"""
    classes = ['Neutral', 'Light\nHate', 'Moderate\nHate', 'Severe\nHate']
    hard_samples = [20, 23, 10, 6]
    percentages = [8.1, 9.6, 4.0, 2.3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left chart: Hard sample counts
    colors = ['#9E9E9E', '#FF9800', '#2196F3', '#F44336']
    bars1 = ax1.bar(classes, hard_samples, color=colors, edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars1, hard_samples):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom',
                fontsize=11, weight='bold')

    ax1.set_ylabel('Number of Hard Samples', fontsize=11, weight='bold')
    ax1.set_xlabel('True Class', fontsize=11, weight='bold')
    ax1.set_title('Hard Negatives by Class\n(Total: 59 samples, 5.9% of test set)',
                  fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right chart: Percentages
    bars2 = ax2.bar(classes, percentages, color=colors, edgecolor='black', linewidth=1.5)

    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{pct}%', ha='center', va='bottom',
                fontsize=11, weight='bold')

    ax2.set_ylabel('% of Class', fontsize=11, weight='bold')
    ax2.set_xlabel('True Class', fontsize=11, weight='bold')
    ax2.set_title('Hard Negatives as Percentage of Each Class',
                  fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/hard_negative_analysis.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/hard_negative_analysis.pdf', bbox_inches='tight', dpi=300)
    print("Hard negative analysis chart saved!")
    plt.close()

def create_dataset_comparison():
    """Create dataset variant comparison chart"""
    datasets = ['Original\n(Imbalanced)', 'Phase 3+4\n(Balanced)', 'Phase 5\n(DeepSeek)']
    sizes = [8000, 10019, 10019]
    f1_scores = [76.50, 81.38, 77.13]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left chart: Dataset sizes
    bars1 = ax1.bar(datasets, sizes, color=['#9E9E9E', '#4CAF50', '#FF9800'],
                    edgecolor='black', linewidth=1.5)

    for bar, size in zip(bars1, sizes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{size:,}', ha='center', va='bottom',
                fontsize=11, weight='bold')

    ax1.set_ylabel('Dataset Size', fontsize=11, weight='bold')
    ax1.set_title('Dataset Sizes', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Right chart: F1 scores
    bars2 = ax2.bar(datasets, f1_scores, color=['#9E9E9E', '#4CAF50', '#FF9800'],
                    edgecolor='black', linewidth=1.5)

    for bar, score in zip(bars2, f1_scores):
        height = bar.get_height()
        color = 'white' if score > 80 else 'black'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{score}%', ha='center', va='bottom',
                fontsize=11, weight='bold', color=color)

    ax2.set_ylabel('F1-Macro Score (%)', fontsize=11, weight='bold')
    ax2.set_title('Performance by Dataset Variant', fontsize=12, weight='bold')
    ax2.set_ylim(70, 85)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    ax2.annotate('Best\nDataset', xy=(1, 81.38), xytext=(1, 84),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', weight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax2.annotate('AI Re-labeling\nDegraded\nPerformance', xy=(2, 77.13), xytext=(2, 80),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, color='red', weight='bold',
                ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig('docs/paper-submission/figures/dataset_comparison.png', bbox_inches='tight', dpi=300)
    plt.savefig('docs/paper-submission/figures/dataset_comparison.pdf', bbox_inches='tight', dpi=300)
    print("Dataset comparison saved!")
    plt.close()

if __name__ == "__main__":
    import os

    # Create figures directory
    os.makedirs('docs/paper-submission/figures', exist_ok=True)

    print("Generating paper figures...")
    create_architecture_diagram()
    create_confusion_matrix()
    create_per_class_performance()
    create_loss_function_comparison()
    create_model_comparison()
    create_ensemble_overfitting()
    create_hard_negative_analysis()
    create_dataset_comparison()
    print("\nAll figures generated successfully!")
    print("Location: docs/paper-submission/figures/")
    print("\nGenerated files:")
    print("  - architecture_diagram.png/pdf")
    print("  - confusion_matrix.png/pdf")
    print("  - per_class_f1.png/pdf")
    print("  - loss_function_comparison.png/pdf")
    print("  - model_comparison.png/pdf")
    print("  - ensemble_overfitting.png/pdf")
    print("  - hard_negative_analysis.png/pdf")
    print("  - dataset_comparison.png/pdf")
