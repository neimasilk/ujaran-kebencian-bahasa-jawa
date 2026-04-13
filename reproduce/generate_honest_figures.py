#!/usr/bin/env python3
"""
Fase 5: Generate Honest Figures

ALL figures are generated from REAL experimental data:
- Figure 1: Dataset distribution (from cleaned dataset)
- Figure 2: Confusion matrix (from real model inference)
- Figure 3: Comparative bar chart (from real training results)
- Figure 4: Label smoothing ablation (from real ablation study)
- Table 2: Comprehensive results table

Input:  data/cleaned/final_dataset.csv
        results/comparative_results.json
        results/ablation_results.json
        results/multi_seed_results.json
Output: paper/figures/*.png, paper/figures/table2_performance.tex
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned", "final_dataset.csv")
COMPARATIVE_RESULTS = os.path.join(PROJECT_ROOT, "results", "comparative_results.json")
ABLATION_RESULTS = os.path.join(PROJECT_ROOT, "results", "ablation_results.json")
MULTI_SEED_RESULTS = os.path.join(PROJECT_ROOT, "results", "multi_seed_results.json")
HONEST_EVAL_RESULTS = os.path.join(PROJECT_ROOT, "results", "honest_evaluation_results.json")
BASELINE_RESULTS = os.path.join(PROJECT_ROOT, "results", "baseline_results.json")
AUGMENTATION_RESULTS = os.path.join(PROJECT_ROOT, "results", "augmentation_impact.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")

LABEL_NAMES = [
    "Not Hate Speech",
    "Light Hate",
    "Moderate Hate",
    "Severe Hate",
]

COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]


def load_json(path):
    """Load JSON file, return None if not found."""
    if not os.path.exists(path):
        print(f"[WARN] Not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_figure1_dataset_distribution():
    """Figure 1: Dataset distribution from cleaned dataset."""
    print("\n" + "=" * 60)
    print("FIGURE 1: Dataset Distribution")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    total = len(df)
    label_counts = df["label"].value_counts().sort_index()

    print(f"Total samples: {total}")
    for idx, count in label_counts.items():
        print(f"  {LABEL_NAMES[idx]}: {count} ({count/total*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = axes[0].bar(LABEL_NAMES, label_counts.values, color=COLORS,
                       edgecolor="black", linewidth=1.2)
    axes[0].set_xlabel("Label", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=13, fontweight="bold")
    axes[0].set_title(f"Dataset Label Distribution (N={total:,})", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=10)
    axes[0].set_ylim(0, max(label_counts.values) * 1.15)

    for bar, count in zip(bars, label_counts.values):
        height = bar.get_height()
        pct = count / total * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, height + 50,
                     f"{count}\n({pct:.1f}%)", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    # Pie chart
    percentages = [c / total * 100 for c in label_counts.values]
    wedges, texts, autotexts = axes[1].pie(
        percentages, labels=LABEL_NAMES, autopct="%1.1f%%",
        colors=COLORS, startangle=90, explode=(0.02, 0.02, 0.02, 0.02)
    )
    axes[1].set_title("Label Proportion (%)", fontsize=14, fontweight="bold")
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure1_dataset_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


def generate_figure2_confusion_matrix():
    """Figure 2: Confusion matrix from the best model's real predictions."""
    print("\n" + "=" * 60)
    print("FIGURE 2: Confusion Matrix (Real)")
    print("=" * 60)

    # Try comparative results first (retrained models)
    data = load_json(COMPARATIVE_RESULTS)
    if data and "best_model" in data:
        best_key = data["best_model"]["key"]
        model_data = data["models"][best_key]
        cm = np.array(model_data["confusion_matrix"])
        f1_macro = model_data["test"]["f1_macro"]
        accuracy = model_data["test"]["accuracy"]
        model_name = model_data["description"]
        n_test = sum(sum(row) for row in cm)
    else:
        # Fallback to honest evaluation results
        data = load_json(HONEST_EVAL_RESULTS)
        if not data or "best_model" not in data:
            print("[SKIP] No results available for confusion matrix")
            return
        best_name = data["best_model"]["name"]
        model_data = data["models"][best_name]
        cm = np.array(model_data["confusion_matrix"])
        f1_macro = model_data["f1_macro"]
        accuracy = model_data["accuracy"]
        model_name = best_name
        n_test = model_data["test_samples"]

    print(f"Model: {model_name}")
    print(f"F1-Macro: {f1_macro:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 8))

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                cbar_kws={"label": "Number of Samples"}, linewidths=1, linecolor="white")

    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Label", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix - {model_name}\n"
                 f"F1-Macro: {f1_macro:.2f}% | Accuracy: {accuracy:.2f}% | N={n_test}",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure2_confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


def generate_figure3_model_comparison():
    """Figure 3: Comparative bar chart of all models on test set."""
    print("\n" + "=" * 60)
    print("FIGURE 3: Model Comparison (Real)")
    print("=" * 60)

    data = load_json(COMPARATIVE_RESULTS)
    if not data:
        print("[SKIP] Comparative results not available")
        return

    models = data["models"]
    names = []
    val_f1s = []
    test_f1s = []

    for key, m in models.items():
        names.append(m["description"].replace(" (", "\n("))
        val_f1s.append(m["validation"]["f1_macro"])
        test_f1s.append(m["test"]["f1_macro"])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, val_f1s, width, label="Validation F1",
                   color="#3498db", edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, test_f1s, width, label="Test F1",
                   color="#2ecc71", edgecolor="black", linewidth=1.2)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    # Highlight gaps
    for i, (v, t) in enumerate(zip(val_f1s, test_f1s)):
        gap = v - t
        if abs(gap) > 0.5:
            color = "green" if gap < 1 else ("orange" if gap < 3 else "red")
            y_pos = max(v, t) + 2
            ax.annotate(f"Gap: {gap:+.1f}%", xy=(i, y_pos),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1-Macro (%)", fontsize=13, fontweight="bold")
    ax.set_title("Model Comparison: Validation vs Test F1-Macro",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")

    # Dynamic y-axis
    all_scores = val_f1s + test_f1s
    y_min = max(0, min(all_scores) - 5)
    y_max = max(all_scores) + 6
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure3_model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


def generate_figure4_ablation():
    """Figure 4: Label smoothing ablation from real results."""
    print("\n" + "=" * 60)
    print("FIGURE 4: Label Smoothing Ablation (Real)")
    print("=" * 60)

    data = load_json(ABLATION_RESULTS)
    if not data:
        print("[SKIP] Ablation results not available")
        return

    results = data["results"]
    epsilons = [r["epsilon"] for r in results]
    f1_scores = [r["test"]["f1_macro"] for r in results]
    acc_scores = [r["test"]["accuracy"] for r in results]

    optimal_idx = f1_scores.index(max(f1_scores))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epsilons, f1_scores, marker="o", linewidth=2.5,
            markersize=10, label="F1-Macro", color="#3498db")
    ax.plot(epsilons, acc_scores, marker="s", linewidth=2.5,
            markersize=10, label="Accuracy", color="#2ecc71")

    # Highlight optimal
    ax.scatter([epsilons[optimal_idx]], [f1_scores[optimal_idx]],
               s=300, facecolors="none", edgecolors="red", linewidth=3)
    ax.annotate(f"Optimal\n\u03b5={epsilons[optimal_idx]}",
                xy=(epsilons[optimal_idx], f1_scores[optimal_idx]),
                xytext=(epsilons[optimal_idx] + 0.03, f1_scores[optimal_idx] - 1.5),
                fontsize=11, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xlabel("Label Smoothing Epsilon (\u03b5)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("Label Smoothing Hyperparameter Ablation (IndoBERT)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xticks(epsilons)

    # Dynamic y range
    all_scores = f1_scores + acc_scores
    y_min = min(all_scores) - 2
    y_max = max(all_scores) + 2
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure4_label_smoothing_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")

    # Print summary
    print(f"\nAblation Results:")
    for e, f1, acc in zip(epsilons, f1_scores, acc_scores):
        marker = " <-- optimal" if e == epsilons[optimal_idx] else ""
        print(f"  eps={e:.2f}: F1={f1:.2f}% Acc={acc:.2f}%{marker}")


def generate_table2():
    """Table 2: Comprehensive results in LaTeX."""
    print("\n" + "=" * 60)
    print("TABLE 2: Results Table")
    print("=" * 60)

    data = load_json(COMPARATIVE_RESULTS)
    if not data:
        print("[SKIP] Comparative results not available")
        return

    # Also load multi-seed stats if available
    multi_seed = load_json(MULTI_SEED_RESULTS)

    models = data["models"]
    best_key = data["best_model"]["key"]

    # Print text table
    print(f"\n{'Model':<40s} {'Val F1':>8s} {'Test F1':>8s} {'Test Acc':>8s} {'Test P':>8s} {'Test R':>8s}")
    print("-" * 80)
    for key, m in models.items():
        marker = " *" if key == best_key else ""
        print(f"{m['description']:<40s} "
              f"{m['validation']['f1_macro']:>7.2f}% "
              f"{m['test']['f1_macro']:>7.2f}% "
              f"{m['test']['accuracy']:>7.2f}% "
              f"{m['test'].get('precision_macro', 0):>7.2f}% "
              f"{m['test'].get('recall_macro', 0):>7.2f}%{marker}")

    if multi_seed:
        stats = multi_seed["statistics"]
        print(f"\n{'Statistical Significance (5 seeds):':<40s}")
        print(f"  F1-Macro:  {stats['f1_macro']['mean']:.2f}% +/- {stats['f1_macro']['std']:.2f}%")
        print(f"  Accuracy:  {stats['accuracy']['mean']:.2f}% +/- {stats['accuracy']['std']:.2f}%")

    # Generate LaTeX
    latex = """% Table 2: Comparative Model Performance
\\begin{table}[h]
\\centering
\\caption{Perbandingan Performa Model pada Test Set}
\\label{tab:model_comparison}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Model} & \\textbf{Val F1} & \\textbf{Test F1} & \\textbf{Test Acc} & \\textbf{Test P} \\\\
\\hline
"""
    for key, m in models.items():
        bold = "\\textbf" if key == best_key else ""
        name = m["description"].replace("_", "\\_")
        if bold:
            latex += (f"{bold}{{{name}}} & {bold}{{{m['validation']['f1_macro']:.2f}}} & "
                      f"{bold}{{{m['test']['f1_macro']:.2f}}} & "
                      f"{bold}{{{m['test']['accuracy']:.2f}}} & "
                      f"{bold}{{{m['test'].get('precision_macro', 0):.2f}}} \\\\\n")
        else:
            latex += (f"{name} & {m['validation']['f1_macro']:.2f} & "
                      f"{m['test']['f1_macro']:.2f} & "
                      f"{m['test']['accuracy']:.2f} & "
                      f"{m['test'].get('precision_macro', 0):.2f} \\\\\n")

    latex += "\\hline\n"

    if multi_seed:
        stats = multi_seed["statistics"]
        latex += (f"\\multicolumn{{5}}{{l}}{{\\textit{{Statistical significance (5 seeds): "
                  f"F1 = {stats['f1_macro']['mean']:.2f}\\% $\\pm$ {stats['f1_macro']['std']:.2f}\\%}}}} \\\\\n")

    latex += """\\hline
\\end{tabular}
\\end{table}
"""

    path = os.path.join(OUTPUT_DIR, "table2_performance.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"\n[OK] Saved: {path}")

    # Per-class table for best model
    best_model = models[best_key]
    if "per_class" in best_model:
        latex_class = """% Table 3: Per-Class Performance of Best Model
\\begin{table}[h]
\\centering
\\caption{Per-Class Performance: """ + best_model["description"].replace("_", "\\_") + """}
\\label{tab:per_class}
\\begin{tabular}{lccc}
\\hline
\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\
\\hline
"""
        for name in LABEL_NAMES:
            if name in best_model["per_class"]:
                pc = best_model["per_class"][name]
                latex_class += f"{name} & {pc['precision']:.2f} & {pc['recall']:.2f} & {pc['f1']:.2f} \\\\\n"

        latex_class += f"""\\hline
\\textbf{{Macro Average}} & \\textbf{{{best_model['test'].get('precision_macro', 0):.2f}}} & \\textbf{{{best_model['test'].get('recall_macro', 0):.2f}}} & \\textbf{{{best_model['test']['f1_macro']:.2f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        path_class = os.path.join(OUTPUT_DIR, "table3_per_class.tex")
        with open(path_class, "w") as f:
            f.write(latex_class)
        print(f"[OK] Saved: {path_class}")


def generate_figure5_baseline_vs_transformer():
    """Figure 5: Baseline vs Transformer comparison bar chart."""
    print("\n" + "=" * 60)
    print("FIGURE 5: Baseline vs Transformer Comparison")
    print("=" * 60)

    baseline = load_json(BASELINE_RESULTS)
    comp = load_json(COMPARATIVE_RESULTS)
    if not baseline or not comp:
        print("[SKIP] Baseline or comparative results not available")
        return

    # Collect all models
    model_names = []
    test_f1s = []
    test_accs = []
    colors = []

    # Baselines
    for key, m in baseline["models"].items():
        model_names.append(m["description"])
        test_f1s.append(m["test"]["f1_macro"])
        test_accs.append(m["test"]["accuracy"])
        colors.append("#95a5a6")  # gray for baselines

    # Transformers
    transformer_colors = {"indobert": "#3498db", "xlmr_large": "#2ecc71", "indobert_ls": "#9b59b6"}
    for key, m in comp["models"].items():
        model_names.append(m["description"])
        test_f1s.append(m["test"]["f1_macro"])
        test_accs.append(m["test"]["accuracy"])
        colors.append(transformer_colors.get(key, "#e74c3c"))

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width / 2, test_f1s, width, label="F1-Macro",
                   color=colors, edgecolor="black", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, test_accs, width, label="Accuracy",
                   color=[c + "80" if len(c) == 7 else c for c in colors],
                   edgecolor="black", linewidth=0.8, alpha=0.6)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Add separator line between baselines and transformers
    n_baselines = len(baseline["models"])
    ax.axvline(x=n_baselines - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_baselines / 2 - 0.5, max(test_f1s) + 3, "Baselines",
            ha="center", fontsize=10, style="italic", color="gray")
    ax.text((n_baselines + len(comp["models"]) / 2), max(test_f1s) + 3, "Transformers",
            ha="center", fontsize=10, style="italic", color="gray")

    # Short names for x labels
    short_names = [n.replace(" + TF-IDF", "\n+TF-IDF").replace(" (", "\n(")
                   for n in model_names]

    ax.set_xlabel("Model", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (%)", fontsize=13, fontweight="bold")
    ax.set_title("Baseline vs Transformer Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=11, loc="lower right")

    all_scores = test_f1s + test_accs
    y_min = max(0, min(all_scores) - 5)
    y_max = max(all_scores) + 5
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "figure5_baseline_vs_transformer.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {path}")


def main():
    print("=" * 60)
    print("FASE 5: GENERATE HONEST FIGURES")
    print("All figures from REAL experimental data")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Figure 1: Always available (uses dataset directly)
    generate_figure1_dataset_distribution()

    # Figure 2: Needs comparative or honest eval results
    generate_figure2_confusion_matrix()

    # Figure 3: Needs comparative results
    generate_figure3_model_comparison()

    # Figure 4: Needs ablation results
    generate_figure4_ablation()

    # Table 2: Needs comparative results
    generate_table2()

    # Figure 5: Baseline vs Transformer comparison
    generate_figure5_baseline_vs_transformer()

    print(f"\n{'='*60}")
    print("FIGURES GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_DIR}")

    # List generated files
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")

    print("\nNOTE: All figures are from real experimental data.")
    print("No synthetic or fabricated data was used.")


if __name__ == "__main__":
    main()
