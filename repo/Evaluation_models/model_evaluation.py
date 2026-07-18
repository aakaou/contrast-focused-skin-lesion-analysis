# ======================================================
# analyze_models_FULL_METRICS_HAM10000_ISIC2024.py
# Unified analysis:
#   - HAM10000: 7-class multi-class
#   - ISIC2024: binary melanoma vs non-melanoma
# ======================================================

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("husl")

# ======================================================
# CONFIG
# ======================================================

# Change this path pattern if needed
CSV_PATTERN = "/path/to/pipeline[1-4]_*_predictions.csv"

HAM10000_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
ISIC2024_CLASSES = ['non_mel', 'mel']

SAVE_PREFIX = "benchmark"

# ======================================================
# HELPERS
# ======================================================

def infer_pipeline_model(filename):
    filename = filename.lower()

    pipeline_match = re.search(r'pipeline(\d+)', filename)
    pipeline = f"P{pipeline_match.group(1)}" if pipeline_match else "P?"

    model_match = re.search(
        r'(resnet\d+|densenet\d+|efficientnet[b\d]+|mobilenetv?\d+|vgg\d+|inceptionv3|inception|xception|vit\w*|convnext\w*)',
        filename
    )
    model_name = model_match.group(1).title() if model_match else "Unknown"

    return pipeline, model_name

def infer_dataset_type(df):
    prob_cols = [c for c in df.columns if c.startswith("prob_")]

    # HAM10000: 7-class layout
    if all(f"prob_{cls}" in df.columns for cls in HAM10000_CLASSES):
        return "ham10000"

    # ISIC2024 binary layout
    if "prob_mel" in df.columns:
        return "isic2024"

    # fallback by class ids
    unique_labels = sorted(pd.unique(df["actual_class_id"]))
    if set(unique_labels).issubset({0, 1}):
        return "isic2024"
    if set(unique_labels).issubset(set(range(7))):
        return "ham10000"

    raise ValueError(f"Cannot infer dataset type from columns: {prob_cols}")

def extract_binary_prob(df):
    if "prob_mel" in df.columns:
        return df["prob_mel"].values

    if "prob_1" in df.columns:
        return df["prob_1"].values

    if "prob_non_mel" in df.columns:
        return 1.0 - df["prob_non_mel"].values

    raise ValueError("Binary file must contain 'prob_mel', 'prob_1', or 'prob_non_mel'")

# ======================================================
# METRICS COMPUTATION
# ======================================================

def compute_ham10000_metrics(df, csv_file, pipeline, model_name):
    y_true = df["actual_class_id"].values
    y_pred = df["pred_class_id"].values
    probs = df[[f"prob_{cls}" for cls in HAM10000_CLASSES]].values

    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=HAM10000_CLASSES,
        output_dict=True,
        zero_division=0
    )

    f1_macro = report["macro avg"]["f1-score"]
    precision_macro = report["macro avg"]["precision"]
    recall_macro = report["macro avg"]["recall"]

    y_true_bin = label_binarize(y_true, classes=list(range(len(HAM10000_CLASSES))))
    auc_macro = roc_auc_score(
        y_true_bin,
        probs,
        multi_class="ovr",
        average="macro"
    )

    return {
        "dataset_type": "ham10000",
        "model": model_name,
        "pipeline": pipeline,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auc_macro": auc_macro,
        "pr_auc": np.nan,
        "n_samples": len(df),
        "file": csv_file
    }

def compute_isic2024_metrics(df, csv_file, pipeline, model_name):
    y_true = df["actual_class_id"].values
    y_pred = df["pred_class_id"].values
    prob_mel = extract_binary_prob(df)

    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=ISIC2024_CLASSES,
        output_dict=True,
        zero_division=0
    )

    f1_macro = report["macro avg"]["f1-score"]
    precision_macro = report["macro avg"]["precision"]
    recall_macro = report["macro avg"]["recall"]

    roc_auc = roc_auc_score(y_true, prob_mel)
    pr_auc = average_precision_score(y_true, prob_mel)

    return {
        "dataset_type": "isic2024",
        "model": model_name,
        "pipeline": pipeline,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auc_macro": roc_auc,
        "pr_auc": pr_auc,
        "n_samples": len(df),
        "file": csv_file
    }

# ======================================================
# PLOTTING: HAM10000
# ======================================================

def plot_best_ham10000(df_results_subset):
    if len(df_results_subset) == 0:
        return

    best_row = df_results_subset.loc[df_results_subset["auc_macro"].idxmax()]
    best_df = pd.read_csv(best_row["file"])

    y_true = best_df["actual_class_id"].values
    y_pred = best_df["pred_class_id"].values
    probs = best_df[[f"prob_{cls}" for cls in HAM10000_CLASSES]].values

    fig = plt.figure(figsize=(20, 15))

    # classification report
    plt.subplot(2, 3, 1)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=HAM10000_CLASSES,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).round(3).T.iloc[:len(HAM10000_CLASSES), :3]
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt="g", cbar=False)
    plt.title("Classification Report\n(Precision, Recall, F1)", fontweight="bold")

    # confusion matrix
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=HAM10000_CLASSES,
        yticklabels=HAM10000_CLASSES
    )
    plt.title("Confusion Matrix", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # ROC curves
    plt.subplot(2, 3, 3)
    y_true_bin = label_binarize(y_true, classes=list(range(len(HAM10000_CLASSES))))
    auc_scores = []

    for i, cls in enumerate(HAM10000_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], probs[:, i])
        auc_scores.append(auc_i)
        plt.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC={auc_i:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)", fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)

    # summary panel
    plt.subplot(2, 3, (4, 6))
    metrics_text = f"""
BEST HAM10000 MODEL: {best_row['model']} ({best_row['pipeline']})
-------------------------------------------------
Accuracy        : {best_row['accuracy']:.4f}
F1-Macro        : {best_row['f1_macro']:.4f}
Precision-Macro : {best_row['precision_macro']:.4f}
Recall-Macro    : {best_row['recall_macro']:.4f}
AUC-Macro       : {best_row['auc_macro']:.4f}
N Samples       : {best_row['n_samples']}

PER-CLASS AUC:
"""
    for cls, auc_i in zip(HAM10000_CLASSES, auc_scores):
        metrics_text += f"  {cls}: {auc_i:.3f}\n"

    plt.axis("off")
    plt.text(
        0.05, 0.5, metrics_text,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
    )

    plt.suptitle("HAM10000 COMPLETE METRICS", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_best_model_full_metrics_ham10000.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_summary_ham10000(df_results_subset):
    if len(df_results_subset) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top10_acc = df_results_subset.groupby("model")["accuracy"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    ).head(10)
    axes[0, 0].barh(range(len(top10_acc)), top10_acc["mean"], xerr=top10_acc["std"], capsize=5)
    axes[0, 0].set_yticks(range(len(top10_acc)))
    axes[0, 0].set_yticklabels(top10_acc.index)
    axes[0, 0].set_xlabel("Accuracy ± Std")
    axes[0, 0].set_title("Top 10 Accuracy")

    top10_auc = df_results_subset.groupby("model")["auc_macro"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    ).head(10)
    axes[0, 1].barh(range(len(top10_auc)), top10_auc["mean"], xerr=top10_auc["std"], capsize=5)
    axes[0, 1].set_yticks(range(len(top10_auc)))
    axes[0, 1].set_yticklabels(top10_auc.index)
    axes[0, 1].set_xlabel("Macro AUC ± Std")
    axes[0, 1].set_title("Top 10 Macro AUC")

    for i, pipe in enumerate(["P1", "P2", "P3", "P4"]):
        row, col = divmod(i, 2)
        pipe_best = df_results_subset[df_results_subset["pipeline"] == pipe]
        if len(pipe_best) == 0:
            axes[row, col].set_title(f"{pipe}: no models")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            continue

        pipe_top = pipe_best.nlargest(5, "accuracy")
        axes[row, col].barh(pipe_top["model"], pipe_top["accuracy"], color="coral")
        axes[row, col].set_title(f"{pipe}: Top 5 Accuracy")
        axes[row, col].set_xlabel("Accuracy")

    plt.suptitle("HAM10000 Benchmark: Accuracy + Macro AUC + Pipeline Winners", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_summary_4plots_ham10000.png", dpi=300, bbox_inches="tight")
    plt.show()

# ======================================================
# PLOTTING: ISIC2024
# ======================================================

def plot_best_isic2024(df_results_subset):
    if len(df_results_subset) == 0:
        return

    best_row = df_results_subset.loc[df_results_subset["auc_macro"].idxmax()]
    best_df = pd.read_csv(best_row["file"])

    y_true = best_df["actual_class_id"].values
    y_pred = best_df["pred_class_id"].values
    prob_mel = extract_binary_prob(best_df)

    fig = plt.figure(figsize=(18, 12))

    # classification report
    plt.subplot(2, 3, 1)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=ISIC2024_CLASSES,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).round(3).T.iloc[:2, :3]
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt="g", cbar=False)
    plt.title("Classification Report\n(Precision, Recall, F1)", fontweight="bold")

    # confusion matrix
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=ISIC2024_CLASSES,
        yticklabels=ISIC2024_CLASSES
    )
    plt.title("Confusion Matrix", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # ROC
    plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_true, prob_mel)
    auc_val = roc_auc_score(y_true, prob_mel)
    plt.plot(fpr, tpr, linewidth=2, label=f"Mel (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontweight="bold")
    plt.legend(loc="lower right", fontsize=8)

    # PR curve
    plt.subplot(2, 3, 4)
    precision, recall, _ = precision_recall_curve(y_true, prob_mel)
    pr_auc = average_precision_score(y_true, prob_mel)
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    # summary panel
    plt.subplot(2, 3, (5, 6))
    metrics_text = f"""
BEST ISIC2024 MODEL: {best_row['model']} ({best_row['pipeline']})
-------------------------------------------------
Accuracy        : {best_row['accuracy']:.4f}
F1-Macro        : {best_row['f1_macro']:.4f}
Precision-Macro : {best_row['precision_macro']:.4f}
Recall-Macro    : {best_row['recall_macro']:.4f}
ROC-AUC         : {best_row['auc_macro']:.4f}
PR-AUC          : {best_row['pr_auc']:.4f}
N Samples       : {best_row['n_samples']}

CLASS MAPPING:
  0 -> non_mel
  1 -> mel
"""
    plt.axis("off")
    plt.text(
        0.05, 0.5, metrics_text,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
    )

    plt.suptitle("ISIC2024 COMPLETE METRICS", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_best_model_full_metrics_isic2024.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_summary_isic2024(df_results_subset):
    if len(df_results_subset) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    top10_acc = df_results_subset.groupby("model")["accuracy"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    ).head(10)
    axes[0, 0].barh(range(len(top10_acc)), top10_acc["mean"], xerr=top10_acc["std"], capsize=5)
    axes[0, 0].set_yticks(range(len(top10_acc)))
    axes[0, 0].set_yticklabels(top10_acc.index)
    axes[0, 0].set_xlabel("Accuracy ± Std")
    axes[0, 0].set_title("Top 10 Accuracy")

    top10_auc = df_results_subset.groupby("model")["auc_macro"].agg(["mean", "std"]).sort_values(
        "mean", ascending=False
    ).head(10)
    axes[0, 1].barh(range(len(top10_auc)), top10_auc["mean"], xerr=top10_auc["std"], capsize=5)
    axes[0, 1].set_yticks(range(len(top10_auc)))
    axes[0, 1].set_yticklabels(top10_auc.index)
    axes[0, 1].set_xlabel("ROC-AUC ± Std")
    axes[0, 1].set_title("Top 10 ROC-AUC")

    for i, pipe in enumerate(["P1", "P2", "P3", "P4"]):
        row, col = divmod(i, 2)
        pipe_best = df_results_subset[df_results_subset["pipeline"] == pipe]
        if len(pipe_best) == 0:
            axes[row, col].set_title(f"{pipe}: no models")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            continue

        pipe_top = pipe_best.nlargest(5, "accuracy")
        axes[row, col].barh(pipe_top["model"], pipe_top["accuracy"], color="coral")
        axes[row, col].set_title(f"{pipe}: Top 5 Accuracy")
        axes[row, col].set_xlabel("Accuracy")

    plt.suptitle("ISIC2024 Benchmark: Accuracy + ROC-AUC + Pipeline Winners", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PREFIX}_summary_4plots_isic2024.png", dpi=300, bbox_inches="tight")
    plt.show()

# ======================================================
# MAIN
# ======================================================

results = []
csv_files = glob.glob(CSV_PATTERN)
print(f"🔍 Found {len(csv_files)} files")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        filename = os.path.basename(csv_file)

        required_cols = {"actual_class_id", "pred_class_id"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        pipeline, model_name = infer_pipeline_model(filename)
        dataset_type = infer_dataset_type(df)

        if dataset_type == "ham10000":
            row = compute_ham10000_metrics(df, csv_file, pipeline, model_name)
        elif dataset_type == "isic2024":
            row = compute_isic2024_metrics(df, csv_file, pipeline, model_name)
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

        results.append(row)

    except Exception as e:
        print(f"⚠️ Skip {csv_file}: {e}")

df_results = pd.DataFrame(results)
print(f"✅ Loaded {len(df_results)} experiments")

if len(df_results) > 0:
    df_results.to_csv(f"{SAVE_PREFIX}_all_models_enhanced_results.csv", index=False)

    ham_df = df_results[df_results["dataset_type"] == "ham10000"].reset_index(drop=True)
    isic_df = df_results[df_results["dataset_type"] == "isic2024"].reset_index(drop=True)

    if len(ham_df) > 0:
        ham_df.to_csv(f"{SAVE_PREFIX}_ham10000_results.csv", index=False)
        plot_best_ham10000(ham_df)
        plot_summary_ham10000(ham_df)

    if len(isic_df) > 0:
        isic_df.to_csv(f"{SAVE_PREFIX}_isic2024_results.csv", index=False)
        plot_best_isic2024(isic_df)
        plot_summary_isic2024(isic_df)

    print("\n✅ SAVED:")
    print(f"   • {SAVE_PREFIX}_all_models_enhanced_results.csv")
    if len(ham_df) > 0:
        print(f"   • {SAVE_PREFIX}_ham10000_results.csv")
        print(f"   • {SAVE_PREFIX}_best_model_full_metrics_ham10000.png")
        print(f"   • {SAVE_PREFIX}_summary_4plots_ham10000.png")
    if len(isic_df) > 0:
        print(f"   • {SAVE_PREFIX}_isic2024_results.csv")
        print(f"   • {SAVE_PREFIX}_best_model_full_metrics_isic2024.png")
        print(f"   • {SAVE_PREFIX}_summary_4plots_isic2024.png")
else:
    print("No valid experiments found.")
