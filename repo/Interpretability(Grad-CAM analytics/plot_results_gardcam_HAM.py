import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# =========================================================
# SETTINGS
# =========================================================
PRED_CSV = "/path/to/ham10000/efficientnet_b7_gradcam_5fold_cv_memory_safe_7class/cv_all_predictions.csv"
GRADCAM_CSV = "/path/to/ham10000/efficientnet_b7_gradcam_5fold_cv_memory_safe_7class/cv_all_gradcam_metrics.csv"
OUTPUT_DIR = Path("ham10000_figures_tables")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)

PROB_COLS = [f"prob_{c}" for c in CLASS_NAMES]
GRADCAM_METRIC_COLS = ["iou", "dice", "percent_cam_inside_lesion", "pixelwise_correlation"]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

# =========================================================
# LOAD DATA
# =========================================================
pred_df = pd.read_csv(PRED_CSV)
gradcam_df = pd.read_csv(GRADCAM_CSV)

# normalize column names just in case
pred_df.columns = [c.strip() for c in pred_df.columns]
gradcam_df.columns = [c.strip() for c in gradcam_df.columns]

# numeric safety
for c in ["fold", "true_label", "predicted_label"] + PROB_COLS:
    if c in pred_df.columns:
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")

for c in ["fold", "true_label", "predicted_label", "target_class_for_cam", "cam_threshold"] + PROB_COLS + GRADCAM_METRIC_COLS:
    if c in gradcam_df.columns:
        gradcam_df[c] = pd.to_numeric(gradcam_df[c], errors="coerce")

pred_df = pred_df.dropna(subset=["fold", "true_label", "predicted_label"]).copy()
gradcam_df = gradcam_df.dropna(subset=["fold", "true_label", "predicted_label"]).copy()

pred_df["fold"] = pred_df["fold"].astype(int)
pred_df["true_label"] = pred_df["true_label"].astype(int)
pred_df["predicted_label"] = pred_df["predicted_label"].astype(int)

gradcam_df["fold"] = gradcam_df["fold"].astype(int)
gradcam_df["true_label"] = gradcam_df["true_label"].astype(int)
gradcam_df["predicted_label"] = gradcam_df["predicted_label"].astype(int)

# keep only valid classes
pred_df = pred_df[pred_df["true_label"].between(0, NUM_CLASSES - 1)]
pred_df = pred_df[pred_df["predicted_label"].between(0, NUM_CLASSES - 1)]

gradcam_df = gradcam_df[gradcam_df["true_label"].between(0, NUM_CLASSES - 1)]
gradcam_df = gradcam_df[gradcam_df["predicted_label"].between(0, NUM_CLASSES - 1)]

# rows with valid gradcam metrics
gradcam_valid = gradcam_df.copy()
for c in GRADCAM_METRIC_COLS:
    gradcam_valid[c] = pd.to_numeric(gradcam_valid[c], errors="coerce")

# =========================================================
# HELPERS
# =========================================================
def mean_sd_text(series, scale=1.0, decimals=2):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{s.mean()*scale:.{decimals}f} ± {s.std(ddof=1)*scale:.{decimals}f}"

def save_df(df, name):
    out_csv = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(out_csv, index=False)
    return out_csv

def multiclass_roc_data(y_true, y_prob, class_names):
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    roc_info = {}
    valid_aucs = []

    for i, cls in enumerate(class_names):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        roc_info[cls] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        valid_aucs.append(roc_auc)

    all_fpr = np.unique(np.concatenate([roc_info[c]["fpr"] for c in roc_info]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in roc_info:
        mean_tpr += np.interp(all_fpr, roc_info[c]["fpr"], roc_info[c]["tpr"])
    mean_tpr /= max(1, len(roc_info))
    macro_auc = auc(all_fpr, mean_tpr) if len(all_fpr) > 1 else np.nan

    return roc_info, all_fpr, mean_tpr, macro_auc

def plot_conf_mat(cm, title, out_path=None, ax=None):
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        own_fig = True

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if np.issubdtype(cm.dtype, np.floating) else "d",
        cmap="Blues",
        xticklabels=[c.upper() for c in CLASS_NAMES],
        yticklabels=[c.upper() for c in CLASS_NAMES],
        cbar=True,
        ax=ax,
        square=True,
        annot_kws={"size": 8}
    )
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if own_fig:
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, bbox_inches="tight")
        plt.close()

# =========================================================
# TABLE 1: Classification metrics for each fold
# =========================================================
table1_rows = []

for fold in sorted(pred_df["fold"].unique()):
    d = pred_df[pred_df["fold"] == fold].copy()
    y_true = d["true_label"].values
    y_pred = d["predicted_label"].values
    y_prob = d[PROB_COLS].values

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = np.nan

    table1_rows.append({
        "Fold": fold,
        "Accuracy (%)": acc * 100,
        "Precision_macro (%)": prec_macro * 100,
        "Recall_macro (%)": rec_macro * 100,
        "F1_macro (%)": f1_macro * 100,
        "AUC_macro (%)": auc_macro * 100 if pd.notna(auc_macro) else np.nan,
    })

table1 = pd.DataFrame(table1_rows)
mean_row = {
    "Fold": "Mean ± SD",
    "Accuracy (%)": mean_sd_text(table1["Accuracy (%)"], scale=1.0, decimals=2),
    "Precision_macro (%)": mean_sd_text(table1["Precision_macro (%)"], scale=1.0, decimals=2),
    "Recall_macro (%)": mean_sd_text(table1["Recall_macro (%)"], scale=1.0, decimals=2),
    "F1_macro (%)": mean_sd_text(table1["F1_macro (%)"], scale=1.0, decimals=2),
    "AUC_macro (%)": mean_sd_text(table1["AUC_macro (%)"], scale=1.0, decimals=2),
}
table1_with_summary = pd.concat([table1, pd.DataFrame([mean_row])], ignore_index=True)
save_df(table1_with_summary, "Table_1_Classification_metrics_each_fold")

# =========================================================
# TABLE 2: Per-class Precision, Recall, F1-score
# =========================================================
report = classification_report(
    pred_df["true_label"],
    pred_df["predicted_label"],
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

table2 = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
table2 = table2[table2["Class"].isin(CLASS_NAMES)].copy()
table2["Precision (%)"] = table2["precision"] * 100
table2["Recall (%)"] = table2["recall"] * 100
table2["F1-score (%)"] = table2["f1-score"] * 100
table2["Support"] = table2["support"].astype(int)
table2 = table2[["Class", "Precision (%)", "Recall (%)", "F1-score (%)", "Support"]]
save_df(table2, "Table_2_Per_class_precision_recall_f1")

# =========================================================
# TABLE 3: Grad-CAM metrics for each fold
# =========================================================
table3 = (
    gradcam_valid.groupby("fold")[GRADCAM_METRIC_COLS]
    .agg(["mean", "std", "count"])
    .reset_index()
)

table3.columns = [
    "Fold",
    "IoU_mean", "IoU_std", "IoU_count",
    "Dice_mean", "Dice_std", "Dice_count",
    "CAM_inside_lesion_mean", "CAM_inside_lesion_std", "CAM_inside_lesion_count",
    "Pixelwise_correlation_mean", "Pixelwise_correlation_std", "Pixelwise_correlation_count",
]

table3_disp = table3.copy()
table3_disp["IoU"] = table3_disp.apply(lambda r: f"{r['IoU_mean']:.3f} ± {r['IoU_std']:.3f}", axis=1)
table3_disp["Dice"] = table3_disp.apply(lambda r: f"{r['Dice_mean']:.3f} ± {r['Dice_std']:.3f}", axis=1)
table3_disp["CAM inside lesion (%)"] = table3_disp.apply(lambda r: f"{r['CAM_inside_lesion_mean']*100:.2f} ± {r['CAM_inside_lesion_std']*100:.2f}", axis=1)
table3_disp["Pixel-wise correlation"] = table3_disp.apply(lambda r: f"{r['Pixelwise_correlation_mean']:.3f} ± {r['Pixelwise_correlation_std']:.3f}", axis=1)
table3_disp = table3_disp[["Fold", "IoU", "Dice", "CAM inside lesion (%)", "Pixel-wise correlation"]]
save_df(table3_disp, "Table_3_GradCAM_metrics_each_fold")

# =========================================================
# TABLE 4: Grad-CAM metrics for each lesion class
# =========================================================
table4 = (
    gradcam_valid.groupby("true_class_name")[GRADCAM_METRIC_COLS]
    .agg(["mean", "std", "count"])
    .reset_index()
)

table4.columns = [
    "Class",
    "IoU_mean", "IoU_std", "IoU_count",
    "Dice_mean", "Dice_std", "Dice_count",
    "CAM_inside_lesion_mean", "CAM_inside_lesion_std", "CAM_inside_lesion_count",
    "Pixelwise_correlation_mean", "Pixelwise_correlation_std", "Pixelwise_correlation_count",
]

table4["Class"] = pd.Categorical(table4["Class"], categories=CLASS_NAMES, ordered=True)
table4 = table4.sort_values("Class")

table4_disp = table4.copy()
table4_disp["IoU"] = table4_disp.apply(lambda r: f"{r['IoU_mean']:.3f} ± {r['IoU_std']:.3f}", axis=1)
table4_disp["Dice"] = table4_disp.apply(lambda r: f"{r['Dice_mean']:.3f} ± {r['Dice_std']:.3f}", axis=1)
table4_disp["CAM inside lesion (%)"] = table4_disp.apply(lambda r: f"{r['CAM_inside_lesion_mean']*100:.2f} ± {r['CAM_inside_lesion_std']*100:.2f}", axis=1)
table4_disp["Pixel-wise correlation"] = table4_disp.apply(lambda r: f"{r['Pixelwise_correlation_mean']:.3f} ± {r['Pixelwise_correlation_std']:.3f}", axis=1)
table4_disp = table4_disp[["Class", "IoU", "Dice", "CAM inside lesion (%)", "Pixel-wise correlation"]]
save_df(table4_disp, "Table_4_GradCAM_metrics_each_class")

# =========================================================
# TABLE 5: Overall Mean ± SD summary
# =========================================================
overall_summary = pd.DataFrame({
    "Metric": [
        "Accuracy_macro (%)",
        "Precision_macro (%)",
        "Recall_macro (%)",
        "F1_macro (%)",
        "AUC_macro (%)",
        "IoU",
        "Dice",
        "CAM inside lesion (%)",
        "Pixel-wise correlation"
    ],
    "Value": [
        mean_sd_text(table1["Accuracy (%)"]),
        mean_sd_text(table1["Precision_macro (%)"]),
        mean_sd_text(table1["Recall_macro (%)"]),
        mean_sd_text(table1["F1_macro (%)"]),
        mean_sd_text(table1["AUC_macro (%)"]),
        mean_sd_text(gradcam_valid["iou"]),
        mean_sd_text(gradcam_valid["dice"]),
        mean_sd_text(gradcam_valid["percent_cam_inside_lesion"], scale=100.0),
        mean_sd_text(gradcam_valid["pixelwise_correlation"]),
    ]
})
save_df(overall_summary, "Table_5_Overall_mean_sd_summary")

# =========================================================
# FIGURE 1: 5-fold ROC curves (7-class OVR)
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

folds = sorted(pred_df["fold"].unique())
macro_auc_by_fold = []

for i, fold in enumerate(folds):
    ax = axes[i]
    d = pred_df[pred_df["fold"] == fold]
    y_true = d["true_label"].values
    y_prob = d[PROB_COLS].values

    roc_info, all_fpr, mean_tpr, macro_auc = multiclass_roc_data(y_true, y_prob, CLASS_NAMES)
    macro_auc_by_fold.append({"fold": fold, "macro_auc": macro_auc})

    for cls in CLASS_NAMES:
        if cls in roc_info:
            ax.plot(roc_info[cls]["fpr"], roc_info[cls]["tpr"], lw=1.4, label=f"{cls.upper()} (AUC={roc_info[cls]['auc']:.3f})")

    ax.plot(all_fpr, mean_tpr, "--", color="black", lw=2, label=f"Macro-average (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title(f"Fold {fold}", weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

# mean ROC in last panel
ax = axes[-1]
all_interp = []
base_fpr = np.linspace(0, 1, 300)
for fold in folds:
    d = pred_df[pred_df["fold"] == fold]
    y_true = d["true_label"].values
    y_prob = d[PROB_COLS].values
    _, all_fpr, mean_tpr, macro_auc = multiclass_roc_data(y_true, y_prob, CLASS_NAMES)
    interp_tpr = np.interp(base_fpr, all_fpr, mean_tpr)
    interp_tpr[0] = 0.0
    all_interp.append(interp_tpr)
    ax.plot(base_fpr, interp_tpr, alpha=0.25, lw=1)

mean_tpr = np.mean(all_interp, axis=0)
std_tpr = np.std(all_interp, axis=0)
mean_auc = auc(base_fpr, mean_tpr)
ax.plot(base_fpr, mean_tpr, color="navy", lw=2.5, label=f"Mean macro ROC (AUC={mean_auc:.3f})")
ax.fill_between(base_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color="navy", alpha=0.15, label="±1 SD")
ax.plot([0, 1], [0, 1], "k:", lw=1)
ax.set_title("Mean ROC across folds", weight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

plt.suptitle("Figure 1. 5-fold ROC curves (7-class One-vs-Rest)", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_DIR / "Figure_1_5fold_ROC_curves.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 2: Confusion matrices (Fold 1–5 + Mean)
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
cms = []

for i, fold in enumerate(folds):
    d = pred_df[pred_df["fold"] == fold]
    cm = confusion_matrix(d["true_label"], d["predicted_label"], labels=list(range(NUM_CLASSES)))
    cms.append(cm)
    plot_conf_mat(cm, f"Fold {fold}", ax=axes[i])

mean_cm = np.mean(np.stack(cms, axis=0), axis=0)
plot_conf_mat(mean_cm, "Mean confusion matrix", ax=axes[-1])

plt.suptitle("Figure 2. Confusion matrices (Fold 1–5 + Mean)", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / "Figure_2_Confusion_matrices.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 3: Grad-CAM localization metrics across folds
# =========================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

metric_titles = {
    "iou": "IoU",
    "dice": "Dice coefficient",
    "percent_cam_inside_lesion": "CAM inside lesion (%)",
    "pixelwise_correlation": "Pixel-wise correlation"
}

for ax, metric in zip(axes, GRADCAM_METRIC_COLS):
    d = gradcam_valid[["fold", metric]].dropna().copy()
    if metric == "percent_cam_inside_lesion":
        d[metric] = d[metric] * 100.0

    sns.boxplot(data=d, x="fold", y=metric, color="skyblue", ax=ax)
    means = d.groupby("fold")[metric].mean()
    for i, fold in enumerate(sorted(d["fold"].unique())):
        ax.text(i, means.loc[fold], f"{means.loc[fold]:.2f}", ha="center", va="bottom", fontsize=8, color="darkblue")
    ax.set_title(metric_titles[metric], weight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")

plt.suptitle("Figure 3. Grad-CAM localization metrics across folds", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "Figure_3_GradCAM_metrics_across_folds.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 4: Mean ± SD performance
# =========================================================
classification_summary = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)", "AUC (macro)"],
    "Mean": [
        table1["Accuracy (%)"].mean(),
        table1["Precision_macro (%)"].mean(),
        table1["Recall_macro (%)"].mean(),
        table1["F1_macro (%)"].mean(),
        table1["AUC_macro (%)"].mean(),
    ],
    "SD": [
        table1["Accuracy (%)"].std(ddof=1),
        table1["Precision_macro (%)"].std(ddof=1),
        table1["Recall_macro (%)"].std(ddof=1),
        table1["F1_macro (%)"].std(ddof=1),
        table1["AUC_macro (%)"].std(ddof=1),
    ]
})

gradcam_summary = pd.DataFrame({
    "Metric": ["IoU", "Dice", "CAM inside lesion (%)", "Pixel-wise correlation"],
    "Mean": [
        gradcam_valid["iou"].mean() * 100,
        gradcam_valid["dice"].mean() * 100,
        gradcam_valid["percent_cam_inside_lesion"].mean() * 100,
        gradcam_valid["pixelwise_correlation"].mean() * 100,
    ],
    "SD": [
        gradcam_valid["iou"].std(ddof=1) * 100,
        gradcam_valid["dice"].std(ddof=1) * 100,
        gradcam_valid["percent_cam_inside_lesion"].std(ddof=1) * 100,
        gradcam_valid["pixelwise_correlation"].std(ddof=1) * 100,
    ]
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].bar(classification_summary["Metric"], classification_summary["Mean"], yerr=classification_summary["SD"], color="royalblue", alpha=0.85, capsize=5)
axes[0].set_title("Classification performance", weight="bold")
axes[0].set_ylabel("Score (%)")
axes[0].tick_params(axis="x", rotation=20)
for i, row in classification_summary.iterrows():
    axes[0].text(i, row["Mean"] + row["SD"] + 0.2, f"{row['Mean']:.2f}\n±{row['SD']:.2f}", ha="center", fontsize=8)

axes[1].bar(gradcam_summary["Metric"], gradcam_summary["Mean"], yerr=gradcam_summary["SD"], color="forestgreen", alpha=0.85, capsize=5)
axes[1].set_title("Grad-CAM localization performance", weight="bold")
axes[1].set_ylabel("Score (%)")
axes[1].tick_params(axis="x", rotation=20)
for i, row in gradcam_summary.iterrows():
    axes[1].text(i, row["Mean"] + row["SD"] + 0.2, f"{row['Mean']:.2f}\n±{row['SD']:.2f}", ha="center", fontsize=8)

plt.suptitle("Figure 4. Mean ± SD performance", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "Figure_4_Mean_SD_performance.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 5: Dice vs CAM inside lesion correlation
# =========================================================
corr_df = gradcam_valid[["dice", "percent_cam_inside_lesion"]].dropna().copy()
corr_df["percent_cam_inside_lesion"] *= 100

r = corr_df["dice"].corr(corr_df["percent_cam_inside_lesion"] / 100.0)

fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(
    data=corr_df,
    x="percent_cam_inside_lesion",
    y="dice",
    scatter_kws={"s": 18, "alpha": 0.65, "color": "royalblue"},
    line_kws={"color": "crimson", "lw": 2},
    ax=ax
)
ax.set_title("Figure 5. Dice vs CAM inside lesion correlation", weight="bold")
ax.set_xlabel("CAM inside lesion (%)")
ax.set_ylabel("Dice coefficient")
ax.text(0.03, 0.97, f"Pearson r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Figure_5_Dice_vs_CAMinside_correlation.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 6: Classification performance across folds
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(table1["Fold"], table1["Accuracy (%)"], marker="o", lw=2, label="Accuracy")
ax.plot(table1["Fold"], table1["Precision_macro (%)"], marker="s", lw=2, label="Precision (macro)")
ax.plot(table1["Fold"], table1["Recall_macro (%)"], marker="^", lw=2, label="Recall (macro)")
ax.plot(table1["Fold"], table1["F1_macro (%)"], marker="D", lw=2, label="F1-score (macro)")
ax.plot(table1["Fold"], table1["AUC_macro (%)"], marker="*", lw=2, label="AUC (macro)")
ax.set_title("Figure 6. Classification performance across the five folds", weight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("Score (%)")
ax.set_xticks(table1["Fold"])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Figure_6_Classification_performance_across_folds.png", bbox_inches="tight")
plt.close()

# =========================================================
# FIGURE 7: Per-class Grad-CAM localization metrics
# =========================================================
per_class = (
    gradcam_valid.groupby("true_class_name")[GRADCAM_METRIC_COLS]
    .mean()
    .reset_index()
)
per_class["true_class_name"] = pd.Categorical(per_class["true_class_name"], categories=CLASS_NAMES, ordered=True)
per_class = per_class.sort_values("true_class_name")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

plot_specs = [
    ("iou", "IoU", "steelblue"),
    ("dice", "Dice coefficient", "forestgreen"),
    ("percent_cam_inside_lesion", "CAM inside lesion (%)", "purple"),
    ("pixelwise_correlation", "Pixel-wise correlation", "orange"),
]

for ax, (metric, title, color) in zip(axes, plot_specs):
    vals = per_class[metric].copy()
    if metric == "percent_cam_inside_lesion":
        vals = vals * 100
    ax.bar(per_class["true_class_name"].str.upper(), vals, color=color, alpha=0.85)
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.5 if metric == "percent_cam_inside_lesion" else 0.01), f"{v:.2f}", ha="center", fontsize=8)

plt.suptitle("Figure 7. Per-class Grad-CAM localization metrics", fontsize=15, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "Figure_7_Per_class_GradCAM_metrics.png", bbox_inches="tight")
plt.close()

# =========================================================
# OPTIONAL: save extra detailed summaries
# =========================================================
macro_auc_by_fold_df = pd.DataFrame(macro_auc_by_fold)
save_df(macro_auc_by_fold_df, "extra_macro_auc_by_fold")

per_fold_confusion_flat = []
for fold in folds:
    d = pred_df[pred_df["fold"] == fold]
    cm = confusion_matrix(d["true_label"], d["predicted_label"], labels=list(range(NUM_CLASSES)))
    for i, true_cls in enumerate(CLASS_NAMES):
        for j, pred_cls in enumerate(CLASS_NAMES):
            per_fold_confusion_flat.append({
                "fold": fold,
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "count": int(cm[i, j])
            })
save_df(pd.DataFrame(per_fold_confusion_flat), "extra_confusion_matrix_counts_long")

print("Done. All figures and tables saved in:", OUTPUT_DIR.resolve())


