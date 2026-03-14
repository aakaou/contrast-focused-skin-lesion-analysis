# ===============================
# plot_segmentation_metrics.py
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)

# ===============================
# Pipelines CSV paths
# ===============================
pipelines_csv = [
    {"name": "Pipeline 1", "csv": r"/aakaou/pipeline1_seg_metrics.csv"},
    {"name": "Pipeline 2", "csv": r"/aakaou/pipeline2_seg_metrics.csv"},
    {"name": "Pipeline 3", "csv": r"/aakaou/pipeline3_seg_metrics.csv"},
    {"name": "Pipeline 4", "csv": r"/aakaou/pipeline4_seg_metrics.csv"},
]

# Function for IoU pie chart labels
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f"{val}\n({pct:.1f}%)"
    return my_autopct

# ===============================
# Plot metrics for a single pipeline
# ===============================
def plot_pipeline_metrics(df, pipeline_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Segmentation Metrics - {pipeline_name}", fontsize=16, fontweight='bold')

    # --------- FIG 1: IoU Donut Pie ---------
    iou_bins = [0, 0.5, 0.75, 1.0]
    iou_labels = ["Poor (<0.5)", "Good (0.5-0.75)", "Excellent (>0.75)"]
    df["IoU Range"] = pd.cut(df["IoU"], bins=iou_bins, labels=iou_labels)
    iou_counts = df["IoU Range"].value_counts().sort_index()
    axes[0, 0].pie(
        iou_counts,
        labels=iou_counts.index,
        autopct=make_autopct(iou_counts),
        startangle=90,
        colors=["gold", "orange", "green"],
        wedgeprops=dict(width=0.4),
        textprops={'weight':'bold'}
    )
    axes[0, 0].set_title("IoU Performance Distribution", fontweight='bold')

    # --------- FIG 2: Accuracy Bar ---------
    acc_bins = [0, 0.25, 0.5, 0.75, 1.0]
    acc_labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]
    df["Accuracy Range"] = pd.cut(df["Accuracy"], bins=acc_bins, labels=acc_labels)
    acc_counts = df["Accuracy Range"].value_counts(normalize=True).sort_index() * 100

    bars = axes[0, 1].bar(
        acc_counts.index,
        acc_counts.values,
        color=["red", "orange", "blue", "green"]
    )
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    avg_acc = df["Accuracy"].mean() * 100
    axes[0, 1].axhline(avg_acc, color="red", linestyle="--", label=f"Avg Accuracy ({avg_acc:.2f}%)")
    axes[0, 1].set_ylabel("Percentage of Images (%)", fontweight='bold')
    axes[0, 1].set_title("Distribution of Image Accuracy Ranges", fontweight='bold')
    axes[0, 1].legend(prop={'weight':'bold'})

    # --------- FIG 3: Accuracy vs Dice KDE ---------
    sns.kdeplot(df["Accuracy"], ax=axes[1, 0], fill=True, label="Accuracy")
    sns.kdeplot(df["Dice"], ax=axes[1, 0], fill=True, label="Dice")
    axes[1, 0].set_title("Accuracy & Dice", fontweight='bold')
    axes[1, 0].set_xlabel("Normalized Metric Value", fontweight='bold')
    axes[1, 0].legend(prop={'weight':'bold'})

    # --------- FIG 4: Jaccard vs Sensitivity KDE ---------
    sns.kdeplot(df["Jaccard Index"], ax=axes[1, 1], fill=True, label="Jaccard Index")
    sns.kdeplot(df["Sensitivity"], ax=axes[1, 1], fill=True, label="Sensitivity")
    axes[1, 1].set_title("Jaccard Index & Sensitivity", fontweight='bold')
    axes[1, 1].set_xlabel("Normalized Metric Value", fontweight='bold')
    axes[1, 1].legend(prop={'weight':'bold'})

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for figure title
    plt.show()

# ===============================
# Run for all pipelines
# ===============================
if __name__ == "__main__":
    for pipeline in pipelines_csv:
        logging.info(f"📊 Plotting metrics for {pipeline['name']}")
        df = pd.read_csv(pipeline["csv"])
        plot_pipeline_metrics(df, pipeline["name"])
