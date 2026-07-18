import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)


# ===============================
# AUTOPCT FUNCTION
# ===============================
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f"{val}\n({pct:.1f}%)"
    return my_autopct


# ===============================
# PLOT FUNCTION
# ===============================
def plot_pipeline_metrics(df, pipeline_name, save_path=None):

    df = df.copy()  # avoid modifying original

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Segmentation Metrics - {pipeline_name}",
                 fontsize=16, fontweight='bold')

    # =====================================
    # FIG 1: IoU DONUT
    # =====================================
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
        textprops={'weight': 'bold'}
    )

    axes[0, 0].set_title("IoU Performance Distribution", fontweight='bold')

    # =====================================
    # FIG 2: ACCURACY BAR
    # =====================================
    acc_bins = [0, 0.25, 0.5, 0.75, 1.0]
    acc_labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]

    df["Accuracy Range"] = pd.cut(df["Accuracy"], bins=acc_bins, labels=acc_labels)

    acc_counts = df["Accuracy Range"].value_counts(normalize=True).sort_index() * 100

    bars = axes[0, 1].bar(
        acc_counts.index,
        acc_counts.values
    )

    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    avg_acc = df["Accuracy"].mean() * 100

    axes[0, 1].axhline(
        avg_acc,
        linestyle="--",
        label=f"Avg Accuracy ({avg_acc:.2f}%)"
    )

    axes[0, 1].set_ylabel("Percentage (%)", fontweight='bold')
    axes[0, 1].set_title("Accuracy Distribution", fontweight='bold')
    axes[0, 1].legend()

    # =====================================
    # FIG 3: Accuracy vs Dice
    # =====================================
    if len(df) > 10:
        sns.kdeplot(df["Accuracy"], ax=axes[1, 0], fill=True, label="Accuracy")
        sns.kdeplot(df["Dice"], ax=axes[1, 0], fill=True, label="Dice")
    else:
        axes[1, 0].hist(df["Accuracy"], alpha=0.5, label="Accuracy")
        axes[1, 0].hist(df["Dice"], alpha=0.5, label="Dice")

    axes[1, 0].set_title("Accuracy vs Dice", fontweight='bold')
    axes[1, 0].legend()

    # =====================================
    # FIG 4: IoU vs Sensitivity
    # =====================================
    if len(df) > 10:
        sns.kdeplot(df["IoU"], ax=axes[1, 1], fill=True, label="IoU")
        sns.kdeplot(df["Sensitivity"], ax=axes[1, 1], fill=True, label="Sensitivity")
    else:
        axes[1, 1].hist(df["IoU"], alpha=0.5, label="IoU")
        axes[1, 1].hist(df["Sensitivity"], alpha=0.5, label="Sensitivity")

    axes[1, 1].set_title("IoU vs Sensitivity", fontweight='bold')
    axes[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"💾 Saved figure to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ===============================
# RUN MULTIPLE PIPELINES & DATASETS
# ===============================
if __name__ == "__main__":

    pipelines_csv = [
        # ISIC 2016–2020 block
        {
            "name": "ISIC2016_2020 - Pipeline1",
            "csv": "/path/to/pipeline1/segmentation_metrics.csv",
            "out_png": "/path/to/pipeline1/segmentation_metrics_plot.png"
        },
        {
            "name": "ISIC2016_2020 - Pipeline2",
            "csv": "/path/to/pipeline2/segmentation_metrics.csv",
            "out_png": "/path/to/pipeline2/segmentation_metrics_plot.png"
        },
        {
            "name": "ISIC2016_2020 - Pipeline3",
            "csv": "/path/to/pipeline3/segmentation_metrics.csv",
            "out_png": "/path/to/pipeline3/segmentation_metrics_plot.png"
        },
        {
            "name": "ISIC2016_2020 - Pipeline4",
            "csv": "/path/to/pipeline4/segmentation_metrics.csv",
            "out_png": "/path/to/pipeline4/segmentation_metrics_plot.png"
        },

        # HAM10000 block
        {
            "name": "HAM10000 - Pipeline1",
            "csv": "/path/to/ham10000/pipeline1/segmentation_metrics.csv",
            "out_png": "/path/to/ham10000/pipeline1/segmentation_metrics_plot.png"
        },
        {
            "name": "HAM10000 - Pipeline2",
            "csv": "/path/to/ham10000/pipeline2/segmentation_metrics.csv",
            "out_png": "/path/to/ham10000/pipeline2/segmentation_metrics_plot.png"
        },
        {
            "name": "HAM10000 - Pipeline3",
            "csv": "/path/to/ham10000/pipeline3/segmentation_metrics.csv",
            "out_png": "/path/to/ham10000/pipeline3/segmentation_metrics_plot.png"
        },
        {
            "name": "HAM10000 - Pipeline4",
            "csv": "/path/to/ham10000/pipeline4/segmentation_metrics.csv",
            "out_png": "/path/to/ham10000/pipeline4/segmentation_metrics_plot.png"
        },
    ]

    for pipeline in pipelines_csv:
        logging.info(f"📊 Plotting: {pipeline['name']}")

        try:
            df = pd.read_csv(pipeline["csv"])
        except FileNotFoundError:
            logging.warning(f"CSV not found: {pipeline['csv']}, skipping.")
            continue

        plot_pipeline_metrics(df, pipeline["name"], save_path=pipeline["out_png"])
