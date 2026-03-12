import pandas as pd                  # For data manipulation and CSV reading
import matplotlib.pyplot as plt      # For plotting
import seaborn as sns                # For advanced statistical plots
import numpy as np                   # For numerical operations

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("/aakaou/ham10000/pipeline4/segmentation_metrics.csv")
# Load segmentation metrics CSV into a DataFrame

# =========================
# FIGURE LAYOUT
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Create a 2x2 grid of subplots with a specified figure size

# =========================
# FIG 1  IoU PIE (DONUT) with bold numbers
# =========================
iou_bins = [0, 0.5, 0.75, 1.0]   # Define IoU bins
iou_labels = ["Poor (<0.5)", "Good (0.5-0.75)", "Excellent (>0.75)"]  # Labels for bins

df["IoU Range"] = pd.cut(df["IoU"], bins=iou_bins, labels=iou_labels)
# Categorize each IoU value into one of the defined ranges

iou_counts = df["IoU Range"].value_counts().sort_index()
# Count how many images fall into each IoU category, sorted by bin order

# Function to display both count and percentage on pie chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f"{val}\n({pct:.1f}%)"
    return my_autopct

axes[0, 0].pie(
    iou_counts,                  # Values for pie slices
    labels=iou_counts.index,     # Labels for each slice
    autopct=make_autopct(iou_counts),  # Display counts and percentages
    startangle=90,               # Start pie at top
    colors=["gold", "orange", "green"],  # Slice colors
    wedgeprops=dict(width=0.4),  # Create donut chart (width < 1)
    textprops={'weight':'bold'}  # Make labels bold
)
axes[0, 0].set_title("IoU Performance Distribution", fontweight='bold')
# Title for first subplot

# =========================
# FIG 2  ACCURACY RANGE BAR with percentages
# =========================
acc_bins = [0, 0.25, 0.5, 0.75, 1.0]           # Define accuracy bins
acc_labels = ["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0"]  # Labels

df["Accuracy Range"] = pd.cut(df["Accuracy"], bins=acc_bins, labels=acc_labels)
# Categorize each Accuracy value into a range

acc_counts = df["Accuracy Range"].value_counts(normalize=True).sort_index() * 100
# Compute percentage of images in each range

bars = axes[0, 1].bar(
    acc_counts.index,             # X-axis labels
    acc_counts.values,            # Heights of bars
    color=["red", "orange", "blue", "green"]  # Bar colors
)

# Add percentage labels above bars in bold
for bar in bars:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

avg_acc = df["Accuracy"].mean() * 100  # Average accuracy
axes[0, 1].axhline(avg_acc, color="red", linestyle="--", label=f"Avg Accuracy ({avg_acc:.2f}%)")
# Add horizontal line showing average accuracy

axes[0, 1].set_ylabel("Percentage of Images (%)", fontweight='bold')
axes[0, 1].set_title("Distribution of Image Accuracy Ranges", fontweight='bold')
axes[0, 1].legend(prop={'weight':'bold'})  # Bold legend

# =========================
# FIG 3  ACCURACY vs DICE
# =========================
sns.kdeplot(df["Accuracy"], ax=axes[1, 0], fill=True, label="Accuracy")
sns.kdeplot(df["Dice"], ax=axes[1, 0], fill=True, label="Dice")
# Plot kernel density estimate (smooth distribution) for Accuracy and Dice metrics

axes[1, 0].set_title("Accuracy & Dice", fontweight='bold')
axes[1, 0].set_xlabel("Normalized Metric Value", fontweight='bold')
axes[1, 0].legend(prop={'weight':'bold'})

# =========================
# FIG 4  JACCARD vs SENSITIVITY
# =========================
sns.kdeplot(df["Jaccard Index"], ax=axes[1, 1], fill=True, label="Jaccard Index")
sns.kdeplot(df["Sensitivity"], ax=axes[1, 1], fill=True, label="Sensitivity")
# Plot kernel density estimate for Jaccard Index and Sensitivity

axes[1, 1].set_title("Jaccard Index & Sensitivity", fontweight='bold')
axes[1, 1].set_xlabel("Normalized Metric Value", fontweight='bold')
axes[1, 1].legend(prop={'weight':'bold'})

# =========================
# FINAL TOUCH
# =========================
plt.tight_layout()  # Adjust spacing to prevent overlap
plt.show()          # Display all plots
