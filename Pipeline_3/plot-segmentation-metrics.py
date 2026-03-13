# ============================================================
# SEGMENTATION METRICS VISUALIZATION SCRIPT
# ------------------------------------------------------------
# This script reads segmentation metrics from a CSV file
# and generates:
# 1. Distribution histograms
# 2. Correlation heatmap
# 3. Box plots
# 4. Pairwise scatter plots
# 5. Best / worst performers
# 6. Threshold analysis scatter plots
# ============================================================

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
sns.set_style("whitegrid")          # Seaborn style
plt.rcParams["figure.dpi"] = 120    # Figure resolution

# ===============================
# LOAD CSV FILE
# ===============================
csv_path = "/aakaou/seg_red_metrics_all.csv"
df = pd.read_csv(csv_path)

# Metrics to visualize
metrics = ['IoU', 'Dice', 'Accuracy', 'Jaccard', 'Sensitivity']

# Color palette for multiple plots
colors = sns.color_palette("Set2", len(metrics))


# =====================================================
# 1. DISTRIBUTION PLOTS (HISTOGRAMS)
# =====================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Flatten for easy indexing

for i, metric in enumerate(metrics):
    axes[i].hist(
        df[metric],           # Metric data
        bins=30,              # Number of bins
        color=colors[i],      # Set color
        edgecolor='black',    # Black edges for bars
        alpha=0.8             # Transparency
    )
    axes[i].set_title(f'{metric} Distribution', fontsize=13, weight='bold')
    axes[i].set_xlabel(metric)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()


# =====================================================
# 2. CORRELATION HEATMAP
# =====================================================
plt.figure(figsize=(10, 8))
sns.heatmap(
    df[metrics].corr(),    # Correlation matrix
    annot=True,            # Show numbers
    fmt=".2f",             # Number format
    cmap='RdBu_r',         # Diverging color map
    center=0,              # Center color scale at 0
    square=True,           # Square cells
    linewidths=0.8,        # Line between cells
    cbar_kws={'shrink': 0.85}  # Colorbar size
)
plt.title('Segmentation Metrics Correlation Matrix', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()


# =====================================================
# 3. BOX PLOTS
# =====================================================
plt.figure(figsize=(15, 6))
box = plt.boxplot(
    df[metrics],            # Data
    patch_artist=True,       # Enable color fill
    labels=metrics          # Metric labels
)

# Set box colors
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

plt.title('Segmentation Metrics – Box Plot', fontsize=14, weight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 4. PAIRWISE SCATTER PLOT MATRIX
# =====================================================
sns.pairplot(
    df[metrics],             # Data
    diag_kind='kde',         # KDE on diagonal
    corner=True,             # Only lower triangle
    plot_kws={'alpha': 0.6, 's': 35}  # Marker options
)
plt.suptitle(
    'Segmentation Metrics Pairwise Relationships',
    y=1.02,
    fontsize=14,
    weight='bold'
)
plt.show()


# =====================================================
# 5. BEST / WORST PERFORMERS
# =====================================================
print("\n🏆 TOP 5 BEST IoU:")
display(df.nlargest(5, 'IoU')[['ISIC ID', 'IoU', 'Dice', 'Accuracy']])

print("\n😞 TOP 5 WORST IoU:")
display(df.nsmallest(5, 'IoU')[['ISIC ID', 'IoU', 'Dice', 'Accuracy']])


# =====================================================
# 6. THRESHOLD ANALYSIS (SCATTER PLOTS)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# IoU vs Dice
axes[0].scatter(
    df['IoU'],
    df['Dice'],
    c=df['Dice'],
    cmap='plasma',
    alpha=0.7,
    edgecolor='black',
    s=50
)
axes[0].plot([0, 1], [0, 1], '--', color='gray', lw=2)
axes[0].set_xlabel('IoU')
axes[0].set_ylabel('Dice')
axes[0].set_title('IoU vs Dice Coefficient', weight='bold')
axes[0].grid(True, alpha=0.3)

# Accuracy vs Sensitivity
sc = axes[1].scatter(
    df['Accuracy'],
    df['Sensitivity'],
    c=df['IoU'],
    cmap='viridis',
    alpha=0.7,
    edgecolor='black',
    s=50
)
axes[1].set_xlabel('Accuracy')
axes[1].set_ylabel('Sensitivity')
axes[1].set_title('Accuracy vs Sensitivity (colored by IoU)', weight='bold')
plt.colorbar(sc, ax=axes[1], label='IoU')

plt.tight_layout()
plt.show()

print("\n✅ Multi-color visualization complete!")
