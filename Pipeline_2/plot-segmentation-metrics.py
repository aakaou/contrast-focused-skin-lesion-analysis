"""
Title: segmentation-metrics-dashboard.py
Description:
This script creates a comprehensive dashboard for visualizing skin lesion
segmentation metrics. It reads a CSV file containing IoU, Dice, Accuracy,
Jaccard, and Sensitivity scores for all processed images, and produces:
1. Correlation heatmap
2. Histogram distributions
3. IoU vs Dice scatter (colored by Accuracy)
4. Box plots for comparison
5. Top vs Worst performers
6. Accuracy vs Sensitivity scatter (colored by IoU)
The final dashboard is saved as a high-resolution PNG and displayed interactively. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# 1. Load segmentation metrics CSV
# -----------------------------
csv_path = "/aakaou/ham10000/pipeline2/segmentation_metrics_all_images.csv"
df = pd.read_csv(csv_path)

# -----------------------------
# 2. Create Dashboard Figure
# -----------------------------
fig = plt.figure(figsize=(20, 16))
plt.suptitle('🩹 Skin Lesion Segmentation Metrics Dashboard', fontsize=24, fontweight='bold', y=0.98)

# -----------------------------
# Top Left: Correlation Heatmap
# -----------------------------
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(df[['IoU', 'Dice', 'Accuracy', 'Jaccard', 'Sensitivity']].corr(), 
            annot=True, cmap='RdYlBu_r', center=0, square=True, linewidths=1, 
            cbar_kws={'shrink': 0.8}, ax=ax1)
ax1.set_title('📊 Correlation Matrix', fontweight='normal', fontsize=14)

# -----------------------------
# Top Middle: Metrics Distributions
# -----------------------------
ax2 = plt.subplot(2, 3, 2)
metrics = ['IoU', 'Dice', 'Accuracy', 'Jaccard', 'Sensitivity']
colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))
for i, metric in enumerate(metrics):
    ax2.hist(df[metric], bins=25, alpha=0.7, label=metric, color=colors[i])
ax2.set_xlabel('Metric Value')
ax2.set_ylabel('Frequency')
ax2.set_title('📈 Metrics Distributions', fontweight='normal', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# -----------------------------
# Top Right: IoU vs Dice Scatter (colored by Accuracy)
# -----------------------------
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(df['IoU'], df['Dice'], c=df['Accuracy'], s=30, 
                      cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=0.5)
ax3.plot([0,1], [0,1], 'k--', lw=2, alpha=0.8)  # reference line y=x
ax3.set_xlabel('IoU')
ax3.set_ylabel('Dice')
ax3.set_title('🔗 IoU vs Dice (colored by Accuracy)', fontweight='normal', fontsize=14)
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, shrink=0.8)

# -----------------------------
# Bottom Left: Box Plot Comparison
# -----------------------------
ax4 = plt.subplot(2, 3, 4)
df[['IoU', 'Dice', 'Accuracy', 'Sensitivity']].boxplot(ax=ax4)
ax4.set_title('Metrics Comparison (Box Plot)', fontweight='normal', fontsize=14)
ax4.set_ylabel('Score')
ax4.grid(True, alpha=0.3)

# -----------------------------
# Bottom Middle: Top/Bottom Performers
# -----------------------------
ax5 = plt.subplot(2, 3, 5)
top5 = df.nlargest(5, 'IoU')['IoU']
bottom5 = df.nsmallest(5, 'IoU')['IoU']
ax5.bar(['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'], top5.values, 
        color='green', alpha=0.8, label='Best IoU')
ax5.bar(['Worst 1', 'Worst 2', 'Worst 3', 'Worst 4', 'Worst 5'], bottom5.values, 
        color='red', alpha=0.8, label='Worst IoU')
ax5.set_ylabel('IoU Score')
ax5.set_title('Best vs Worst IoU', fontweight='normal', fontsize=14)
ax5.legend()
ax5.grid(True, alpha=0.3)

# -----------------------------
# Bottom Right: Accuracy vs Sensitivity (colored by IoU)
# -----------------------------
ax6 = plt.subplot(2, 3, 6)
scatter2 = ax6.scatter(df['Accuracy'], df['Sensitivity'], c=df['IoU'], 
                       s=40, cmap='plasma', alpha=0.7, edgecolors='white')
ax6.set_xlabel('Accuracy')
ax6.set_ylabel('Sensitivity')
ax6.set_title('Accuracy vs Sensitivity (colored by IoU)', fontweight='normal', fontsize=14)
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax6, shrink=0.8)

# -----------------------------
# Final Layout Adjustments & Save
# -----------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.83)
plt.show()

# -----------------------------
# Print Summary Stats
# -----------------------------

print(f"📈 Processed {len(df)} images")
print(f"🎯 Mean IoU: {df['IoU'].mean():.4f}")
print(f"🎯 Mean Dice: {df['Dice'].mean():.4f}")
