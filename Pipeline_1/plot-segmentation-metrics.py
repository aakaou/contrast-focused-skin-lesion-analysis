# ==========================================================
# Skin Lesion Segmentation Metrics Dashboard
# This script loads segmentation metrics from a CSV file
# and creates a visualization dashboard of model performance
# ==========================================================

# Import the pandas library for data manipulation and CSV reading
import pandas as pd

# Import matplotlib for creating plots and visualizations
import matplotlib.pyplot as plt

# Import seaborn for advanced statistical data visualization
import seaborn as sns

# Import numpy for numerical operations
import numpy as np


# ----------------------------------------------------------
# 1. LOAD THE CSV FILE CONTAINING SEGMENTATION METRICS
# ----------------------------------------------------------

# Define the path to the CSV file containing segmentation metrics
csv_path = "/aakaou/seg_metrics_all_p1.csv"

# Read the CSV file and store it in a pandas DataFrame
df = pd.read_csv(csv_path)


# ----------------------------------------------------------
# 2. CREATE THE MAIN FIGURE FOR THE DASHBOARD
# ----------------------------------------------------------

# Create a large figure that will contain multiple plots
fig = plt.figure(figsize=(20, 16))

# Add a global title to the dashboard
plt.suptitle(
    "🩹 Skin Lesion Segmentation Metrics Dashboard",
    fontsize=24,
    fontweight="bold",
    y=0.98
)


# ----------------------------------------------------------
# 3. CORRELATION HEATMAP (Top Left)
# ----------------------------------------------------------

# Create the first subplot in a 2x3 grid (position 1)
ax1 = plt.subplot(2, 3, 1)

# Compute the correlation matrix between segmentation metrics
corr_matrix = df[["IoU", "Dice", "Accuracy", "Jaccard", "Sensitivity"]].corr()

# Draw the heatmap showing correlation between metrics
sns.heatmap(
    corr_matrix,            # correlation matrix
    annot=True,             # show correlation values in each cell
    cmap="RdYlBu_r",        # color map (red to blue reversed)
    center=0,               # center color scale at 0
    square=True,            # square cells
    linewidths=1,           # line width between cells
    cbar_kws={"shrink":0.8},# shrink the color bar
    ax=ax1                  # draw plot in ax1
)

# Set the title of the heatmap
ax1.set_title("📊 Correlation Matrix", fontweight="bold", fontsize=14)


# ----------------------------------------------------------
# 4. METRICS DISTRIBUTION HISTOGRAMS (Top Middle)
# ----------------------------------------------------------

# Create the second subplot
ax2 = plt.subplot(2, 3, 2)

# Define the list of metrics to visualize
metrics = ["IoU", "Dice", "Accuracy", "Jaccard", "Sensitivity"]

# Generate colors using the viridis colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(metrics)))

# Loop through each metric and create a histogram
for i, metric in enumerate(metrics):

    # Plot histogram for each metric
    ax2.hist(
        df[metric],      # values of the metric
        bins=25,         # number of bins in histogram
        alpha=0.7,       # transparency
        label=metric,    # label for legend
        color=colors[i]  # color from colormap
    )

# Label x-axis
ax2.set_xlabel("Metric Value")

# Label y-axis
ax2.set_ylabel("Frequency")

# Set title
ax2.set_title("📈 Metrics Distributions", fontweight="bold", fontsize=14)

# Show legend
ax2.legend()

# Enable grid for better readability
ax2.grid(True, alpha=0.3)


# ----------------------------------------------------------
# 5. IoU vs Dice SCATTER PLOT (Top Right)
# ----------------------------------------------------------

# Create third subplot
ax3 = plt.subplot(2, 3, 3)

# Scatter plot comparing IoU and Dice scores
scatter = ax3.scatter(
    df["IoU"],              # x-axis values
    df["Dice"],             # y-axis values
    c=df["Accuracy"],       # color points by Accuracy
    s=30,                   # marker size
    cmap="coolwarm",        # color map
    alpha=0.7,              # transparency
    edgecolors="black",     # marker border color
    linewidth=0.5
)

# Plot a diagonal reference line (perfect IoU = Dice)
ax3.plot([0,1], [0,1], "k--", lw=2, alpha=0.8)

# Label axes
ax3.set_xlabel("IoU")
ax3.set_ylabel("Dice")

# Title of the plot
ax3.set_title("🔗 IoU vs Dice (colored by Accuracy)", fontweight="bold", fontsize=14)

# Enable grid
ax3.grid(True, alpha=0.3)

# Add color bar to explain Accuracy values
plt.colorbar(scatter, ax=ax3, shrink=0.8)


# ----------------------------------------------------------
# 6. BOX PLOT FOR METRICS COMPARISON (Bottom Left)
# ----------------------------------------------------------

# Create subplot
ax4 = plt.subplot(2, 3, 4)

# Draw boxplots for selected metrics
df[["IoU", "Dice", "Accuracy", "Sensitivity"]].boxplot(ax=ax4)

# Set plot title
ax4.set_title("📋 Metrics Comparison (Box Plot)", fontweight="bold", fontsize=14)

# Label y-axis
ax4.set_ylabel("Score")

# Enable grid
ax4.grid(True, alpha=0.3)


# ----------------------------------------------------------
# 7. BEST AND WORST IoU SCORES (Bottom Middle)
# ----------------------------------------------------------

# Create subplot
ax5 = plt.subplot(2, 3, 5)

# Select the top 5 images with highest IoU
top5 = df.nlargest(5, "IoU")["IoU"]

# Select the 5 images with lowest IoU
bottom5 = df.nsmallest(5, "IoU")["IoU"]

# Plot bar chart for best IoU values
ax5.bar(
    ["Top 1","Top 2","Top 3","Top 4","Top 5"],
    top5.values,
    color="green",
    alpha=0.8,
    label="Best IoU"
)

# Plot bar chart for worst IoU values
ax5.bar(
    ["Worst 1","Worst 2","Worst 3","Worst 4","Worst 5"],
    bottom5.values,
    color="red",
    alpha=0.8,
    label="Worst IoU"
)

# Label y-axis
ax5.set_ylabel("IoU Score")

# Title
ax5.set_title("🥇🏆 Best vs Worst IoU", fontweight="bold", fontsize=14)

# Show legend
ax5.legend()

# Enable grid
ax5.grid(True, alpha=0.3)


# ----------------------------------------------------------
# 8. ACCURACY vs SENSITIVITY SCATTER PLOT (Bottom Right)
# ----------------------------------------------------------

# Create subplot
ax6 = plt.subplot(2, 3, 6)

# Scatter plot comparing Accuracy and Sensitivity
scatter2 = ax6.scatter(
    df["Accuracy"],
    df["Sensitivity"],
    c=df["IoU"],         # color points by IoU
    s=40,
    cmap="plasma",
    alpha=0.7,
    edgecolors="white"
)

# Label axes
ax6.set_xlabel("Accuracy")
ax6.set_ylabel("Sensitivity")

# Set title
ax6.set_title(
    "⚖️ Accuracy vs Sensitivity\n(colored by IoU)",
    fontweight="bold",
    fontsize=14
)

# Enable grid
ax6.grid(True, alpha=0.3)

# Add colorbar for IoU values
plt.colorbar(scatter2, ax=ax6, shrink=0.8)


# ----------------------------------------------------------
# 9. FINALIZE LAYOUT AND SAVE FIGURE
# ----------------------------------------------------------

# Adjust layout to prevent overlapping plots
plt.tight_layout()

# Adjust space for main title
plt.subplots_adjust(top=0.93)

# Save the dashboard image
plt.savefig(
    "/aakaou/segmentation_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the dashboard
plt.show()


# ----------------------------------------------------------
# 10. PRINT SUMMARY STATISTICS
# ----------------------------------------------------------

# Confirm that the dashboard image has been saved
print("✅ Dashboard saved as 'segmentation_dashboard.png'")

# Print the number of processed images
print(f"📈 Processed {len(df)} images")

# Print the mean IoU score
print(f"🎯 Mean IoU: {df['IoU'].mean():.4f}")

# Print the mean Dice score
print(f"🎯 Mean Dice: {df['Dice'].mean():.4f}")
