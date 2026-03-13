# ============================================================
# SEGMENTATION METRICS EVALUATION SCRIPT
# ------------------------------------------------------------
# This script compares predicted segmentation masks with
# ground truth masks and computes segmentation metrics:
#
# 1. IoU (Intersection over Union)
# 2. Dice Coefficient
# 3. Accuracy
# 4. Jaccard Index
# 5. Sensitivity (Recall)
#
# Results are saved to a CSV file for further analysis.
# ============================================================


# ===============================
# IMPORT LIBRARIES
# ===============================

import cv2                 # OpenCV library for image processing
import numpy as np         # Numerical computations with arrays
import os                  # File and directory operations
import csv                 # Writing results into CSV files
import re                  # Regular expressions for filename parsing


# ============================================================
# FUNCTION: extract_number
# ------------------------------------------------------------
# Extracts the numeric ID from filenames such as:
# ISIC_0024306.jpg → 24306
#
# This allows correct numeric sorting of images.
# ============================================================

def extract_number(filename):
    """Extract numeric part from filename like ISIC_0024306.jpg."""

    # Use regex to search for the pattern ISIC_XXXXXXX
    m = re.search(r'ISIC_(\d+)', filename)

    # If found → return numeric value
    if m:
        return int(m.group(1))

    # If not found → return infinity so it goes to the end
    return float('inf')


# ============================================================
# FUNCTION: load_mask
# ------------------------------------------------------------
# Loads an image mask and converts it into a binary mask.
#
# threshold:
# Pixel values above threshold = 1 (lesion)
# Pixel values below threshold = 0 (background)
# ============================================================

def load_mask(path, threshold=127):

    # Read image in grayscale mode
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Check if image exists
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # Convert grayscale image into binary mask
    # True = lesion region
    mask = (img > threshold).astype(np.uint8)

    return mask


# ============================================================
# FUNCTION: compute_metrics
# ------------------------------------------------------------
# Computes segmentation evaluation metrics.
#
# Input:
# pred → predicted mask
# gt   → ground truth mask
#
# Output:
# IoU, Dice, Accuracy, Jaccard, Sensitivity
# ============================================================

def compute_metrics(pred, gt):

    # Convert masks to boolean arrays
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()

    # -------------------------
    # Confusion matrix values
    # -------------------------

    # True Positive → correctly predicted lesion
    tp = np.logical_and(pred, gt).sum()

    # True Negative → correctly predicted background
    tn = np.logical_and(~pred, ~gt).sum()

    # False Positive → predicted lesion but actually background
    fp = np.logical_and(pred, ~gt).sum()

    # False Negative → predicted background but actually lesion
    fn = np.logical_and(~pred, gt).sum()

    # -------------------------
    # Metric calculations
    # -------------------------

    # Intersection = TP
    inter = tp

    # Union = TP + FP + FN
    union = tp + fp + fn

    # IoU (Intersection over Union)
    iou = inter / union if union > 0 else 0.0

    # Dice coefficient
    dice = (2 * inter) / (2 * inter + fp + fn) if (2 * inter + fp + fn) > 0 else 0.0

    # Pixel accuracy
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Jaccard index (same as IoU)
    jacc = iou

    # Sensitivity (Recall)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, acc, jacc, sens


# ============================================================
# FUNCTION: evaluate_segmentation_metrics
# ------------------------------------------------------------
# Main function that:
#
# 1. Loads processed images
# 2. Matches them with ground truth masks
# 3. Computes segmentation metrics
# 4. Saves results to CSV file
#
# No limit on number of images → processes ALL images.
# ============================================================

def evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv):

    # --------------------------------------------------------
    # Step 1: Get list of processed images
    # --------------------------------------------------------

    processed_files = [
        f for f in os.listdir(processed_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ]

    # Sort images numerically using ISIC ID
    processed_files.sort(key=extract_number)

    print("Processed folder:", processed_folder)
    print("Found", len(processed_files), "processed files")
    print("Processing ALL images...")


    # --------------------------------------------------------
    # Step 2: Create CSV file and write header
    # --------------------------------------------------------

    with open(output_csv, "w", newline="") as f:

        writer = csv.writer(f)

        # Column names
        writer.writerow([
            "ISIC ID",
            "IoU",
            "Dice",
            "Accuracy",
            "Jaccard",
            "Sensitivity"
        ])


    # --------------------------------------------------------
    # Step 3: Process images one by one
    # --------------------------------------------------------

    success_count = 0

    for proc_fname in processed_files:

        # Full path to processed image
        proc_path = os.path.join(processed_folder, proc_fname)

        # Extract image ID (without extension)
        isic_id = os.path.splitext(proc_fname)[0]

        # Ground truth filename
        gt_fname = f"{isic_id}.jpg"

        # Full path to GT mask
        gt_path = os.path.join(seg_folder, gt_fname)

        # Skip if ground truth mask not found
        if not os.path.exists(gt_path):
            continue


        # ----------------------------------------------------
        # Step 4: Load predicted and GT masks
        # ----------------------------------------------------

        proc_mask = load_mask(proc_path, threshold=127)
        gt_mask   = load_mask(gt_path, threshold=127)


        # ----------------------------------------------------
        # Step 5: Resize predicted mask if needed
        # ----------------------------------------------------

        if proc_mask.shape != gt_mask.shape:

            proc_mask = cv2.resize(
                proc_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )


        # ----------------------------------------------------
        # Step 6: Compute metrics
        # ----------------------------------------------------

        iou, dice, acc, jacc, sens = compute_metrics(proc_mask, gt_mask)


        # ----------------------------------------------------
        # Step 7: Save results to CSV
        # ----------------------------------------------------

        with open(output_csv, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                isic_id,
                iou,
                dice,
                acc,
                jacc,
                sens
            ])


        # Increase counter
        success_count += 1


        # Print progress every 100 images
        if success_count % 100 == 0:
            print(f"Processed {success_count} images...")


    # --------------------------------------------------------
    # Step 8: Final summary
    # --------------------------------------------------------

    print(f"✅ Finished! Successfully processed {success_count} image pairs")

    print("✅ Full metrics saved to:", output_csv)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Folder containing processed predicted masks
    processed_folder = "/aakaou/HAM10000_processed_images"

    # Folder containing ground truth segmentation masks
    seg_folder = "/aakaou/red_sonar_segmented_images"

    # Output CSV file for metrics
    output_csv = "/aakaou/seg_red_metrics_all.csv"

    # Run evaluation
    evaluate_segmentation_metrics(
        processed_folder,
        seg_folder,
        output_csv
    )
