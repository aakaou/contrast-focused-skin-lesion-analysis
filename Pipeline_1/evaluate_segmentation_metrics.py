# ============================================================
# HAM10000 Segmentation Metrics Evaluation
# ============================================================
# This script compares predicted segmentation masks with
# ground-truth masks and computes several metrics:
#   - IoU (Intersection over Union)
#   - Dice coefficient
#   - Accuracy
#   - Jaccard Index
#   - Sensitivity (Recall)
#
# Results are saved to a CSV file for all processed images.
# ============================================================

# ----------- 1. Import required libraries -----------

import cv2                     # OpenCV for image loading and resizing
import numpy as np             # Numerical operations (arrays, logical operations)
import os                      # File system operations
import csv                     # Writing results to CSV file
import re                      # Regular expressions (used to extract image numbers)


# ----------- 2. Helper function: extract numeric part from filename -----------

def extract_number(filename):
    """
    Extract numeric ID from filenames like:
        ISIC_0024306.jpg
    This allows proper numeric sorting of images.
    """

    # Search pattern "ISIC_" followed by digits
    m = re.search(r'ISIC_(\d+)', filename)

    # If match found → return number as integer
    # If not found → return infinity so it goes to the end of sorting
    return int(m.group(1)) if m else float('inf')


# ----------- 3. Load segmentation mask -----------

def load_mask(path, threshold=127):
    """
    Load a segmentation mask and convert it to binary.

    Parameters
    ----------
    path : str
        Path to the mask image
    threshold : int
        Threshold used to binarize the grayscale mask

    Returns
    -------
    numpy array
        Binary mask (0 or 1)
    """

    # Read image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # If image cannot be loaded → raise error
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")

    # Convert grayscale mask to binary mask
    # Pixels > threshold become 1, others become 0
    return (img > threshold).astype(np.uint8)


# ----------- 4. Compute segmentation metrics -----------

def compute_metrics(pred, gt):
    """
    Compute evaluation metrics between predicted mask and ground truth mask.

    Metrics:
        IoU (Intersection over Union)
        Dice coefficient
        Accuracy
        Jaccard index
        Sensitivity (Recall)

    Parameters
    ----------
    pred : numpy array
        Predicted binary mask
    gt : numpy array
        Ground truth binary mask
    """

    # Convert masks to boolean and flatten to 1D arrays
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()

    # True Positive: predicted lesion AND ground truth lesion
    tp = np.logical_and(pred, gt).sum()

    # True Negative: predicted background AND ground truth background
    tn = np.logical_and(~pred, ~gt).sum()

    # False Positive: predicted lesion but ground truth background
    fp = np.logical_and(pred, ~gt).sum()

    # False Negative: predicted background but ground truth lesion
    fn = np.logical_and(~pred, gt).sum()

    # Intersection (common positive pixels)
    inter = tp

    # Union (all pixels belonging to lesion in either mask)
    union = tp + fp + fn

    # ---- Metric formulas ----

    # Intersection over Union
    iou = inter / union if union > 0 else 0.0

    # Dice coefficient
    dice = (2 * inter) / (2 * inter + fp + fn) if (2*inter + fp + fn) > 0 else 0.0

    # Pixel accuracy
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # Jaccard index (same as IoU)
    jacc = iou

    # Sensitivity (Recall)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Return all computed metrics
    return iou, dice, acc, jacc, sens


# ----------- 5. Main evaluation function -----------

def evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv):
    """
    Compare predicted masks with ground truth masks for ALL images.

    processed_folder : folder containing predicted masks
    seg_folder       : folder containing ground truth masks
    output_csv       : file where metrics will be saved
    """

    # Collect predicted mask filenames
    processed_files = [
        f for f in os.listdir(processed_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ]

    # Sort files by numeric ISIC ID
    processed_files.sort(key=extract_number)

    # Print information about dataset
    print("Processed folder:", processed_folder)
    print("Found", len(processed_files), "processed files")
    print("Processing ALL images...")

    # Create CSV file and write header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ISIC ID", "IoU", "Dice", "Accuracy", "Jaccard", "Sensitivity"])

    success_count = 0  # Counter of processed image pairs

    # Loop through ALL predicted masks
    for proc_fname in processed_files:

        # Full path to predicted mask
        proc_path = os.path.join(processed_folder, proc_fname)

        # Extract image ID without extension
        isic_id = os.path.splitext(proc_fname)[0]

        # Expected ground truth filename
        gt_fname = f"{isic_id}.jpg"

        # Full path to ground truth mask
        gt_path = os.path.join(seg_folder, gt_fname)

        # If ground truth does not exist → skip
        if not os.path.exists(gt_path):
            continue

        # Load predicted mask
        proc_mask = load_mask(proc_path, threshold=127)

        # Load ground truth mask
        gt_mask = load_mask(gt_path, threshold=127)

        # If mask sizes differ → resize predicted mask
        if proc_mask.shape != gt_mask.shape:
            proc_mask = cv2.resize(
                proc_mask,
                (gt_mask.shape[1], gt_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Compute metrics
        iou, dice, acc, jacc, sens = compute_metrics(proc_mask, gt_mask)

        # Append results to CSV
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([isic_id, iou, dice, acc, jacc, sens])

        # Update counter
        success_count += 1

        # Show progress every 100 images
        if success_count % 100 == 0:
            print(f"Processed {success_count} images...")

    # Print completion message
    print(f"✅ Finished! Successfully processed {success_count} image pairs")
    print("✅ Full metrics saved to:", output_csv)


# ----------- 6. Run evaluation on all images -----------

# Folder containing predicted segmentation masks
processed_folder = "/aakaou/HAM10000_processed"

# Folder containing ground truth segmentation masks
seg_folder = "/aakaou/HAM10000_segmented_p1"

# Output CSV file with metrics
output_csv = "/aakaou/seg_metrics_all_p1.csv"

# Run evaluation
evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv)
