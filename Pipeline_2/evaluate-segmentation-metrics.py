"""
Title: evaluate_segmentation_metrics.py
Description: 
This script evaluates segmentation quality of processed skin lesion images against ground-truth masks. 
It computes five common metrics (IoU, Dice, Accuracy, Jaccard, Sensitivity) for each image and saves the results to a CSV file.
"""

import cv2       # OpenCV for image loading and resizing
import numpy as np  # Numerical operations
import os        # File system operations
import csv       # Writing CSV files
import re        # Regular expressions for extracting numbers from filenames

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_number(filename):
    """
    Extract numeric ID from filenames like 'ISIC_0024306.jpg'.
    Used to sort images numerically rather than alphabetically.
    """
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

def load_mask(path, threshold=127):
    """
    Load a grayscale mask image and binarize it.
    Pixels > threshold are set to 1, else 0.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return (img > threshold).astype(np.uint8)

def compute_metrics(pred, gt):
    """
    Compute segmentation metrics between prediction and ground-truth masks:
    IoU, Dice coefficient, Accuracy, Jaccard, Sensitivity.
    """
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()

    tp = np.logical_and(pred,  gt).sum()  # True positives
    tn = np.logical_and(~pred, ~gt).sum()  # True negatives
    fp = np.logical_and(pred,  ~gt).sum()  # False positives
    fn = np.logical_and(~pred, gt).sum()   # False negatives

    inter = tp
    union = tp + fp + fn

    iou   = inter / union if union > 0 else 0.0
    dice  = (2 * inter) / (2 * inter + fp + fn) if (2*inter + fp + fn) > 0 else 0.0
    acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    jacc  = iou  # Jaccard index same as IoU
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, acc, jacc, sens

# -----------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------
def evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv, max_images=5):
    """
    Evaluate segmentation metrics for each image pair:
    - processed image vs corresponding ground-truth mask.
    - Saves all metrics to CSV.
    """
    # List processed images
    processed_files = [
        f for f in os.listdir(processed_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ]
    processed_files.sort(key=extract_number)  # Sort numerically

    print("Processed folder:", processed_folder)
    print("Found", len(processed_files), "processed files")
    print("First 5 processed files:", processed_files[:5])

    # Write CSV header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ISIC ID", "IoU", "Dice", "Accuracy", "Jaccard", "Sensitivity"])

    # Loop through each image (limit to max_images for testing)
    for proc_fname in processed_files[:max_images]:
        proc_path = os.path.join(processed_folder, proc_fname)
        isic_id = os.path.splitext(proc_fname)[0]  # Remove extension

        gt_fname = f"{isic_id}.jpg"
        gt_path = os.path.join(seg_folder, gt_fname)

        print(f"Looking for GT: {gt_path}")
        if not os.path.exists(gt_path):
            print(f"❌ Missing GT: {gt_path}")
            continue

        # Load predicted mask and ground-truth mask
        proc_mask = load_mask(proc_path, threshold=127)
        gt_mask = load_mask(gt_path, threshold=127)

        # Resize if dimensions mismatch
        if proc_mask.shape != gt_mask.shape:
            proc_mask = cv2.resize(proc_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Compute all metrics
        iou, dice, acc, jacc, sens = compute_metrics(proc_mask, gt_mask)

        # Append results to CSV
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([isic_id, iou, dice, acc, jacc, sens])

        # Print metrics to console
        print(f"{isic_id}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}, Jacc={jacc:.4f}, Sens={sens:.4f}")

    print("✅ Saved to:", output_csv)

# -----------------------------
# RUN SCRIPT
# -----------------------------
processed_folder = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_processed_images"
seg_folder = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images"
output_csv = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/segmentation_metrics_tested.csv"

evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv)

"""This script evaluate_segmentation_metrics.py compares processed skin lesion images with their ground-truth masks. For each image pair, 
it computes five metrics—IoU, Dice coefficient, Accuracy, Jaccard index, and Sensitivity—after optionally resizing masks to match. 
The results are saved to a CSV file, while also printing metrics to the console. Files are sorted numerically by ISIC ID to maintain correct order, 
and a threshold is applied to binarize masks. This allows easy evaluation of segmentation performance on small or large batches of images."""
