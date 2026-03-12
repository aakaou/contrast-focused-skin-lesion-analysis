"""
Title: evaluate_segmentation_metrics.py
Description:
This script evaluates segmentation quality for ALL processed skin lesion images
against their ground-truth masks. It computes five metrics per image:
IoU, Dice coefficient, Accuracy, Jaccard index, and Sensitivity. Results are
saved to a CSV file. The script automatically matches processed images with
ground-truth masks based on their ISIC IDs and handles resizing when necessary.
"""

import cv2       # For reading and resizing mask images
import numpy as np  # For numerical operations and logical calculations
import os        # For filesystem operations
import csv       # For writing metrics to CSV
import re        # For extracting numeric IDs from filenames

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_number(filename):
    """Extract numeric ID from filenames like 'ISIC_0024306.jpg' for proper sorting."""
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

def load_mask(path, threshold=127):
    """
    Load a grayscale mask and binarize it using the given threshold.
    Pixels > threshold are 1, else 0.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return (img > threshold).astype(np.uint8)

def compute_metrics(pred, gt):
    """
    Compute segmentation metrics between prediction and ground-truth masks:
    - IoU (Intersection over Union)
    - Dice coefficient
    - Accuracy
    - Jaccard index (same as IoU)
    - Sensitivity (recall)
    """
    pred = pred.astype(bool).flatten()
    gt   = gt.astype(bool).flatten()

    tp = np.logical_and(pred,  gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred,  ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    inter = tp
    union = tp + fp + fn

    iou   = inter / union if union > 0 else 0.0
    dice  = (2 * inter) / (2 * inter + fp + fn) if (2*inter + fp + fn) > 0 else 0.0
    acc   = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    jacc  = iou
    sens  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return iou, dice, acc, jacc, sens

# -----------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------
def evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv):
    """
    Evaluate segmentation metrics for ALL images in the processed folder.
    Each processed image is matched with the ground-truth mask based on ISIC ID.
    Results are saved to a CSV file.
    """
    # List processed images
    processed_files = [
        f for f in os.listdir(processed_folder)
        if f.lower().endswith(('.png', '.jpg'))
    ]
    processed_files.sort(key=extract_number)

    print("Processed folder:", processed_folder)
    print("Found", len(processed_files), "processed files")
    print("Processing ALL images...")

    # Write CSV header
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ISIC ID", "IoU", "Dice", "Accuracy", "Jaccard", "Sensitivity"])

    success_count = 0
    for proc_fname in processed_files:  # Process all images
        proc_path = os.path.join(processed_folder, proc_fname)
        isic_id = os.path.splitext(proc_fname)[0]

        gt_fname = f"{isic_id}.jpg"
        gt_path = os.path.join(seg_folder, gt_fname)

        if not os.path.exists(gt_path):
            continue  # Skip if ground-truth is missing

        proc_mask = load_mask(proc_path, threshold=127)
        gt_mask = load_mask(gt_path, threshold=127)

        # Resize prediction mask to match GT if necessary
        if proc_mask.shape != gt_mask.shape:
            proc_mask = cv2.resize(proc_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Compute metrics
        iou, dice, acc, jacc, sens = compute_metrics(proc_mask, gt_mask)

        # Append metrics to CSV
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([isic_id, iou, dice, acc, jacc, sens])

        success_count += 1
        if success_count % 100 == 0:  # Show progress every 100 images
            print(f"Processed {success_count} images...")

    print(f"✅ Finished! Successfully processed {success_count} image pairs")
    print("✅ Full metrics saved to:", output_csv)

# -----------------------------
# RUN SCRIPT
# -----------------------------
processed_folder = "/aakaou/ham10000/pipeline2/HAM10000_processed_images"
seg_folder = "/aakaou/ham10000/pipeline2/HAM10000_segmented_images"
output_csv = "/aakaou/ham10000/pipeline2/segmentation_metrics_all_images.csv"

evaluate_segmentation_metrics(processed_folder, seg_folder, output_csv)

"""This script evaluate_segmentation_metrics.py evaluates the segmentation performance of all processed skin lesion images against their corresponding ground-truth masks. 
For each image pair, it computes IoU, Dice coefficient, Accuracy, Jaccard index, and Sensitivity. Masks are binarized using a threshold and resized if necessary to match dimensions. 
All results are saved to a CSV file, while progress updates are printed every 100 images. The script automatically matches files based on their ISIC IDs, 
making it robust for large datasets, and provides a comprehensive evaluation of segmentation quality."""
