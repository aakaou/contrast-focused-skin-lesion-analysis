# ==============================
# evaluate_segmentation_all_pipelines.py
# ==============================

import os
import cv2
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# -------------------------------
# Metric Calculations
# -------------------------------

def load_mask(mask_path):
    """
    Load a segmentation mask and convert it to binary (0 or 1)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)  # Convert non-zero pixels to 1, zeros stay 0

def calculate_metrics(pred_mask, gt_mask):
    """
    Compute segmentation metrics: IoU, Jaccard, Sensitivity, Dice, Accuracy
    """
    pred_flat = pred_mask.flatten()  # Flatten predicted mask to 1D
    gt_flat = gt_mask.flatten()      # Flatten ground truth mask to 1D

    # True positives, false positives, true negatives, false negatives
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))

    union = TP + FP + FN
    iou = TP / union if union != 0 else 0
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    return iou, iou, sensitivity, dice, accuracy  # IoU repeated as Jaccard

# -------------------------------
# Evaluate folder of masks
# -------------------------------

def evaluate_segmentation(pred_folder, gt_folder, csv_path):
    """
    Evaluate segmentation for all predicted masks against ground truth
    Save metrics to a CSV file
    """
    pred_folder = os.path.abspath(pred_folder)
    gt_folder = os.path.abspath(gt_folder)
    results = []

    for pred_file in os.listdir(pred_folder):
        if not pred_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, pred_file)  # assume same filename
        if not os.path.exists(gt_path):
            logging.warning(f"Ground truth not found: {pred_file}")
            continue

        pred_mask = load_mask(pred_path)
        gt_mask = load_mask(gt_path)

        iou, jaccard, sensitivity, dice, accuracy = calculate_metrics(pred_mask, gt_mask)
        isic_id = os.path.splitext(pred_file)[0]

        results.append({
            "ISIC ID": isic_id,
            "IoU": iou,
            "Jaccard Index": jaccard,
            "Sensitivity": sensitivity,
            "Dice": dice,
            "Accuracy": accuracy
        })

    # Save results to CSV
    df = pd.DataFrame(results, columns=["ISIC ID", "IoU", "Jaccard Index", "Sensitivity", "Dice", "Accuracy"])
    df.to_csv(csv_path, index=False)
    logging.info(f"✅ Metrics saved to {csv_path}")


# -------------------------------
# Run evaluation for all pipelines
# -------------------------------

if __name__ == "__main__":
    pipelines_info = [
        {
            "pred_masks": r"/aakaou/pipeline1_seg_masks",
            "gt_masks": r"/aakaou/pipeline1_seg_overlays",
            "csv": r"/aakaou/pipeline1_seg_metrics.csv"
        },
        {
            "pred_masks": r"/aakaou/pipeline2_seg_masks",
            "gt_masks": r"/aakaou/pipeline2_seg_overlays",
            "csv": r"/aakaou/pipeline2_seg_metrics.csv"
        },
        {
            "pred_masks": r"/aakaou/pipeline3_seg_masks",
            "gt_masks": r"/aakaou/pipeline3_seg_overlays",
            "csv": r"/aakaou/pipeline3_seg_metrics.csv"
        },
        {
            "pred_masks": r"/aakaou/pipeline4_seg_masks",
            "gt_masks": r"/aakaou/pipeline4_overlays",
            "csv": r"/aakaou/pipeline4_seg_metrics.csv"
        }
    ]

    for pipeline in pipelines_info:
        logging.info(f"📊 Evaluating pipeline: {pipeline['pred_masks']}")
        evaluate_segmentation(pipeline['pred_masks'], pipeline['gt_masks'], pipeline['csv'])
