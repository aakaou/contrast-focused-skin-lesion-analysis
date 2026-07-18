# ==============================
# evaluate_segmentation_all_pipelines_two_datasets.py
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
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def calculate_metrics(pred_mask, gt_mask):
    """
    Compute segmentation metrics: IoU, Jaccard, Sensitivity, Dice, Accuracy
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    TP = np.sum((pred_flat == 1) & (gt_flat == 1))
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))

    union = TP + FP + FN
    iou = TP / union if union != 0 else 0.0
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0

    # Jaccard = IoU in this setup
    jaccard = iou

    return iou, jaccard, sensitivity, dice, accuracy


# -------------------------------
# Evaluate folder of masks
# -------------------------------

def evaluate_segmentation(pred_folder, gt_folder, csv_path):
    """
    Evaluate segmentation for all predicted masks against ground truth.
    Save metrics to a CSV file.
    """
    pred_folder = os.path.abspath(pred_folder)
    gt_folder = os.path.abspath(gt_folder)

    logging.info(f"Predicted masks folder: {pred_folder}")
    logging.info(f"Ground truth masks folder: {gt_folder}")

    results = []

    if not os.path.isdir(pred_folder):
        logging.error(f"❌ Pred folder not found: {pred_folder}")
        return

    if not os.path.isdir(gt_folder):
        logging.error(f"❌ GT folder not found: {gt_folder}")
        return

    for pred_file in os.listdir(pred_folder):
        if not pred_file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, pred_file)  # assume same filename

        if not os.path.exists(gt_path):
            logging.warning(f"Ground truth not found: {pred_file}")
            continue

        try:
            pred_mask = load_mask(pred_path)
            gt_mask = load_mask(gt_path)

            iou, jaccard, sensitivity, dice, accuracy = calculate_metrics(
                pred_mask, gt_mask
            )
            image_id = os.path.splitext(pred_file)[0]

            results.append(
                {
                    "Image ID": image_id,
                    "IoU": iou,
                    "Jaccard Index": jaccard,
                    "Sensitivity": sensitivity,
                    "Dice": dice,
                    "Accuracy": accuracy,
                }
            )
        except Exception as e:
            logging.error(f"Error processing {pred_file}: {e}")

    if len(results) == 0:
        logging.warning("No results to save (no valid mask pairs found).")
        return

    df = pd.DataFrame(
        results,
        columns=["Image ID", "IoU", "Jaccard Index", "Sensitivity", "Dice", "Accuracy"],
    )
    df.to_csv(csv_path, index=False)
    logging.info(f"✅ Metrics saved to {csv_path}")


# -------------------------------
# Build configs for both datasets
# -------------------------------

def build_pipeline_configs():
    """
    Define predicted/GT mask folders and CSV output for:
    - 4 pipelines on ISIC 2024
    - 4 pipelines on HAM10000
    Adjust these base paths to your actual directory structure.
    """
    configs = []

    # ISIC 2024 base
    isic_base = "/home/aboubakr/Descargas/article4/isic2016_2020"

    configs.extend(
        [
            {
                "name": "isic2024_pipeline1",
                "pred_masks": os.path.join(isic_base, "pipeline1/unet_masks"),
                "gt_masks": os.path.join(isic_base, "pipeline1/gt_masks"),
                "csv": os.path.join(isic_base, "pipeline1/seg_metrics.csv"),
            },
            {
                "name": "isic2024_pipeline2",
                "pred_masks": os.path.join(isic_base, "pipeline2/unet_masks"),
                "gt_masks": os.path.join(isic_base, "pipeline2/gt_masks"),
                "csv": os.path.join(isic_base, "pipeline2/seg_metrics.csv"),
            },
            {
                "name": "isic2024_pipeline3",
                "pred_masks": os.path.join(isic_base, "pipeline3/unet_masks"),
                "gt_masks": os.path.join(isic_base, "pipeline3/gt_masks"),
                "csv": os.path.join(isic_base, "pipeline3/seg_metrics.csv"),
            },
            {
                "name": "isic2024_pipeline4",
                "pred_masks": os.path.join(isic_base, "pipeline4/unet_masks"),
                "gt_masks": os.path.join(isic_base, "pipeline4/gt_masks"),
                "csv": os.path.join(isic_base, "pipeline4/seg_metrics.csv"),
            },
        ]
    )

    # HAM10000 base
    ham_base = "/home/aboubakr/Descargas/article4/ham10000"

    configs.extend(
        [
            {
                "name": "ham10000_pipeline1",
                "pred_masks": os.path.join(ham_base, "pipeline1/unet_masks"),
                "gt_masks": os.path.join(ham_base, "pipeline1/gt_masks"),
                "csv": os.path.join(ham_base, "pipeline1/seg_metrics.csv"),
            },
            {
                "name": "ham10000_pipeline2",
                "pred_masks": os.path.join(ham_base, "pipeline2/unet_masks"),
                "gt_masks": os.path.join(ham_base, "pipeline2/gt_masks"),
                "csv": os.path.join(ham_base, "pipeline2/seg_metrics.csv"),
            },
            {
                "name": "ham10000_pipeline3",
                "pred_masks": os.path.join(ham_base, "pipeline3/unet_masks"),
                "gt_masks": os.path.join(ham_base, "pipeline3/gt_masks"),
                "csv": os.path.join(ham_base, "pipeline3/seg_metrics.csv"),
            },
            {
                "name": "ham10000_pipeline4",
                "pred_masks": os.path.join(ham_base, "pipeline4/unet_masks"),
                "gt_masks": os.path.join(ham_base, "pipeline4/gt_masks"),
                "csv": os.path.join(ham_base, "pipeline4/seg_metrics.csv"),
            },
        ]
    )

    return configs


# -------------------------------
# Run evaluation for all pipelines
# -------------------------------

if __name__ == "__main__":
    pipeline_configs = build_pipeline_configs()

    for cfg in pipeline_configs:
        logging.info(
            f"📊 Evaluating {cfg['name']} "
            f"pred={cfg['pred_masks']} gt={cfg['gt_masks']}"
        )
        evaluate_segmentation(
            pred_folder=cfg["pred_masks"],
            gt_folder=cfg["gt_masks"],
            csv_path=cfg["csv"],
        )
