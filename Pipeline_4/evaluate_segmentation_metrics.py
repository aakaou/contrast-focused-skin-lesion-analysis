import cv2                     # OpenCV for image reading and processing
import numpy as np             # NumPy for numerical operations
import os                      # OS module to handle file paths and directories
import pandas as pd            # Pandas to store results and save CSV

# -------------------------------
# Metric Calculations
# -------------------------------

def load_mask(mask_path):
    """
    Load a segmentation mask and convert it to binary (0 or 1)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
    return (mask > 0).astype(np.uint8)                 # Convert non-zero pixels to 1, zeros stay 0

def calculate_metrics(pred_mask, gt_mask):
    """
    Compute segmentation metrics: IoU, Jaccard, Sensitivity, Dice, Accuracy
    """
    pred_flat = pred_mask.flatten()   # Flatten predicted mask to 1D array
    gt_flat = gt_mask.flatten()       # Flatten ground truth mask to 1D array

    # Compute True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    TP = np.sum((pred_flat == 1) & (gt_flat == 1))  # Correctly predicted lesion pixels
    FP = np.sum((pred_flat == 1) & (gt_flat == 0))  # Predicted lesion but actually background
    TN = np.sum((pred_flat == 0) & (gt_flat == 0))  # Correctly predicted background pixels
    FN = np.sum((pred_flat == 0) & (gt_flat == 1))  # Missed lesion pixels

    # Intersection over Union (IoU) / Jaccard Index
    union = TP + FP + FN
    iou = TP / union if union != 0 else 0           # Avoid division by zero

    # Dice coefficient (F1 score for segmentation)
    dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

    # Sensitivity / Recall (true positive rate)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0

    # Overall accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

    return iou, iou, sensitivity, dice, accuracy  # IoU repeated as Jaccard

# -------------------------------
# Evaluate all images
# -------------------------------

def evaluate_segmentation(pred_folder, gt_folder, csv_path):
    """
    Evaluate segmentation for all images in pred_folder against gt_folder
    and save metrics to a CSV file.
    """
    results = []  # Store results for each image

    for pred_file in os.listdir(pred_folder):  # Loop through predicted masks
        if not pred_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
            continue

        pred_path = os.path.join(pred_folder, pred_file)  # Full path to predicted mask
        gt_path = os.path.join(gt_folder, pred_file)      # Full path to ground truth mask (assume same filename)

        if not os.path.exists(gt_path):                   # Check if ground truth exists
            print(f"Ground truth not found for {pred_file}")
            continue

        pred_mask = load_mask(pred_path)  # Load predicted mask
        gt_mask = load_mask(gt_path)      # Load ground truth mask

        # Calculate metrics
        iou, jaccard, sensitivity, dice, accuracy = calculate_metrics(pred_mask, gt_mask)

        # Extract ISIC ID from filename (remove extension)
        isic_id = os.path.splitext(pred_file)[0]

        # Append metrics for this image
        results.append({
            "ISIC ID": isic_id,
            "IoU": iou,
            "Jaccard Index": jaccard,
            "Sensitivity": sensitivity,
            "Dice": dice,
            "Accuracy": accuracy
        })

    # Save all results to a CSV file
    df = pd.DataFrame(results, columns=["ISIC ID", "IoU", "Jaccard Index", "Sensitivity", "Dice", "Accuracy"])
    df.to_csv(csv_path, index=False)  # Write CSV without index
    print(f"✅ Metrics saved to {csv_path}")

# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    # Folders containing predicted masks and ground truth masks
    prediction_folder = r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_masks1"
    ground_truth_folder = r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_overlays1"

    # Output CSV file path
    csv_output = r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/segmentation_metrics.csv"

    # Run evaluation
    evaluate_segmentation(prediction_folder, ground_truth_folder, csv_output)
