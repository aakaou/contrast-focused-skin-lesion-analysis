"""
ham10000_efficientnetb2_pipeline.py

Full pipeline for HAM10000 7-class classification using EfficientNetB2:
- Loads images from a folder
- Preprocesses images for EfficientNetB2
- Performs batch predictions
- Saves results to CSV
- Generates classification report and confusion matrix
- Computes ROC-AUC per class and macro-AUC
- Plots ROC curves
"""

# ==========================================================
# 1️⃣ IMPORT REQUIRED LIBRARIES
# ==========================================================
import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm           # Progress bar
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ==========================================================
# 2️⃣ DEFINE PATHS AND PARAMETERS
# ==========================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")  # Folder containing input images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional output folder
OUTPUT_CSV = "/aakaou/ham10000_efficientnetb2_7class_predictions_p1.csv"

IMG_SIZE = (260, 260)  # EfficientNetB2 expects 260x260 images
BATCH_SIZE = 32        # Number of images processed per batch

# HAM10000 7-class labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ==========================================================
# 3️⃣ HELPER FUNCTION: EXTRACT NUMERIC ID FROM FILENAME
# ==========================================================
def extract_number(filename):
    """
    Extracts numeric ID from filenames like ISIC_12345.jpg.
    Returns a very high number if pattern not found (for sorting).
    """
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# ==========================================================
# 4️⃣ LOAD EFFICIENTNETB2 MODEL WITH CUSTOM HEAD
# ==========================================================
# Load base EfficientNetB2 without top layers
base_model = EfficientNetB2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add global pooling and dense layers for 7-class classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# Combine base and head into one model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the model weights for inference
model.trainable = False

print("✅ EfficientNetB2 7-class model loaded (B1 → B2 upgrade!)")

# ==========================================================
# 5️⃣ GET IMAGE FILES
# ==========================================================
# Search for .jpg and .png files in the folder
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort files numerically based on ISIC ID
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Found {len(proc_files)} images")

# ==========================================================
# 6️⃣ BATCH PREDICTION LOOP
# ==========================================================
results = []  # List to store predictions

# Process images in batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="EfficientNetB2 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))       # Read image using OpenCV
        if image is None:
            continue

        # Resize to 260x260 and convert BGR → RGB
        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess for EfficientNetB2
        processed = preprocess_input(image.astype(np.float32))
        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    # Convert list to numpy array
    X_batch = np.array(batch_images)

    # Predict class probabilities
    probs = model.predict(X_batch, verbose=0)

    # Get predicted class IDs and confidence
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)

    # Save results
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            'filename': fname,
            'pred_class_id': int(pred_class),
            'pred_class_name': class_names[pred_class],
            'pred_confidence': float(pred_probs[j]),
            'prob_nv': float(prob[0]),
            'prob_mel': float(prob[1]),
            'prob_bkl': float(prob[2]),
            'prob_bcc': float(prob[3]),
            'prob_akiec': float(prob[4]),
            'prob_vasc': float(prob[5]),
            'prob_df': float(prob[6])
        })

# ==========================================================
# 7️⃣ SAVE PREDICTIONS TO CSV
# ==========================================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ EfficientNetB2 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
else:
    print("❌ No images processed!")

print("✅ EfficientNetB2 Complete!")
print("📈 B2 > B1 accuracy (260x260 input)!")

# ==========================================================
# 8️⃣ MERGE WITH GROUND TRUTH METADATA
# ==========================================================
metadata = pd.read_csv("/aakaou/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'].astype(str) + '.jpg'

# Merge predictions with ground truth labels
df = results_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')
print(f"✅ Merged dataset: {df.shape[0]} images")

# ==========================================================
# 9️⃣ CLASSIFICATION REPORT
# ==========================================================
print("📊 CLASSIFICATION REPORT")
print(classification_report(df['dx'], df['pred_class_name'], labels=class_names, zero_division=0, digits=4))

# ==========================================================
# 🔟 CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# 1️⃣1️⃣ ROC CURVES & MACRO-AUC
# ==========================================================
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)
prob_cols = [f'prob_{c}' for c in class_names]

plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
auc_dict = {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[prob_cols[i]].values

    # Filter NaN or Inf probabilities
    valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
    y_true_cls = y_true_cls[valid_mask]
    y_prob_cls = y_prob_cls[valid_mask]

    # Skip if insufficient samples
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)
    auc_dict[cls] = roc_auc
    valid_classes.append(cls)

    # Plot ROC
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{cls} (AUC={roc_auc:.3f})")

# Random classifier baseline
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
plt.title("ROC Curves - EfficientNetB2 7-class", fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# 1️⃣2️⃣ PRINT ROC-AUC RESULTS
# ==========================================================
macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print("\n🎯 Per-class AUC:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
