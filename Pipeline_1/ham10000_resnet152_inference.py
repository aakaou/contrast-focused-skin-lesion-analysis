# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ==========================================================
# 2. PATHS & PARAMETERS
# ==========================================================

# Folder containing preprocessed segmented images
PROC_DIR = Path("/kaggle/working/HAM10000_segmented_p1")

# CSV to save predictions
OUTPUT_CSV = "/kaggle/working/ham10000_resnet152_segmented_predictions_p1.csv"

# Image size for ResNet152V2
IMG_SIZE = (224, 224)

# Batch size for predictions
BATCH_SIZE = 32

# Skin lesion classes
class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']

# ==========================================================
# 3. LOAD RESNET152V2 PRETRAINED MODEL
# ==========================================================

print("🔄 Loading ResNet152V2 (ImageNet pretrained)...")

# Load backbone without classifier head
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

# Build final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pretrained weights
model.trainable = False

print("✅ ResNet152V2 ready (152 layers, ImageNet pretrained!)")

# ==========================================================
# 5. PREDICTION LOOP
# ==========================================================

print("🚀 Running ResNet152V2 predictions...")

results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet152 Segmented"):

    batch_paths = seg_files[i:i+BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:

        # Read and resize
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess for ResNet
        image = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    # Convert batch to array and predict
    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)

    # Store results per image
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

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Predictions saved to: {OUTPUT_CSV}")
print(f"📊 Images processed: {len(results_df)}")
print("🎯 Top predicted classes:")
print(results_df['pred_class_name'].value_counts().head())
print("📈 Prediction confidence stats:")
print(results_df['pred_confidence'].describe())

# ==========================================================
# 6. MERGE PREDICTIONS WITH TRUE LABELS
# ==========================================================

# Load HAM10000 metadata
metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with ground truth
df = results_df.merge(metadata[['filename','dx']], on='filename', how='inner')
print("Merged dataset size:", df.shape)

# ==========================================================
# 7. CLASSIFICATION REPORT
# ==========================================================
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(
    df['dx'],
    df['pred_class_name'],
    labels=class_names,
    digits=4,
    zero_division=0
))

# ==========================================================
# 8. CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ==========================================================
# 9. ROC CURVES (One-vs-Rest)
# ==========================================================
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'])

plt.figure(figsize=(12,9))

colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))
auc_dict = {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[f'prob_{cls}'].values

    # Skip if insufficient positive samples
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue

    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)

    auc_dict[cls] = roc_auc
    valid_classes.append(cls)

    # Plot ROC
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{cls} (AUC={roc_auc:.3f})")

# Random classifier baseline
plt.plot([0,1],[0,1],'k--', label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - ResNet152V2 7-class")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# 10. MACRO AUC
# ==========================================================
macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])

print("\n🎯 Per-class AUC:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

print(f"\n🏆 Macro-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
