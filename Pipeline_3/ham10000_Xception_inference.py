#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAM10000 Xception 7-Class Classification Pipeline
-------------------------------------------------
1. Predict 7-class probabilities
2. Save predictions CSV
3. Compute classification metrics
4. Generate confusion matrix
5. Plot ROC curves + Macro AUC
"""

# =========================
# 1. IMPORTS
# =========================
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# =========================
# 2. SETTINGS
# =========================
PROC_DIR = Path("/aakaou/HAM10000_images_all")
OUTPUT_CSV = "/aakaou/ham10000_xception_7class_predictions_p3.csv"
METADATA_PATH = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"

IMG_SIZE = (299, 299)
BATCH_SIZE = 32

CLASS_NAMES = ['nv','mel','bkl','bcc','akiec','vasc','df']


# =========================
# 3. MODEL
# =========================
print("Loading Xception model...")

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.trainable = False

print("✅ Xception model loaded")


# =========================
# 4. HELPER FUNCTION
# =========================
def extract_number(filename):
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# =========================
# 5. LOAD IMAGE LIST
# =========================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
proc_files.sort(key=lambda x: extract_number(x.name))

print(f"Images found: {len(proc_files)}")


# =========================
# 6. PREDICTION
# =========================
results = []

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="Xception Predicting"):

    batch_paths = proc_files[i:i+BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32))

        batch_images.append(img)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    X_batch = np.array(batch_images)

    probs = model.predict(X_batch, verbose=0)

    pred_classes = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)

    for j,(fname,prob,pred_class) in enumerate(zip(batch_filenames,probs,pred_classes)):

        results.append({
            "filename":fname,
            "pred_class_id":int(pred_class),
            "pred_class_name":CLASS_NAMES[pred_class],
            "pred_confidence":float(pred_conf[j]),

            "prob_nv":float(prob[0]),
            "prob_mel":float(prob[1]),
            "prob_bkl":float(prob[2]),
            "prob_bcc":float(prob[3]),
            "prob_akiec":float(prob[4]),
            "prob_vasc":float(prob[5]),
            "prob_df":float(prob[6])
        })


# =========================
# 7. SAVE CSV
# =========================
if results:

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV,index=False)

    print(f"\nPredictions saved: {OUTPUT_CSV}")
    print("Images processed:",len(results_df))

else:

    print("No images processed")
    exit()


# =========================
# 8. LOAD METADATA
# =========================
metadata = pd.read_csv(METADATA_PATH)
metadata["filename"] = metadata["image_id"] + ".jpg"

df = results_df.merge(metadata[['filename','dx']], on="filename", how="inner")

print("Merged dataset:",df.shape)


# =========================
# 9. CLASSIFICATION REPORT
# =========================
print("\nClassification Report\n")

print(classification_report(
    df["dx"],
    df["pred_class_name"],
    labels=CLASS_NAMES,
    digits=4,
    zero_division=0
))


# =========================
# 10. CONFUSION MATRIX
# =========================
cm = confusion_matrix(df["dx"], df["pred_class_name"], labels=CLASS_NAMES)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("Confusion Matrix - Xception")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()


# =========================
# 11. ROC AUC ANALYSIS
# =========================
print("\nROC-AUC analysis")

prob_cols = [f"prob_{c}" for c in CLASS_NAMES]

lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df["dx"])

auc_scores = {}

for i,c in enumerate(CLASS_NAMES):

    y_true = y_true_bin[:,i]
    y_prob = df[prob_cols[i]].values

    if y_true.sum() < 2:
        continue

    roc_auc = roc_auc_score(y_true,y_prob)
    auc_scores[c] = roc_auc

    print(f"{c}: {roc_auc:.4f}")

macro_auc = np.mean(list(auc_scores.values()))
print("\nMacro AUC:",macro_auc)


# =========================
# 12. ROC CURVES
# =========================
plt.figure(figsize=(12,9))

colors = plt.cm.tab10(np.linspace(0,1,len(CLASS_NAMES)))

valid_classes = []

for i,c in enumerate(CLASS_NAMES):

    y_true = y_true_bin[:,i]
    y_prob = df[prob_cols[i]].values

    if y_true.sum() < 2:
        continue

    fpr,tpr,_ = roc_curve(y_true,y_prob)

    roc_auc = auc(fpr,tpr)

    valid_classes.append(c)

    plt.plot(
        fpr,
        tpr,
        lw=3,
        color=colors[i],
        label=f"{c} (AUC={roc_auc:.3f})"
    )

plt.plot([0,1],[0,1],'k--',label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Xception 7-class")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

print("\nROC curve saved")
print("Valid classes:",len(valid_classes))
