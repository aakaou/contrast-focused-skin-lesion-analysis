"""
HAM10000 MobileNetV1 7-Class Skin Lesion Classification Pipeline

Steps:
1. Load MobileNetV1 pretrained on ImageNet
2. Predict 7 HAM10000 classes
3. Save predictions CSV
4. Merge with metadata
5. Generate:
   - Classification report
   - Confusion matrix
   - ROC-AUC scores
   - ROC curve plot
"""

# =====================================================
# 1. IMPORTS
# =====================================================
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# =====================================================
# 2. SETTINGS
# =====================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")
OUTPUT_CSV = "/aakaou/ham10000_mobilenetv1_7class_predictions_p3.csv"
METADATA_PATH = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ['nv','mel','bkl','bcc','akiec','vasc','df']


# =====================================================
# 3. LOAD MODEL
# =====================================================
print("Loading MobileNetV1 model...")

base_model = MobileNet(
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

print("✅ MobileNetV1 loaded (lightweight CNN)")


# =====================================================
# 4. HELPER FUNCTION
# =====================================================
def extract_number(filename):

    m = re.search(r'ISIC_(\d+)', filename)

    return int(m.group(1)) if m else float('inf')


# =====================================================
# 5. LOAD IMAGE LIST
# =====================================================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

proc_files.sort(key=lambda x: extract_number(x.name))

print("Images found:", len(proc_files))


# =====================================================
# 6. PREDICTIONS
# =====================================================
results = []

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="MobileNetV1 Predicting"):

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


# =====================================================
# 7. SAVE CSV
# =====================================================
if results:

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_CSV,index=False)

    print("\nPredictions saved:", OUTPUT_CSV)

    print("Images processed:", len(results_df))

    print("\nPrediction distribution:")

    print(results_df["pred_class_name"].value_counts())

    print("\nConfidence statistics:")

    print(results_df["pred_confidence"].describe())

else:

    print("No images processed")

    exit()

print("MobileNetV1 inference complete ⚡")


# =====================================================
# 8. LOAD METADATA
# =====================================================
metadata = pd.read_csv(METADATA_PATH)

metadata["filename"] = metadata["image_id"] + ".jpg"

df = results_df.merge(metadata[['filename','dx']], on='filename', how='inner')

print("\nMerged dataset:", df.shape)


# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================
print("\nClassification Report\n")

print(classification_report(
    df["dx"],
    df["pred_class_name"],
    labels=CLASS_NAMES,
    digits=4,
    zero_division=0
))


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================
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

plt.title("Confusion Matrix - MobileNetV1")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# =====================================================
# 11. ROC AUC
# =====================================================
print("\nROC-AUC Analysis")

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

print("\nMacro AUC:", macro_auc)


# =====================================================
# 12. ROC CURVES
# =====================================================
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

plt.title("ROC Curves - MobileNetV1 7-class")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

print("\nROC curve saved")

print("Valid classes:", len(valid_classes))
