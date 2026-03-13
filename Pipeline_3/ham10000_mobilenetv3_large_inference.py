"""
MobileNetV3-Large HAM10000 7-Class Skin Lesion Classification
Prediction + Evaluation + ROC Curves
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# PATHS
# ==========================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")
META_PATH = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"

OUTPUT_CSV = "/aakaou/ham10000_mobilenetv3large_7class_predictions_p3.csv"
ROC_FIG = "/aakaou/mobilenetv3large_p3_roc_curve.png"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# CLASSES
# ==========================================================
class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']

# ==========================================================
# MODEL BUILD
# ==========================================================
def build_model():
    base_model = MobileNetV3Large(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.trainable = False
    print("✅ MobileNetV3-Large model loaded")
    return model

# ==========================================================
# UTILS
# ==========================================================
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# ==========================================================
# PREDICTION PIPELINE
# ==========================================================
def run_predictions(model):
    files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
    files.sort(key=lambda x: extract_number(x.name))
    print(f"Images found: {len(files)}")

    results=[]
    for i in tqdm(range(0,len(files),BATCH_SIZE), desc="MobileNetV3-Large Predicting"):
        batch = files[i:i+BATCH_SIZE]
        imgs=[]
        names=[]
        for path in batch:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img.astype(np.float32))
            imgs.append(img)
            names.append(path.name)
        if not imgs:
            continue
        X = np.array(imgs)
        probs = model.predict(X, verbose=0)
        pred_classes = np.argmax(probs, axis=1)
        pred_probs = np.max(probs, axis=1)
        for j, (fname, prob, cls) in enumerate(zip(names, probs, pred_classes)):
            results.append({
                "filename": fname,
                "pred_class_id": int(cls),
                "pred_class_name": class_names[cls],
                "pred_confidence": float(pred_probs[j]),
                "prob_nv": float(prob[0]),
                "prob_mel": float(prob[1]),
                "prob_bkl": float(prob[2]),
                "prob_bcc": float(prob[3]),
                "prob_akiec": float(prob[4]),
                "prob_vasc": float(prob[5]),
                "prob_df": float(prob[6])
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Predictions saved: {OUTPUT_CSV}")
    return df

# ==========================================================
# EVALUATION
# ==========================================================
def evaluate(pred_df):
    meta = pd.read_csv(META_PATH)
    meta["filename"] = meta["image_id"] + ".jpg"
    df = pred_df.merge(meta[["filename","dx"]], on="filename")
    print(f"Merged dataset: {df.shape}")

    # Classification report
    print("\n📊 Classification Report\n")
    print(classification_report(
        df["dx"], df["pred_class_name"], labels=class_names, digits=4, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(df["dx"], df["pred_class_name"], labels=class_names)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ROC-AUC per class
    lb = LabelBinarizer()
    y_true = lb.fit_transform(df["dx"].values)
    auc_scores = {}
    for i, cls in enumerate(class_names):
        y_true_cls = y_true[:,i]
        y_prob_cls = df[f"prob_{cls}"].values
        if y_true_cls.sum() < 2:
            continue
        score = roc_auc_score(y_true_cls, y_prob_cls)
        auc_scores[cls] = score
        print(f"{cls}: AUC={score:.4f}")
    macro_auc = np.mean(list(auc_scores.values()))
    print(f"\n🎯 MACRO-AUC: {macro_auc:.4f}")
    return df

# ==========================================================
# ROC PLOT
# ==========================================================
def plot_roc(df):
    lb = LabelBinarizer()
    y_true = lb.fit_transform(df["dx"].values)
    plt.figure(figsize=(12,9))
    colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))
    valid_classes=[]
    auc_dict={}
    for i, cls in enumerate(class_names):
        y_true_cls = y_true[:,i]
        y_prob_cls = df[f"prob_{cls}"].values
        if y_true_cls.sum() < 2:
            print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
            continue
        fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
        roc_auc = auc(fpr,tpr)
        auc_dict[cls] = roc_auc
        valid_classes.append(cls)
        plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--', lw=2, label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - MobileNetV3-Large")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"✅ ROC Plot saved: {ROC_FIG}")
    plt.show()

# ==========================================================
# MAIN
# ==========================================================
def main():
    model = build_model()
    pred_df = run_predictions(model)
    df = evaluate(pred_df)
    plot_roc(df)

if __name__=="__main__":
    main()
