import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve, auc

# ==============================
# CONFIGURATION
# ==============================

PROC_DIR = Path("/aakaou/HAM10000_images_all")
METADATA_PATH = "/aakaou/HAM10000_metadata.csv"

OUTPUT_CSV = "/aakaou/ham10000_efficientnetb1_predictions.csv"
ROC_OUTPUT = "/aakaou/efficientnetb1_roc_curve.png"

IMG_SIZE = (240, 240)
BATCH_SIZE = 32

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


# ==============================
# MODEL CREATION
# ==============================

def build_model():

    base_model = EfficientNetB1(
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

    print("Model EfficientNetB1 loaded")

    return model


# ==============================
# IMAGE SORTING
# ==============================

def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# ==============================
# PREDICTION PIPELINE
# ==============================

def run_predictions(model):

    proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
    proc_files.sort(key=lambda x: extract_number(x.name))

    print("Images found:", len(proc_files))

    results = []

    for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="Predicting"):

        batch_paths = proc_files[i:i+BATCH_SIZE]

        batch_images = []
        batch_names = []

        for img_path in batch_paths:

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            image = cv2.resize(image, IMG_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            processed = preprocess_input(image.astype(np.float32))

            batch_images.append(processed)
            batch_names.append(img_path.name)

        if not batch_images:
            continue

        X_batch = np.array(batch_images)

        probs = model.predict(X_batch, verbose=0)

        pred_classes = np.argmax(probs, axis=1)
        pred_conf = np.max(probs, axis=1)

        for j, fname in enumerate(batch_names):

            prob = probs[j]
            pred_class = pred_classes[j]

            results.append({
                'filename': fname,
                'pred_class_id': int(pred_class),
                'pred_class_name': CLASS_NAMES[pred_class],
                'pred_confidence': float(pred_conf[j]),
                'prob_nv': float(prob[0]),
                'prob_mel': float(prob[1]),
                'prob_bkl': float(prob[2]),
                'prob_bcc': float(prob[3]),
                'prob_akiec': float(prob[4]),
                'prob_vasc': float(prob[5]),
                'prob_df': float(prob[6])
            })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Predictions saved:", OUTPUT_CSV)

    return df


# ==============================
# EVALUATION
# ==============================

def evaluate_predictions(pred_df):

    metadata = pd.read_csv(METADATA_PATH)
    metadata['filename'] = metadata['image_id'] + ".jpg"

    df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')

    print("Merged dataset:", df.shape)

    print("\nClassification Report\n")

    print(classification_report(
        df['dx'],
        df['pred_class_name'],
        labels=CLASS_NAMES,
        zero_division=0,
        digits=4
    ))

    cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=CLASS_NAMES)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.show()

    return df


# ==============================
# ROC CURVES
# ==============================

def plot_roc(df):

    prob_cols = [f'prob_{c}' for c in CLASS_NAMES]

    lb = LabelBinarizer()

    y_true_bin = lb.fit_transform(df['dx'].values)

    plt.figure(figsize=(10,8))

    colors = plt.cm.tab10(np.linspace(0,1,len(CLASS_NAMES)))

    auc_scores = {}

    for i, cls in enumerate(CLASS_NAMES):

        y_true = y_true_bin[:, i]
        y_prob = df[prob_cols[i]].values

        if y_true.sum() < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        auc_scores[cls] = roc_auc

        plt.plot(
            fpr,
            tpr,
            color=colors[i],
            lw=3,
            label=f"{cls} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves EfficientNetB1")

    plt.legend()

    plt.grid(True)

    plt.show()

    print("\nAUC scores")

    for k,v in auc_scores.items():
        print(k, ":", round(v,3))

    print("\nMacro AUC:", np.mean(list(auc_scores.values())))


# ==============================
# MAIN
# ==============================

def main():

    model = build_model()

    pred_df = run_predictions(model)

    df = evaluate_predictions(pred_df)

    plot_roc(df)


if __name__ == "__main__":
    main()
