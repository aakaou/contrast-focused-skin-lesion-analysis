# ============================================================
# MobileNetV3-Small HAM10000 7-Class Classification Pipeline
# Prediction + Evaluation + ROC Curves
# ============================================================

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROC_DIR = Path("/aakaou/HAM10000_images_all")
OUTPUT_CSV = "/aakaou/ham10000_mobilenetv3small_7class_predictions_p3.csv"
METADATA_PATH = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CLASSES
# ============================================================

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    base_model = MobileNetV3Small(
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

    print("✅ MobileNetV3-Small model loaded")

    return model


# ============================================================
# SORT IMAGE NAMES
# ============================================================

def extract_number(filename):

    import re
    m = re.search(r'ISIC_(\d+)', filename)

    if m:
        return int(m.group(1))
    return float('inf')


# ============================================================
# RUN PREDICTIONS
# ============================================================

def run_predictions(model):

    proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
    proc_files.sort(key=lambda x: extract_number(x.name))

    print(f"Found {len(proc_files)} images")

    results = []

    for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="Predicting"):

        batch_paths = proc_files[i:i + BATCH_SIZE]

        batch_images = []
        batch_filenames = []

        for img_path in batch_paths:

            image = cv2.imread(str(img_path))

            if image is None:
                continue

            image = cv2.resize(image, IMG_SIZE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            processed = preprocess_input(image.astype(np.float32))

            batch_images.append(processed)
            batch_filenames.append(img_path.name)

        if not batch_images:
            continue

        X_batch = np.array(batch_images)

        probs = model.predict(X_batch, verbose=0)

        pred_classes = np.argmax(probs, axis=1)
        pred_probs = np.max(probs, axis=1)

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

    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Predictions saved: {OUTPUT_CSV}")

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_results():

    pred_df = pd.read_csv(OUTPUT_CSV)

    metadata = pd.read_csv(METADATA_PATH)
    metadata['filename'] = metadata['image_id'] + '.jpg'

    df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')

    print("Merged dataset:", df.shape)

    print("\n📊 Classification Report\n")

    print(classification_report(
        df['dx'],
        df['pred_class_name'],
        labels=class_names,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return df


# ============================================================
# ROC CURVES
# ============================================================

def plot_roc(df):

    prob_cols = [f'prob_{c}' for c in class_names]

    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(df['dx'])

    plt.figure(figsize=(12,9))

    colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))

    auc_dict = {}

    for i, cls in enumerate(class_names):

        y_true_cls = y_true_bin[:, i]
        y_prob_cls = df[prob_cols[i]].values

        valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))

        y_true_cls = y_true_cls[valid_mask]
        y_prob_cls = y_prob_cls[valid_mask]

        if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
            print(f"Skip {cls}")
            continue

        fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
        roc_auc = auc(fpr, tpr)

        auc_dict[cls] = roc_auc

        plt.plot(
            fpr,
            tpr,
            lw=3,
            color=colors[i],
            label=f'{cls} (AUC={roc_auc:.3f})'
        )

    plt.plot([0,1],[0,1],'--',color='black',label='Random')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curves - MobileNetV3-Small")

    plt.legend(loc="lower right")

    plt.grid(True)

    plt.tight_layout()


    plt.show()

    macro_auc = np.mean(list(auc_dict.values()))

    print("\n🏆 MACRO AUC:", macro_auc)

    for c in auc_dict:
        print(c, auc_dict[c])


# ============================================================
# MAIN
# ============================================================

def main():

    model = build_model()

    run_predictions(model)

    df = evaluate_results()

    plot_roc(df)


if __name__ == "__main__":
    main()
