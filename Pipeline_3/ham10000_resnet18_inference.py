#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HAM10000 SEGMENTED ResNet18 7-class Classification Pipeline
- Load segmented images
- Batch predictions
- Save CSV
- Classification report
- Confusion matrix
- ROC curves with AUC
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# === SETTINGS ===
PROC_DIR = Path("/aakaou/red_sonar_segmented_images")  
OUTPUT_CSV = "/aakaou/ham10000_resnet18_segmented_predictions_p3.csv"
METADATA_CSV = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === FUNCTIONS ===
def extract_number(filename):
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

def basic_block(x, filters, stride=1, downsample=False):
    shortcut = x
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def ResNet18(input_shape=(224,224,3), num_classes=7):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 128, stride=2, downsample=True)
    x = basic_block(x, 128)
    x = basic_block(x, 256, stride=2, downsample=True)
    x = basic_block(x, 256)
    x = basic_block(x, 512, stride=2, downsample=True)
    x = basic_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

def load_images(image_paths):
    images, filenames = [], []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.keras.applications.resnet.preprocess_input(img.astype(np.float32))
        images.append(img)
        filenames.append(img_path.name)
    return np.array(images), filenames

def predict_images(model, files, batch_size=BATCH_SIZE):
    results = []
    for i in tqdm(range(0, len(files), batch_size), desc="ResNet18 Segmented"):
        batch_paths = files[i:i+batch_size]
        X_batch, batch_filenames = load_images(batch_paths)
        if X_batch.size == 0:
            continue
        probs = model.predict(X_batch, verbose=0)
        pred_classes = np.argmax(probs, axis=1)
        pred_confidences = np.max(probs, axis=1)
        for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
            results.append({
                'filename': fname,
                'pred_class_id': int(pred_class),
                'pred_class_name': CLASS_NAMES[pred_class],
                'pred_confidence': float(pred_confidences[j]),
                **{f'prob_{c}': float(prob[idx]) for idx, c in enumerate(CLASS_NAMES)}
            })
    return results

def save_results_csv(results, output_csv=OUTPUT_CSV):
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Predictions saved: {output_csv}")
        return df
    else:
        print("❌ No segmented images!")
        return None

def merge_with_metadata(pred_df, metadata_csv=METADATA_CSV):
    meta_df = pd.read_csv(metadata_csv)
    meta_df['filename'] = meta_df['image_id'].astype(str) + '.jpg'
    df = pred_df.merge(meta_df[['filename', 'dx']], on='filename', how='inner')
    print(f"✅ Merged dataset: {df.shape[0]} images")
    return df

def classification_metrics(df):
    print("\n📊 CLASSIFICATION REPORT")
    print(classification_report(df['dx'], df['pred_class_name'], labels=CLASS_NAMES, zero_division=0, digits=4))
    cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=CLASS_NAMES)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(df):
    prob_cols = [f'prob_{c}' for c in CLASS_NAMES]
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(df['dx'])
    
    plt.figure(figsize=(12, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))
    auc_dict = {}
    valid_classes = []

    for i, cls in enumerate(CLASS_NAMES):
        y_true_cls = y_true_bin[:, i]
        y_prob_cls = df[prob_cols[i]].values
        valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
        y_true_cls = y_true_cls[valid_mask]
        y_prob_cls = y_prob_cls[valid_mask]
        if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
            print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
            continue
        fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
        roc_auc = auc(fpr, tpr)
        auc_dict[cls] = roc_auc
        valid_classes.append(cls)
        plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{cls} (AUC = {roc_auc:.3f})')

    plt.plot([0,1], [0,1], color='black', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves - ResNet18 7-class', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if auc_dict:
        macro_auc = np.mean(list(auc_dict.values()))
        print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
        print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
    else:
        print("\n⚠️ No valid ROC curves computed")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
    seg_files.sort(key=lambda x: extract_number(x.name))
    print(f"📁 Segmented images found: {len(seg_files)}")
    
    model = ResNet18()
    model.trainable = False
    print("✅ ResNet18 ready")
    
    results = predict_images(model, seg_files)
    pred_df = save_results_csv(results)
    
    if pred_df is not None:
        df = merge_with_metadata(pred_df)
        classification_metrics(df)
        plot_roc_curves(df)
    
    print("🎉 SEGMENTED ResNet18 COMPLETE!")
