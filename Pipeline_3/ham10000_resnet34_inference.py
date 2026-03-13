"""
HAM10000 ResNet34 Segmented Images Classification Pipeline
- Predicts 7-class skin lesion probabilities
- Saves CSV of predictions
- Generates classification report, confusion matrix
- Plots ROC curves with AUC
"""

# ======================
# 1️⃣ IMPORTS
# ======================
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

# ======================
# 2️⃣ SETTINGS
# ======================
PROC_DIR = Path("/aakaou/red_sonar_segmented_images")  # Path to segmented images
OUTPUT_CSV = "/aakaou/ham10000_resnet34_segmented_predictions_p3.csv"  # Output CSV
METADATA_CSV = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"  # HAM10000 metadata
IMG_SIZE = (224, 224)  # Input image size for ResNet
BATCH_SIZE = 32  # Batch size for prediction
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # 7 HAM10000 classes

# ======================
# 3️⃣ HELPER FUNCTIONS
# ======================

# Extract numeric ID from filename, e.g., ISIC_12345.jpg -> 12345
def extract_number(filename):
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# ResNet basic block: two conv layers + residual connection
def basic_block(x, filters, stride=1, downsample=False):
    shortcut = x
    # 1st conv
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 2nd conv
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    # Downsample shortcut if needed
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    # Add residual
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# Build ResNet34 architecture
def ResNet34(input_shape=(224,224,3), num_classes=7):
    inputs = Input(shape=input_shape)
    # Initial conv + maxpool
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Block 1: 64 filters [3 blocks]
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    
    # Block 2: 128 filters [4 blocks]
    x = basic_block(x, 128, stride=2, downsample=True)
    x = basic_block(x, 128)
    x = basic_block(x, 128)
    x = basic_block(x, 128)
    
    # Block 3: 256 filters [6 blocks]
    x = basic_block(x, 256, stride=2, downsample=True)
    for _ in range(5):
        x = basic_block(x, 256)
    
    # Block 4: 512 filters [3 blocks]
    x = basic_block(x, 512, stride=2, downsample=True)
    x = basic_block(x, 512)
    x = basic_block(x, 512)
    
    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Load and preprocess images in a batch
def load_images(image_paths):
    images, filenames = [], []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ResNet preprocessing
        img = tf.keras.applications.resnet.preprocess_input(img.astype(np.float32))
        images.append(img)
        filenames.append(img_path.name)
    return np.array(images), filenames

# ======================
# 4️⃣ MAIN PREDICTION
# ======================
if __name__ == "__main__":
    # List segmented images
    seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
    seg_files.sort(key=lambda x: extract_number(x.name))
    print(f"📁 Segmented images: {len(seg_files)}")
    
    # Build model
    print("🔄 Building ResNet34...")
    model = ResNet34()
    model.trainable = False
    print("✅ ResNet34 ready (34 layers)")
    
    # Predict in batches
    results = []
    print("🚀 ResNet34 predictions...")
    for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet34 Segmented"):
        batch_paths = seg_files[i:i+BATCH_SIZE]
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
    
    # Save CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ ResNet34 SEGMENTED predictions saved: {OUTPUT_CSV}")
        print(f"📊 Total images: {len(results_df)}")
        print("🎯 Top predictions distribution:")
        print(results_df['pred_class_name'].value_counts().head())
    else:
        print("❌ No images processed!")

    # ======================
    # 5️⃣ METRICS & EVALUATION
    # ======================
    pred_df = pd.read_csv(OUTPUT_CSV)
    meta_df = pd.read_csv(METADATA_CSV)
    meta_df['filename'] = meta_df['image_id'].astype(str) + '.jpg'
    df = pred_df.merge(meta_df[['filename','dx']], on='filename', how='inner')
    print(f"✅ Merged dataset: {df.shape[0]} images")

    # Classification report
    print("\n📊 CLASSIFICATION REPORT")
    print(classification_report(df['dx'], df['pred_class_name'], labels=CLASS_NAMES, zero_division=0, digits=4))

    # Confusion matrix
    cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=CLASS_NAMES)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ROC curves
    prob_cols = [f'prob_{c}' for c in CLASS_NAMES]
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(df['dx'])
    
    plt.figure(figsize=(12,9))
    colors = plt.cm.tab10(np.linspace(0,1,len(CLASS_NAMES)))
    auc_dict = {}
    valid_classes = []

    for i, cls in enumerate(CLASS_NAMES):
        y_true_cls = y_true_bin[:,i]
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
        plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{cls} (AUC={roc_auc:.3f})')

    # Random baseline
    plt.plot([0,1],[0,1], color='black', lw=2, linestyle='--', label='Random (AUC=0.5)')
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves - ResNet34 7-class', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if auc_dict:
        macro_auc = np.mean(list(auc_dict.values()))
        print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
        print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
