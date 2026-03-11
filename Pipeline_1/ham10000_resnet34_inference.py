# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Add, MaxPooling2D, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# 2. PATHS AND PARAMETERS
# ==========================================================

# Directory containing segmented lesion images
PROC_DIR = Path("/kaggle/working/HAM10000_segmented_p1")

# Output CSV file with predictions
OUTPUT_CSV = "/kaggle/working/ham10000_resnet34_segmented_predictions_p1.csv"

# Metadata file (ground truth labels)
META_PATH = "/kaggle/input/ham10000-dataset/HAM10000_metadata.csv"

# CNN input size
IMG_SIZE = (224, 224)

# Batch size for prediction
BATCH_SIZE = 32


# ==========================================================
# 3. HAM10000 CLASS LABELS
# ==========================================================

class_names = [
    'nv',     # melanocytic nevi
    'mel',    # melanoma
    'bkl',    # benign keratosis-like lesions
    'bcc',    # basal cell carcinoma
    'akiec',  # actinic keratoses
    'vasc',   # vascular lesions
    'df'      # dermatofibroma
]


# ==========================================================
# 4. HELPER FUNCTION — SORT FILES BY ISIC NUMBER
# ==========================================================

def extract_number(filename):
    """
    Extract numeric ID from filenames like:
    ISIC_0001234.jpg -> 1234
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# ==========================================================
# 5. RESNET BASIC BLOCK
# ==========================================================

def basic_block(x, filters, stride=1, downsample=False):
    """
    Basic residual block used in ResNet34.
    """

    shortcut = x

    # First convolution layer
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if size changes
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Residual addition
    x = Add()([shortcut, x])

    x = Activation('relu')(x)

    return x


# ==========================================================
# 6. BUILD RESNET34 MODEL
# ==========================================================

def ResNet34(input_shape=(224,224,3), num_classes=7):

    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Max pooling
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Block 1
    x = basic_block(x,64)
    x = basic_block(x,64)
    x = basic_block(x,64)

    # Block 2
    x = basic_block(x,128,stride=2,downsample=True)
    x = basic_block(x,128)
    x = basic_block(x,128)
    x = basic_block(x,128)

    # Block 3
    x = basic_block(x,256,stride=2,downsample=True)
    x = basic_block(x,256)
    x = basic_block(x,256)
    x = basic_block(x,256)
    x = basic_block(x,256)
    x = basic_block(x,256)

    # Block 4
    x = basic_block(x,512,stride=2,downsample=True)
    x = basic_block(x,512)
    x = basic_block(x,512)

    # Classification head
    x = GlobalAveragePooling2D()(x)

    x = Dense(512,activation='relu')(x)
    x = Dense(256,activation='relu')(x)

    outputs = Dense(num_classes,activation='softmax')(x)

    return Model(inputs,outputs)


# ==========================================================
# 7. BUILD MODEL
# ==========================================================

print("Building ResNet34...")

model = ResNet34()

# Freeze weights (inference mode)
model.trainable = False

print("Model ready.")


# ==========================================================
# 8. LOAD SEGMENTED IMAGES
# ==========================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

seg_files.sort(key=lambda x: extract_number(x.name))

print("Segmented images:",len(seg_files))


# ==========================================================
# 9. PREDICTION LOOP
# ==========================================================

print("Running predictions...")

results = []

for i in tqdm(range(0,len(seg_files),BATCH_SIZE)):

    batch_paths = seg_files[i:i+BATCH_SIZE]

    batch_images=[]
    batch_filenames=[]

    for img_path in batch_paths:

        image = cv2.imread(str(img_path))

        if image is None:
            continue

        image = cv2.resize(image,IMG_SIZE)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        processed = tf.keras.applications.resnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    X_batch = np.array(batch_images)

    probs = model.predict(X_batch,verbose=0)

    pred_classes = np.argmax(probs,axis=1)

    pred_probs = np.max(probs,axis=1)

    for j,(fname,prob,pred_class) in enumerate(
        zip(batch_filenames,probs,pred_classes)
    ):

        results.append({
            'filename':fname,
            'pred_class_id':int(pred_class),
            'pred_class_name':class_names[pred_class],
            'pred_confidence':float(pred_probs[j]),

            'prob_nv':float(prob[0]),
            'prob_mel':float(prob[1]),
            'prob_bkl':float(prob[2]),
            'prob_bcc':float(prob[3]),
            'prob_akiec':float(prob[4]),
            'prob_vasc':float(prob[5]),
            'prob_df':float(prob[6])
        })


# ==========================================================
# 10. SAVE PREDICTIONS
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_CSV,index=False)

print("Predictions saved:",OUTPUT_CSV)
print("Images processed:",len(results_df))


# ==========================================================
# 11. LOAD METADATA AND MERGE
# ==========================================================

pred_df = pd.read_csv(OUTPUT_CSV)

metadata = pd.read_csv(META_PATH)

metadata['filename'] = metadata['image_id'] + ".jpg"

df = pred_df.merge(metadata[['filename','dx']],on='filename')

print("Merged dataset:",df.shape)


# ==========================================================
# 12. CLASSIFICATION REPORT
# ==========================================================

print("\nCLASSIFICATION REPORT\n")

print(classification_report(
    df['dx'],
    df['pred_class_name'],
    labels=class_names,
    zero_division=0,
    digits=4
))


# ==========================================================
# 13. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    df['dx'],
    df['pred_class_name'],
    labels=class_names
)

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

plt.tight_layout()
plt.show()


# ==========================================================
# 14. ROC CURVES
# ==========================================================

lb = LabelBinarizer()

y_true_bin = lb.fit_transform(df['dx'])

plt.figure(figsize=(10,8))

auc_scores = {}

for i,cls in enumerate(class_names):

    y_true_cls = y_true_bin[:,i]

    y_prob_cls = df[f'prob_{cls}'].values

    if y_true_cls.sum() < 2:
        continue

    fpr,tpr,_ = roc_curve(y_true_cls,y_prob_cls)

    roc_auc = auc(fpr,tpr)

    auc_scores[cls]=roc_auc

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{cls} (AUC={roc_auc:.3f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - ResNet34")
plt.legend()

plt.grid()

plt.show()


# ==========================================================
# 15. MACRO AUC
# ==========================================================

macro_auc = np.mean(list(auc_scores.values()))

print("\nMACRO AUC:",macro_auc)

print("\nAUC per class")

for k,v in auc_scores.items():
    print(k,":",round(v,3))
