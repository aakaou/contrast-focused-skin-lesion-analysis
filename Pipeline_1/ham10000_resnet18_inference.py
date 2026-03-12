# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os                          # Provides OS utilities (file handling)
import cv2                         # OpenCV library for image processing
import numpy as np                 # Numerical computing library
import pandas as pd                # Data analysis library (CSV handling)
from pathlib import Path           # Object-oriented filesystem paths
from tqdm import tqdm              # Progress bar for loops

import tensorflow as tf            # TensorFlow deep learning framework
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    Add, MaxPooling2D, GlobalAveragePooling2D, Dense
)                                  # Keras layers used to build ResNet18
from tensorflow.keras.models import Model  # Keras model class


# ============================================================
# PATHS AND PARAMETERS
# ============================================================

PROC_DIR = Path("/aakaou/HAM10000_segmented_p1")
# Folder containing segmented skin lesion images

OUTPUT_CSV = "/aakaou/ham10000_resnet18_segmented_predictions_p1.csv"
# CSV file where predictions will be stored

IMG_SIZE = (224, 224)              # Image size expected by CNN models

BATCH_SIZE = 32                    # Number of images processed per batch


# ============================================================
# RESNET18 BASIC RESIDUAL BLOCK
# ============================================================

def basic_block(x, filters, stride=1, downsample=False):
    """
    Implements the basic residual block used in ResNet18.
    Residual connections help prevent vanishing gradients
    and allow deeper networks to train effectively.
    """

    shortcut = x  # Save input tensor for residual connection

    # First convolution layer
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)  # Normalize activations
    x = Activation('relu')(x)    # Apply ReLU activation

    # Second convolution layer
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    # If feature map size changes, adjust shortcut path
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut connection (ResNet identity mapping)
    x = Add()([shortcut, x])

    # Apply activation after addition
    x = Activation('relu')(x)

    return x


# ============================================================
# BUILD RESNET18 ARCHITECTURE
# ============================================================

def ResNet18(input_shape=(224, 224, 3), num_classes=7):
    """
    Creates a ResNet18 architecture from scratch.
    Output layer uses softmax for 7 skin lesion classes.
    """

    inputs = Input(shape=input_shape)  # Input tensor

    # Initial convolution layer
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Initial max pooling layer
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual block group 1
    x = basic_block(x, 64)
    x = basic_block(x, 64)

    # Residual block group 2
    x = basic_block(x, 128, stride=2, downsample=True)
    x = basic_block(x, 128)

    # Residual block group 3
    x = basic_block(x, 256, stride=2, downsample=True)
    x = basic_block(x, 256)

    # Residual block group 4
    x = basic_block(x, 512, stride=2, downsample=True)
    x = basic_block(x, 512)

    # Global pooling reduces feature map to vector
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers for classification
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    # Final softmax output layer (7 classes)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ============================================================
# LOAD MODEL
# ============================================================

print("🔄 Loading ResNet18...")

model = ResNet18()  # Build ResNet18 architecture

model.trainable = False  # Freeze weights (inference only)

print("✅ ResNet18 ready")


# ============================================================
# HAM10000 CLASS LABELS
# ============================================================

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# Skin lesion diagnostic classes


# ============================================================
# FUNCTION TO SORT IMAGE FILES
# ============================================================

def extract_number(filename):
    """
    Extract numeric part from ISIC image names.
    Example: ISIC_0024306.jpg → 24306
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# ============================================================
# LOAD SEGMENTED IMAGES
# ============================================================

# Collect all segmented images
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images numerically
seg_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Segmented images found: {len(seg_files)}")


# ============================================================
# MODEL PREDICTION
# ============================================================

print("🚀 Segment-aware predictions...")

results = []  # Store prediction results

# Process images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet18 Segmented"):

    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:

        image = cv2.imread(str(img_path))  # Read image

        if image is None:  # Skip invalid images
            continue

        image = cv2.resize(image, IMG_SIZE)  # Resize to 224x224

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

        # Apply ResNet preprocessing
        processed = tf.keras.applications.resnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:  # Skip empty batches
        continue

    X_batch = np.array(batch_images)

    # Run model inference
    probs = model.predict(X_batch, verbose=0)

    pred_classes = np.argmax(probs, axis=1)  # Predicted class index

    pred_probs = np.max(probs, axis=1)  # Prediction confidence

    # Store predictions
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


# ============================================================
# SAVE PREDICTIONS
# ============================================================

if results:

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ SEGMENTED ResNet18 saved: {OUTPUT_CSV}")

    print(f"📊 Images processed: {len(results_df)}")

    print("🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())

else:
    print("❌ No segmented images!")

print("🎉 SEGMENTED ResNet18 COMPLETE!")


# ============================================================
# EVALUATION SECTION
# ============================================================

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# Load predictions CSV
pred_df = pd.read_csv(OUTPUT_CSV)

# Load HAM10000 metadata
metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")

# Create filename column
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with ground truth labels
df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')

print("Merged:", df.shape)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("📊 CLASSIFICATION REPORT")

print(classification_report(
    df['dx'], df['pred_class_name'],
    labels=class_names,
    zero_division=0,
    digits=4
))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


# ============================================================
# ROC CURVE ANALYSIS
# ============================================================

lb = LabelBinarizer()

y_true_bin = lb.fit_transform(df['dx'])

plt.figure(figsize=(12,9))

colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))

auc_dict = {}

for i, cls in enumerate(class_names):

    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[f'prob_{cls}'].values

    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)

    roc_auc = auc(fpr, tpr)

    auc_dict[cls] = roc_auc

    plt.plot(
        fpr, tpr,
        lw=3,
        color=colors[i],
        label=f'{cls} (AUC = {roc_auc:.3f})'
    )

# Random baseline
plt.plot([0,1],[0,1],'k--',label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - ResNet18 HAM10000")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

macro_auc = np.mean(list(auc_dict.values()))

print("\n🏆 MACRO-AUC:", round(macro_auc,3))

print("Best class:", max(auc_dict, key=auc_dict.get))
