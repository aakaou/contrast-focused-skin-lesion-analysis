"""
VGG16 HAM10000 7-Class Classification Pipeline
----------------------------------------------
This script performs the following steps:

1. Loads dermoscopic images from a processed directory
2. Runs inference using a VGG16 deep learning model
3. Saves prediction probabilities to a CSV file
4. Merges predictions with HAM10000 metadata
5. Computes evaluation metrics (classification report)
6. Plots confusion matrix
7. Generates ROC curves and AUC scores

Dataset: HAM10000 Skin Lesion Dataset
Classes: nv, mel, bkl, bcc, akiec, vasc, df
"""

# ================================
# IMPORT LIBRARIES
# ================================

import cv2                              # OpenCV for image loading and processing
import numpy as np                      # Numerical operations
import pandas as pd                     # Data manipulation
from pathlib import Path                # File system paths
from tqdm import tqdm                   # Progress bar

import tensorflow as tf                 # Deep learning framework
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt         # Plotting
import seaborn as sns                   # Visualization


# ================================
# CONFIGURATION PATHS
# ================================

# Directory containing processed HAM10000 images
PROC_DIR = Path("/aakaou/HAM10000_images_all")

# Output directory
OUT_DIR = Path("/aakaou/")

# CSV file to store predictions
OUTPUT_CSV = "/aakaou/ham10000_vgg16_7class_predictions_p1.csv"

# Metadata file containing ground truth labels
METADATA_PATH = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"

# Model input image size
IMG_SIZE = (224, 224)

# Batch size for prediction
BATCH_SIZE = 32


# ================================
# HAM10000 CLASS LABELS
# ================================

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


# ================================
# BUILD VGG16 MODEL
# ================================

# Load pretrained VGG16 model without top classifier
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)   # Reduce spatial dimensions
x = Dense(512, activation='relu')(x)              # Fully connected layer
x = Dense(256, activation='relu')(x)              # Another dense layer
predictions = Dense(7, activation='softmax')(x)   # 7-class output layer

# Build final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pretrained layers
model.trainable = False

print("✅ VGG16 model loaded")


# ================================
# LOAD IMAGE FILES
# ================================

# Collect all JPG and PNG images
image_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

print(f"Found {len(image_files)} images")


# ================================
# IMAGE PREDICTION LOOP
# ================================

results = []   # List to store predictions

# Iterate through dataset in batches
for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Predicting"):

    batch_paths = image_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_names = []

    # Load and preprocess each image
    for img_path in batch_paths:

        image = cv2.imread(str(img_path))      # Read image
        if image is None:
            continue

        image = cv2.resize(image, IMG_SIZE)    # Resize to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

        # Apply VGG16 preprocessing
        image = preprocess_input(image.astype(np.float32))

        batch_images.append(image)
        batch_names.append(img_path.name)

    # Skip if batch empty
    if not batch_images:
        continue

    # Convert batch to numpy array
    X_batch = np.array(batch_images)

    # Run model prediction
    probs = model.predict(X_batch, verbose=0)

    # Predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Maximum probability (confidence)
    pred_conf = np.max(probs, axis=1)

    # Store predictions
    for j, fname in enumerate(batch_names):

        row = {
            'filename': fname,
            'pred_class_id': int(pred_classes[j]),
            'pred_class_name': class_names[pred_classes[j]],
            'pred_confidence': float(pred_conf[j])
        }

        # Store probability of each class
        for k, cls in enumerate(class_names):
            row[f'prob_{cls}'] = float(probs[j][k])

        results.append(row)


# ================================
# SAVE PREDICTIONS CSV
# ================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Predictions saved to: {OUTPUT_CSV}")


# ================================
# LOAD METADATA AND MERGE
# ================================

metadata = pd.read_csv(METADATA_PATH)

# Create filename column to match predictions
metadata['filename'] = metadata['image_id'] + ".jpg"

# Merge predictions with ground truth labels
df = results_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')

print("Merged dataset size:", df.shape)


# ================================
# CLASSIFICATION REPORT
# ================================

print("\n📊 Classification Report")

print(classification_report(
    df['dx'],
    df['pred_class_name'],
    labels=class_names,
    digits=4
))


# ================================
# CONFUSION MATRIX
# ================================

cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()


# ================================
# ROC CURVES
# ================================

print("\n🎯 ROC Curve Analysis")

# Convert labels to binary format
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'])

plt.figure(figsize=(10,8))

for i, cls in enumerate(class_names):

    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[f'prob_{cls}'].values

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{cls} (AUC={roc_auc:.3f})"
    )

# Random classifier baseline
plt.plot([0,1], [0,1], linestyle="--", color="black")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - VGG16 HAM10000")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

print("✅ ROC curve saved")
