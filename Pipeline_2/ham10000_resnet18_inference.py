"""
This script performs segmentation-aware classification of HAM10000 skin lesion images using a custom ResNet18 model implemented in Keras. 
The workflow uses pre-segmented images to potentially improve classification accuracy. Images are loaded from a directory, resized to 224×224, 
and preprocessed with ResNet preprocessing. Predictions are made in batches, and the script outputs the predicted class, confidence score, 
and class probabilities for all 7 skin lesion categories. Results are saved to a CSV for downstream analysis, including distribution checks and confidence evaluation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# === Helper function to extract numeric ID from ISIC filenames ===
# Example: ISIC_0024306.jpg → 24306, used for proper sorting of images
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# === Paths & Settings ===
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")  # segmented image folder
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_resnet18_segmented_predictions_p2.csv"  # output CSV
IMG_SIZE = (224, 224)  # input size for ResNet18
BATCH_SIZE = 32         # batch size for predictions

# === ResNet18 Basic Residual Block ===
def basic_block(x, filters, stride=1, downsample=False):
    """
    Defines a single ResNet18 residual block with 2 conv layers.
    - x: input tensor
    - filters: number of channels
    - stride: stride for first conv layer
    - downsample: whether to apply 1x1 conv to shortcut
    """
    shortcut = x
    # First conv layer
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Second conv layer
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    # Adjust shortcut if needed
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    # Add shortcut and main path
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

# === ResNet18 Model Definition ===
def ResNet18(input_shape=(224, 224, 3), num_classes=7):
    """
    Build ResNet18 for 7-class classification.
    """
    inputs = Input(shape=input_shape)
    # Initial conv + maxpool
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 128, stride=2, downsample=True)
    x = basic_block(x, 128)
    x = basic_block(x, 256, stride=2, downsample=True)
    x = basic_block(x, 256)
    x = basic_block(x, 512, stride=2, downsample=True)
    x = basic_block(x, 512)
    
    # Global pooling + dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# === Initialize ResNet18 Model ===
print("🔄 Loading ResNet18...")
model = ResNet18()
model.trainable = False  # Freeze weights for inference only
print("✅ ResNet18 ready")

# Class names for HAM10000 dataset
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === Load segmented image filenames ===
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
seg_files.sort(key=lambda x: extract_number(x.name))  # sort by numeric ID
print(f"📁 Segmented images found: {len(seg_files)}")

# === Predict on segmented images in batches ===
print("🚀 Segment-aware predictions...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet18 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]
    
    batch_images = []
    batch_filenames = []
    
    # Load, resize, convert to RGB, preprocess
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))
        batch_images.append(processed)
        batch_filenames.append(img_path.name)
    
    if not batch_images:
        continue
    
    X_batch = np.array(batch_images)
    # Predict probabilities for each class
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # predicted class index
    pred_probs = np.max(probs, axis=1)       # confidence for predicted class
    
    # Save results for each image
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

# === Save results to CSV ===
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SEGMENTED ResNet18 predictions saved: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(results_df)}")
    print("🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
else:
    print("❌ No segmented images processed!")

print("🎉 SEGMENTED ResNet18 COMPLETE!")
