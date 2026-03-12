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

# =====================================================
# Paths
# =====================================================
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")  # folder with segmented images
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_resnet152_segmented_predictions_p2.csv"  # save predictions here

IMG_SIZE = (224, 224)  # ResNet standard input size
BATCH_SIZE = 32  # batch size for prediction

# =====================================================
# Utility: extract numeric ID from ISIC filename
# =====================================================
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)  # regex captures number after 'ISIC_'
    return int(m.group(1)) if m else float('inf')  # return int or infinity if not found

# =====================================================
# Bottleneck block (used in ResNet architectures)
# =====================================================
def bottleneck_block(x, filters, stride=1, downsample=False):
    shortcut = x  # save input for skip connection

    # 1x1 convolution
    x = Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 convolution
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 convolution to expand channels
    x = Conv2D(filters * 4, 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # adjust shortcut if needed
    if downsample or stride != 1 or shortcut.shape[-1] != filters * 4:
        shortcut = Conv2D(filters * 4, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # add skip connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)  # final activation
    return x

# =====================================================
# ResNet152 Architecture
# =====================================================
def ResNet152(input_shape=(224, 224, 3), num_classes=7):
    inputs = Input(shape=input_shape)

    # Stem: initial convolution + maxpool
    x = Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Conv2_x: 3 bottleneck blocks
    x = bottleneck_block(x, 64, stride=1, downsample=True)
    for _ in range(2):
        x = bottleneck_block(x, 64)

    # Conv3_x: 8 bottleneck blocks
    x = bottleneck_block(x, 128, stride=2, downsample=True)
    for _ in range(7):
        x = bottleneck_block(x, 128)

    # Conv4_x: 36 bottleneck blocks (ResNet152 key difference from ResNet101)
    x = bottleneck_block(x, 256, stride=2, downsample=True)
    for _ in range(35):
        x = bottleneck_block(x, 256)

    # Conv5_x: 3 bottleneck blocks
    x = bottleneck_block(x, 512, stride=2, downsample=True)
    for _ in range(2):
        x = bottleneck_block(x, 512)

    # Head: global average pooling + dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # 7-class output

    return Model(inputs, outputs)

# =====================================================
# Load model
# =====================================================
print("🔄 Loading ResNet152...")
model = ResNet152()
model.trainable = False  # freeze weights
print("✅ ResNet152 ready")

# Class names for HAM10000
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
seg_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (ResNet152)...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet152 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load and preprocess images
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # skip unreadable image

        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # skip empty batch

    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)

    # Store results
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            "filename": fname,
            "pred_class_id": int(pred_class),
            "pred_class_name": class_names[pred_class],
            "pred_confidence": float(pred_probs[j]),
            "prob_nv": float(prob[0]),
            "prob_mel": float(prob[1]),
            "prob_bkl": float(prob[2]),
            "prob_bcc": float(prob[3]),
            "prob_akiec": float(prob[4]),
            "prob_vasc": float(prob[5]),
            "prob_df": float(prob[6]),
        })

# =====================================================
# Save results
# =====================================================
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ ResNet152 predictions saved to: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")
    print("🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
else:
    print("❌ No images processed")

print("🎉 ResNet152 SEGMENTED PIPELINE COMPLETE!")
