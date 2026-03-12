import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# === Helper function to extract numeric ID from ISIC filename for sorting ===
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)  # Match ISIC_XXXXX pattern
    return int(m.group(1)) if m else float('inf')  # Return number or infinity if not found

# === Paths and settings ===
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")  # segmented images dir
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_resnet50_segmented_predictions_p2.csv"  # output CSV
IMG_SIZE = (224, 224)  # standard input size for ResNet50
BATCH_SIZE = 32  # batch size for prediction

# === Bottleneck Block (used in ResNet50) ===
def bottleneck_block(x, filters, stride=1, downsample=False):
    shortcut = x  # save input for skip connection

    # 1x1 convolution
    x = Conv2D(filters, 1, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 convolution
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 convolution
    x = Conv2D(filters * 4, 1, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if dimensions mismatch
    if downsample or stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters * 4, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add skip connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)  # final activation
    return x

# === ResNet50 Architecture ===
def ResNet50(input_shape=(224, 224, 3), num_classes=7):
    inputs = Input(shape=input_shape)

    # Initial convolution + maxpool
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    # Conv2_x
    x = bottleneck_block(x, 64, stride=1, downsample=True)
    x = bottleneck_block(x, 64)
    x = bottleneck_block(x, 64)

    # Conv3_x
    x = bottleneck_block(x, 128, stride=2, downsample=True)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)
    x = bottleneck_block(x, 128)

    # Conv4_x
    x = bottleneck_block(x, 256, stride=2, downsample=True)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)
    x = bottleneck_block(x, 256)

    # Conv5_x
    x = bottleneck_block(x, 512, stride=2, downsample=True)
    x = bottleneck_block(x, 512)
    x = bottleneck_block(x, 512)

    # Head
    x = GlobalAveragePooling2D()(x)  # reduce to vector
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # softmax for 7 classes

    return Model(inputs, outputs)

# === Load the ResNet50 model ===
print("🔄 Loading ResNet50...")
model = ResNet50()
model.trainable = False  # freeze weights for inference
print("✅ ResNet50 ready")

# 7 classes of HAM10000
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === Collect segmented images ===
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
seg_files.sort(key=lambda x: extract_number(x.name))  # sort numerically
print(f"📁 Segmented images found: {len(seg_files)}")

# === Prediction loop ===
print("🚀 Segment-aware predictions...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet50 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load, resize, and preprocess images
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # skip if failed to read

        image = cv2.resize(image, IMG_SIZE)  # resize to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR → RGB
        processed = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))  # ResNet preprocessing
        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # skip empty batch

    X_batch = np.array(batch_images)  # convert to numpy array

    # Predict probabilities for batch
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # class with max probability
    pred_probs = np.max(probs, axis=1)  # confidence of predicted class

    # Collect predictions
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
    results_df.to_csv(OUTPUT_CSV, index=False)  # write CSV
    print(f"\n✅ SEGMENTED ResNet50 saved: {OUTPUT_CSV}")
    print(f"📊 Images: {len(results_df)}")
    print("🎯 Distribution:")
    print(results_df['pred_class_name'].value_counts())
else:
    print("❌ No segmented images!")

print("🎉 SEGMENTED ResNet50 COMPLETE!")
