import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# === Paths to segmented images and output CSV ===
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")  # folder with segmented images
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_resnet101_segmented_predictions_p2.csv"  # CSV to save predictions
IMG_SIZE = (224, 224)  # standard input size for ResNet
BATCH_SIZE = 32  # batch size for inference

# === Helper: extract numeric ID from ISIC filename for sorting ===
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)  # regex to get digits after "ISIC_"
    return int(m.group(1)) if m else float('inf')  # return integer or infinity if not found

# === Bottleneck Block used in ResNet ===
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

    # adjust shortcut if dimensions mismatch
    if downsample or stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters * 4, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # add skip connection
    x = Add()([shortcut, x])
    x = Activation('relu')(x)  # final activation
    return x

# === ResNet101 Architecture ===
def ResNet101(input_shape=(224, 224, 3), num_classes=7):
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

    # Conv4_x: ResNet101 has 23 bottleneck blocks in this stage
    x = bottleneck_block(x, 256, stride=2, downsample=True)  # first block
    for _ in range(22):  # remaining 22 blocks
        x = bottleneck_block(x, 256)

    # Conv5_x
    x = bottleneck_block(x, 512, stride=2, downsample=True)
    x = bottleneck_block(x, 512)
    x = bottleneck_block(x, 512)

    # Head: global pooling + dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # softmax for multi-class classification

    return Model(inputs, outputs)

# === Load the ResNet101 model ===
print("🔄 Loading ResNet101...")
model = ResNet101()
model.trainable = False  # freeze weights for inference
print("✅ ResNet101 ready")

# 7 HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === Load segmented images ===
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
seg_files.sort(key=lambda x: extract_number(x.name))  # sort numerically
print(f"📁 Segmented images found: {len(seg_files)}")

# === Predict on segmented images ===
print("🚀 Segment-aware predictions...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet101 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load, resize, and preprocess images
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # skip if reading fails

        image = cv2.resize(image, IMG_SIZE)  # resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR → RGB
        processed = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))  # ResNet preprocessing
        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # skip empty batch

    X_batch = np.array(batch_images)

    # Predict probabilities for batch
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # predicted class
    pred_probs = np.max(probs, axis=1)  # confidence of predicted class

    # Collect results
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
    print(f"\n✅ SEGMENTED ResNet101 saved: {OUTPUT_CSV}")
    print(f"📊 Images: {len(results_df)}")
    print("🎯 Distribution:")
    print(results_df['pred_class_name'].value_counts())
else:
    print("❌ No segmented images!")

print("🎉 SEGMENTED ResNet101 COMPLETE!")
