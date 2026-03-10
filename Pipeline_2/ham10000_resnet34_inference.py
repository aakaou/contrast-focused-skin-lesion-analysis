import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# === Helper: extract numeric ID from ISIC filename for proper sorting ===
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)   # Look for numeric part after 'ISIC_'
    return int(m.group(1)) if m else float('inf')  # Return large number if no match

# === Paths - Segmented Images ===
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")  # Folder with segmented images
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_resnet34_segmented_predictions_p2.csv"  # Output CSV path
IMG_SIZE = (224, 224)  # Resize images to 224x224 for ResNet34
BATCH_SIZE = 32        # Number of images per batch

# === ResNet34 Basic Block Function ===
def basic_block(x, filters, stride=1, downsample=False):
    shortcut = x  # Save original input for skip connection
    x = Conv2D(filters, 3, strides=stride, padding='same')(x)  # First conv layer
    x = BatchNormalization()(x)  # Batch norm for stability
    x = Activation('relu')(x)    # ReLU activation
    x = Conv2D(filters, 3, padding='same')(x)  # Second conv layer
    x = BatchNormalization()(x)  # Batch norm

    # Adjust shortcut if downsampling is needed
    if downsample or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])  # Add skip connection
    x = Activation('relu')(x)  # Final ReLU
    return x

# === Build ResNet34 Architecture ===
def ResNet34(input_shape=(224, 224, 3), num_classes=7):
    inputs = Input(shape=input_shape)  # Input layer
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)  # Initial conv layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)  # Initial max pooling

    # Layer1: 64 filters, 3 basic blocks
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 64)

    # Layer2: 128 filters, 4 blocks, first downsampling
    x = basic_block(x, 128, stride=2, downsample=True)
    x = basic_block(x, 128)
    x = basic_block(x, 128)
    x = basic_block(x, 128)

    # Layer3: 256 filters, 6 blocks, first downsampling
    x = basic_block(x, 256, stride=2, downsample=True)
    for _ in range(5):
        x = basic_block(x, 256)

    # Layer4: 512 filters, 3 blocks, first downsampling
    x = basic_block(x, 512, stride=2, downsample=True)
    x = basic_block(x, 512)
    x = basic_block(x, 512)

    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(512, activation='relu')(x)  # Fully connected layer
    x = Dense(256, activation='relu')(x)  # Fully connected layer
    outputs = Dense(num_classes, activation='softmax')(x)  # Final softmax for 7 classes
    return Model(inputs, outputs)  # Return the model

# === Initialize ResNet34 ===
print("🔄 Loading ResNet34...")
model = ResNet34()        # Build the model
model.trainable = False   # Freeze weights for inference
print("✅ ResNet34 ready")

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === Load segmented images ===
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Collect image paths
seg_files.sort(key=lambda x: extract_number(x.name))  # Sort by numeric ID
print(f"📁 Segmented images found: {len(seg_files)}")

# === Predict on segmented images in batches ===
print("🚀 Segment-aware predictions...")
results = []  # Store predictions

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet34 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Select batch

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Load image
        if image is None:
            continue

        image = cv2.resize(image, IMG_SIZE)  # Resize to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
        processed = tf.keras.applications.resnet.preprocess_input(image.astype(np.float32))  # Preprocess for ResNet
        batch_images.append(processed)  # Add to batch
        batch_filenames.append(img_path.name)  # Track filename

    if not batch_images:  # Skip empty batches
        continue

    X_batch = np.array(batch_images)  # Convert batch to numpy array
    probs = model.predict(X_batch, verbose=0)  # Predict probabilities
    pred_classes = np.argmax(probs, axis=1)  # Predicted class index
    pred_probs = np.max(probs, axis=1)       # Confidence score

    # Store results for each image
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

# === Save predictions to CSV ===
if results:
    results_df = pd.DataFrame(results)  # Convert list to DataFrame
    results_df.to_csv(OUTPUT_CSV, index=False)  # Save to CSV
    print(f"\n✅ SEGMENTED ResNet34 predictions saved: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(results_df)}")
    print("🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())  # Show counts per class
else:
    print("❌ No segmented images processed!")

# === Finish message ===
print("🎉 SEGMENTED ResNet34 COMPLETE!")  # Script finished
