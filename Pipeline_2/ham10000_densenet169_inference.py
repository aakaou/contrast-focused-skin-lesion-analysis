import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =====================================================
# Paths
# =====================================================
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_densenet169_segmented_predictions_p2.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =====================================================
# Utility: extract ISIC number
# =====================================================
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# =====================================================
# Load DenseNet169 (ImageNet)
# =====================================================
print("🔄 Loading DenseNet169...")

# Base model without top layer
base_model = DenseNet169(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Custom head for 7 classes
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False

print("✅ DenseNet169 ready")

# HAM10000 classes
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
print("🚀 Segment-aware predictions (DenseNet169)...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="DenseNet169 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load and preprocess
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # DenseNet preprocessing
        image = tf.keras.applications.densenet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    # Predict probabilities
    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)

    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)

    # Save results
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

    print(f"\n✅ DenseNet169 predictions saved to: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")
    print("🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
else:
    print("❌ No images processed")

print("🎉 DenseNet169 SEGMENTED PIPELINE COMPLETE!")
