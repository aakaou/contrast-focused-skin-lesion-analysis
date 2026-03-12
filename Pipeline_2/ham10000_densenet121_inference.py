import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =====================================================
# Paths
# =====================================================
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")  # folder with segmented images
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_densenet121_segmented_predictions_p2.csv"  # output CSV

IMG_SIZE = (224, 224)  # DenseNet121 input size
BATCH_SIZE = 32  # batch size for predictions

# =====================================================
# Utility: extract numeric ID from ISIC filename
# =====================================================
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)  # find number in 'ISIC_XXXXXXX'
    return int(m.group(1)) if m else float('inf')  # return int or infinity if not found

# =====================================================
# Load DenseNet121 (pretrained on ImageNet)
# =====================================================
print("🔄 Loading DenseNet121...")

# Base model without top classification layer
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Add custom head for 7-class HAM10000 classification
x = GlobalAveragePooling2D()(base_model.output)  # global pooling
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)  # final softmax for 7 classes

model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False  # freeze all weights

print("✅ DenseNet121 ready")

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # collect jpg/png
seg_files.sort(key=lambda x: extract_number(x.name))  # sort by numeric ID
print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (DenseNet121)...")
results = []

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="DenseNet121 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # slice batch

    batch_images = []
    batch_filenames = []

    # Load and preprocess images
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # skip if cannot read

        image = cv2.resize(image, IMG_SIZE)  # resize to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB

        # DenseNet preprocessing
        image = tf.keras.applications.densenet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # skip empty batch

    # Predict probabilities
    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)

    # Get predicted class and max probability
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

    print(f"\n✅ DenseNet121 predictions saved to: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")
    print("🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
else:
    print("❌ No images processed")

print("🎉 DenseNet121 SEGMENTED PIPELINE COMPLETE!")
