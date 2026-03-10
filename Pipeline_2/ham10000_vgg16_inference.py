"""
Title: ham10000_vgg16_inference.py
Description:
This script performs **7-class skin lesion classification** on processed HAM10000 images 
using a pre-trained VGG16 model (ImageNet weights). It loads all images from a specified 
folder, preprocesses them, runs batch predictions, and saves the results in a CSV file. 
The output includes predicted class IDs, names, confidence scores, and probabilities for 
all 7 classes. It also prints class distributions and confidence statistics for analysis.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# ================================
# PATHS & SETTINGS
# ================================
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_images_all")  # Folder of processed images
OUT_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_vgg16_7class_predictions_p2.csv"

IMG_SIZE = (224, 224)  # VGG16 input size
BATCH_SIZE = 32        # Adjust based on GPU memory

# ================================
# HELPER FUNCTION: SORT IMAGES BY ISIC NUMBER
# ================================
def extract_number(filename):
    """
    Extract numeric ID from ISIC filename for stable sorting.
    Example: ISIC_0024306.jpg → 24306
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else 10**12

# ================================
# LOAD IMAGE FILE LIST
# ================================
proc_files = (
    list(PROC_DIR.glob("*.jpg")) +
    list(PROC_DIR.glob("*.png")) +
    list(PROC_DIR.glob("*.jpeg"))
)
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Found {len(proc_files)} processed images")

# ================================
# LOAD PRE-TRAINED VGG16 MODEL (7-CLASS)
# ================================
base_model = VGG16(
    weights="imagenet",
    include_top=False,  # Remove final classification layers
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add custom classification layers for 7 classes
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.trainable = False  # Freeze weights (we only want inference)
print("✅ VGG16 7-class model loaded (ImageNet weights)")

# ================================
# CLASS NAMES
# ================================
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # HAM10000 classes
class_map = {i: name for i, name in enumerate(class_names)}

# ================================
# BATCH PREDICTION
# ================================
results = []

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="🔍 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # Skip unreadable images

        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image.astype(np.float32))

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    X_batch = np.array(batch_images)

    # Predict probabilities for all 7 classes
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)   # Index of max probability
    pred_conf = np.max(probs, axis=1)         # Confidence score

    for fname, prob, cls_id, conf in zip(batch_filenames, probs, pred_classes, pred_conf):
        results.append({
            "filename": fname,
            "pred_class_id": int(cls_id),
            "pred_class_name": class_names[cls_id],
            "pred_confidence": float(conf),
            "prob_nv": float(prob[0]),
            "prob_mel": float(prob[1]),
            "prob_bkl": float(prob[2]),
            "prob_bcc": float(prob[3]),
            "prob_akiec": float(prob[4]),
            "prob_vasc": float(prob[5]),
            "prob_df": float(prob[6]),
        })

# ================================
# SAVE RESULTS TO CSV
# ================================
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Predictions saved to:")
    print(OUTPUT_CSV)

    print("\n📊 Class distribution:")
    print(df["pred_class_name"].value_counts())

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
else:
    print("❌ No images processed")

print("\n🎉 DONE – VGG16 7-class inference completed")
