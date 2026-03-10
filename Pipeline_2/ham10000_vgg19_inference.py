"""
This Python script performs batch classification of HAM10000 skin lesion images using a pre-trained VGG19 model adapted for 7 classes. 
Images are loaded from a specified directory, resized to 224×224, and preprocessed according to the VGG19 requirements. Predictions are done in batches for efficiency, 
returning both the predicted class and the confidence score, as well as the probability for each of the seven classes. All results are stored in a CSV file, 
along with basic statistics such as prediction distribution and confidence summary. This workflow allows rapid evaluation of VGG19’s performance on a large skin lesion 
dataset while keeping memory usage manageable.
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf

# === Helper: extract numeric ID from ISIC filename for sorting ===
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# === Paths and settings ===
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_images_all")
OUT_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_vgg19_7class_predictions_p2.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === Load VGG19 and add custom classification layers ===
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.trainable = False  # Freeze weights for inference
print("✅ VGG19 7-class model loaded")

# 7 classes in HAM10000
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# === Collect and sort image files ===
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Found {len(proc_files)} images")

# === Batch prediction loop ===
results = []

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="VGG19 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load & preprocess images
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed = preprocess_input(image.astype(np.float32))

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue

    X_batch = np.array(batch_images)

    # Predict probabilities
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)

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
            'prob_df': float(prob[6]),
        })

# === Save results to CSV ===
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ VGG19 predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

print("✅ VGG19 pipeline finished successfully!")
