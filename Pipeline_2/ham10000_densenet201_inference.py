import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =====================================================
# Paths
# =====================================================
# Directory containing segmented images
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# CSV file where predictions will be saved
OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_densenet201_segmented_predictions_p2.csv"

# Image input size for the model
IMG_SIZE = (224, 224)
# Batch size for predictions
BATCH_SIZE = 32

# =====================================================
# Utility: extract ISIC number
# =====================================================
def extract_number(filename):
    """
    Extract numeric ID from ISIC filename for sorting.
    e.g., "ISIC_12345.jpg" -> 12345
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# =====================================================
# Load DenseNet201 (ImageNet pretrained)
# =====================================================
print("🔄 Loading DenseNet201...")

# Load DenseNet201 without top layer
base_model = DenseNet201(
    weights="imagenet",        # Use pretrained ImageNet weights
    include_top=False,         # Remove default classification layer
    input_shape=(224, 224, 3) # Input image shape
)

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output) # Reduce spatial dims
x = Dense(512, activation="relu")(x)            # Fully connected layer
x = Dense(256, activation="relu")(x)            # Another dense layer
outputs = Dense(7, activation="softmax")(x)    # 7-class softmax output

# Define final model
model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False  # Freeze all weights (no training)

print("✅ DenseNet201 ready")

# HAM10000 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================
# Find all JPG and PNG images in the directory
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Sort images numerically based on ISIC number
seg_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (DenseNet201)...")
results = []

# Loop over images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="DenseNet201 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Current batch

    batch_images = []     # Store preprocessed images
    batch_filenames = []  # Store filenames for results

    # Process each image in batch
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))       # Load image
        if image is None:
            continue                            # Skip if failed to read

        image = cv2.resize(image, IMG_SIZE)    # Resize to 224x224
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR->RGB

        # DenseNet-specific preprocessing
        image = tf.keras.applications.densenet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # Skip empty batches

    # Convert batch to NumPy array and predict
    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)  # Softmax probabilities

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)    # Index of max probability
    pred_probs = np.max(probs, axis=1)         # Max probability value

    # Store results for each image
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            "filename": fname,                   # Image filename
            "pred_class_id": int(pred_class),    # Class index
            "pred_class_name": class_names[pred_class], # Class label
            "pred_confidence": float(pred_probs[j]),    # Confidence
            "prob_nv": float(prob[0]),           # Probability for 'nv'
            "prob_mel": float(prob[1]),          # Probability for 'mel'
            "prob_bkl": float(prob[2]),          # Probability for 'bkl'
            "prob_bcc": float(prob[3]),          # Probability for 'bcc'
            "prob_akiec": float(prob[4]),        # Probability for 'akiec'
            "prob_vasc": float(prob[5]),         # Probability for 'vasc'
            "prob_df": float(prob[6]),           # Probability for 'df'
        })

# =====================================================
# Save results to CSV
# =====================================================
if results:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)  # Save as CSV

    print(f"\n✅ DenseNet201 predictions saved to:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())  # Count per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())     # Mean, min, max, etc.
else:
    print("❌ No images processed")

print("🎉 DenseNet201 SEGMENTED PIPELINE COMPLETE!")
