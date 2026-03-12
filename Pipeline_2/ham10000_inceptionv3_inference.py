import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =====================================================
# Paths
# =====================================================
# Directory containing segmented images
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")
# CSV file where predictions will be saved
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_inceptionv3_segmented_predictions_p2.csv"

# Image input size (InceptionV3 can work with 224x224 for fine-tuning)
IMG_SIZE = (224, 224)
# Batch size for prediction
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
# Load InceptionV3 model
# =====================================================
print("🔄 Loading InceptionV3...")

# Load pretrained InceptionV3 without top layer
base_model = InceptionV3(
    weights="imagenet",         # Use ImageNet pretrained weights
    include_top=False,          # Remove default classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input shape
)

# Add custom classification head for 7 classes
x = GlobalAveragePooling2D()(base_model.output)  # Pooling layer to reduce spatial dimensions
x = Dense(512, activation="relu")(x)             # Fully connected layer with 512 units
x = Dense(256, activation="relu")(x)             # Fully connected layer with 256 units
outputs = Dense(7, activation="softmax")(x)     # Output layer with softmax for 7 classes

# Define final model
model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False  # Freeze all layers; no training

print("✅ InceptionV3 ready")

# HAM10000 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================
# Get all JPG and PNG files from segmented images directory
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Sort images numerically by ISIC ID
seg_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (InceptionV3)...")
results = []  # To store results

# Loop over images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="InceptionV3 Segmented"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Current batch

    batch_images = []     # Preprocessed images
    batch_filenames = []  # Corresponding filenames

    # Process each image in batch
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))       # Read image from disk
        if image is None:
            continue                            # Skip if image failed to load

        image = cv2.resize(image, IMG_SIZE)    # Resize to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

        # Preprocess image for InceptionV3
        image = tf.keras.applications.inception_v3.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)             # Add to batch
        batch_filenames.append(img_path.name)  # Keep filename

    if not batch_images:
        continue  # Skip empty batch

    # Convert batch to NumPy array and predict probabilities
    X_batch = np.array(batch_images)
    probs = model.predict(X_batch, verbose=0)  # Softmax probabilities for 7 classes

    # Get predicted class and confidence
    pred_classes = np.argmax(probs, axis=1)    # Class index with max probability
    pred_probs = np.max(probs, axis=1)         # Max probability value

    # Store predictions for each image
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            "filename": fname,                   # Image filename
            "pred_class_id": int(pred_class),    # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
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
    df = pd.DataFrame(results)             # Convert to DataFrame
    df.to_csv(OUTPUT_CSV, index=False)     # Save CSV

    print(f"\n✅ InceptionV3 predictions saved to:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())  # Count per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())     # Confidence stats: mean, min, max, etc.
else:
    print("❌ No images processed")

print("🎉 InceptionV3 SEGMENTED PIPELINE COMPLETE!")
