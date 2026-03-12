import os                          # Provides operating system utilities
import cv2                         # OpenCV library for image loading and processing
import numpy as np                 # Numerical computing library for array operations
import pandas as pd                # Data analysis library used for tables and CSV files
from pathlib import Path           # Object-oriented path management for files and folders
from tqdm import tqdm              # Progress bar for loops
import tensorflow as tf            # TensorFlow deep learning framework
from tensorflow.keras.applications import MobileNet   # Import pretrained MobileNetV1 architecture
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers used to build classification head
from tensorflow.keras.models import Model              # Keras model constructor

# =====================================================
# Paths
# =====================================================

# Directory containing segmented HAM10000 images
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")

# Path where the prediction results will be saved
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_mobilenetv1_segmented_predictions_p2.csv"

# Input image size required by MobileNet
IMG_SIZE = (224, 224)

# Batch size for prediction
# MobileNet is lightweight → larger batch sizes are safe
BATCH_SIZE = 32

# =====================================================
# Utility: extract ISIC number (must be defined early)
# =====================================================

def extract_number(filename):
    """
    Extract the numeric identifier from an ISIC image filename.
    Example:
        ISIC_123456.jpg → 123456
    This allows images to be sorted numerically instead of alphabetically.
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# =====================================================
# Load MobileNetV1
# =====================================================

print("🔄 Loading MobileNetV1...")

# Load MobileNet pretrained on ImageNet without the original classifier
base_model = MobileNet(
    weights="imagenet",                    # Load pretrained weights
    include_top=False,                     # Remove ImageNet classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Define input image dimensions
)

# Global Average Pooling converts feature maps into a single feature vector
x = GlobalAveragePooling2D()(base_model.output)

# First dense layer to learn HAM10000-specific patterns
x = Dense(256, activation="relu")(x)

# Second dense layer for deeper feature representation
x = Dense(128, activation="relu")(x)

# Final classification layer for the 7 HAM10000 lesion classes
outputs = Dense(7, activation="softmax")(x)

# Build final model by combining base CNN and classification head
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze model parameters since we only perform inference
model.trainable = False

print("✅ MobileNetV1 ready")

# HAM10000 lesion class labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================

# Collect all JPG and PNG images from the segmented image directory
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images numerically according to their ISIC ID
seg_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (MobileNetV1)...")

# List used to store prediction results
results = []

# Process images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="MobileNetV1 Segmented"):

    # Select batch of image paths
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []        # List to store processed images
    batch_filenames = []     # List to store filenames

    # Process each image in the batch
    for img_path in batch_paths:

        # Load image from disk
        image = cv2.imread(str(img_path))

        # Skip if image loading fails
        if image is None:
            continue

        # Resize image to match CNN input size
        image = cv2.resize(image, IMG_SIZE)

        # Convert OpenCV format from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MobileNet-specific preprocessing
        image = tf.keras.applications.mobilenet.preprocess_input(
            image.astype(np.float32)
        )

        # Store processed image
        batch_images.append(image)

        # Store filename
        batch_filenames.append(img_path.name)

    # Skip batch if no valid images were processed
    if not batch_images:
        continue

    # Convert list of images to NumPy array
    X_batch = np.array(batch_images)

    # Perform prediction and obtain probability distribution
    probs = model.predict(X_batch, verbose=0)

    # Determine predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Determine prediction confidence (maximum probability)
    pred_probs = np.max(probs, axis=1)

    # Store predictions for each image
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({

            "filename": fname,                          # Image filename
            "pred_class_id": int(pred_class),           # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
            "pred_confidence": float(pred_probs[j]),    # Prediction confidence

            # Probability values for each class
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

# Check if predictions exist
if results:

    # Convert results list into a pandas DataFrame
    df = pd.DataFrame(results)

    # Save predictions to CSV file
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ MobileNetV1 predictions saved:")
    print(f"   {OUTPUT_CSV}")

    # Print number of processed images
    print(f"📊 Images processed: {len(df)}")

    # Show predicted class distribution
    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())

    # Show statistical summary of prediction confidence
    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())

else:
    print("❌ No images processed")

# Indicate that the pipeline finished successfully
print("🎉 MOBILENETV1 SEGMENTED PIPELINE COMPLETE!")
