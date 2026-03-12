import os                          # Provides functions to interact with the operating system
import cv2                         # OpenCV library for image processing
import numpy as np                 # Numerical computing library for array operations
import pandas as pd                # Data manipulation and CSV handling
from pathlib import Path           # Object-oriented filesystem paths
from tqdm import tqdm              # Progress bar for loops
import tensorflow as tf            # TensorFlow deep learning framework
from tensorflow.keras.applications import Xception   # Import pretrained Xception model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model            # Keras functional model class

# =====================================================
# Paths
# =====================================================

# Directory containing segmented HAM10000 images
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")

# Path where prediction results will be saved as CSV
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_xception_segmented_predictions_p2.csv"

# Image size used as input for the CNN model
# Xception default is 299x299 but 224x224 reduces memory usage and computation
IMG_SIZE = (224, 224)

# Batch size used during prediction
BATCH_SIZE = 16

# =====================================================
# Utility: extract ISIC number (IMPORTANT: defined early)
# =====================================================

def extract_number(filename):
    """
    Extract numeric ID from an ISIC image filename.
    Example: ISIC_123456.jpg -> 123456
    This allows correct numeric sorting of dataset images.
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# =====================================================
# Load Xception
# =====================================================

print("🔄 Loading Xception...")

# Load the pretrained Xception model without the original classification layer
base_model = Xception(
    weights="imagenet",                 # Load weights pretrained on ImageNet dataset
    include_top=False,                  # Remove ImageNet classification head
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)   # Input shape of images
)

# Global average pooling converts feature maps into a feature vector
x = GlobalAveragePooling2D()(base_model.output)

# Fully connected layer to learn dataset-specific features
x = Dense(512, activation="relu")(x)

# Second dense layer to refine feature representation
x = Dense(256, activation="relu")(x)

# Final classification layer with softmax activation for 7 HAM10000 classes
outputs = Dense(7, activation="softmax")(x)

# Build final model combining base CNN and classification head
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze model weights to avoid training (used only for feature inference)
model.trainable = False

print("✅ Xception ready")

# List of HAM10000 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================

# Collect all JPG and PNG images from the segmented image directory
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images numerically by ISIC ID to maintain dataset order
seg_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (Xception)...")

# List to store prediction results
results = []

# Iterate over images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="Xception Segmented"):

    # Select current batch of image paths
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []       # List to store processed images
    batch_filenames = []    # List to store corresponding filenames

    # Process each image in the batch
    for img_path in batch_paths:

        # Read image using OpenCV
        image = cv2.imread(str(img_path))

        # Skip image if loading failed
        if image is None:
            continue

        # Resize image to CNN input size
        image = cv2.resize(image, IMG_SIZE)

        # Convert OpenCV BGR format to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Xception-specific preprocessing
        image = tf.keras.applications.xception.preprocess_input(
            image.astype(np.float32)
        )

        # Store processed image
        batch_images.append(image)

        # Store filename
        batch_filenames.append(img_path.name)

    # Skip batch if no valid images were processed
    if not batch_images:
        continue

    # Convert image batch into NumPy array
    X_batch = np.array(batch_images)

    # Predict class probabilities for batch
    probs = model.predict(X_batch, verbose=0)

    # Get predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Get prediction confidence (maximum probability)
    pred_probs = np.max(probs, axis=1)

    # Store prediction results for each image
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({

            "filename": fname,                          # Image filename
            "pred_class_id": int(pred_class),           # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
            "pred_confidence": float(pred_probs[j]),    # Prediction confidence

            # Probability distribution across all classes
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

# Check if predictions were generated
if results:

    # Convert results list to DataFrame
    df = pd.DataFrame(results)

    # Save predictions to CSV file
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Xception predictions saved:")
    print(f"   {OUTPUT_CSV}")

    # Display number of processed images
    print(f"📊 Images processed: {len(df)}")

    # Show distribution of predicted classes
    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())

    # Show statistical summary of prediction confidence
    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())

else:
    print("❌ No images processed")

# Final message indicating pipeline completion
print("🎉 XCEPTION SEGMENTED PIPELINE COMPLETE!")
