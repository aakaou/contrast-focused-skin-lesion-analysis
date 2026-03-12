import os                          # Provides operating system utilities
import cv2                         # OpenCV library for image processing
import numpy as np                 # Numerical computing library for arrays and matrices
import pandas as pd                # Data analysis library used for tables and CSV files
from pathlib import Path           # Object-oriented file system path handling
from tqdm import tqdm              # Progress bar for loops
import tensorflow as tf            # TensorFlow deep learning framework
from tensorflow.keras.applications import InceptionResNetV2  # Import pretrained InceptionResNetV2 model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model                      # Keras model constructor

# =====================================================
# Paths
# =====================================================

# Directory containing segmented HAM10000 images
PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")

# CSV file where prediction results will be stored
OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_inceptionresnetv2_segmented_predictions_p2.csv"

# Input image size for the CNN
# 224x224 reduces memory usage compared to the original 299x299
IMG_SIZE = (224, 224)

# Batch size used during prediction
# Smaller batch reduces GPU/CPU memory usage because the model is heavy
BATCH_SIZE = 16

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):
    """
    Extract the numeric ID from an ISIC image filename.
    Example: ISIC_123456.jpg → 123456
    This ensures images are sorted numerically rather than alphabetically.
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# =====================================================
# Load InceptionResNetV2
# =====================================================

print("🔄 Loading InceptionResNetV2...")

# Load pretrained InceptionResNetV2 model without the ImageNet classifier
base_model = InceptionResNetV2(
    weights="imagenet",                    # Load pretrained weights
    include_top=False,                     # Remove default ImageNet classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image dimensions
)

# Global average pooling converts feature maps to a 1D feature vector
x = GlobalAveragePooling2D()(base_model.output)

# Fully connected layer to learn HAM10000-specific patterns
x = Dense(512, activation="relu")(x)

# Second dense layer to refine learned features
x = Dense(256, activation="relu")(x)

# Final classification layer for the 7 HAM10000 classes
outputs = Dense(7, activation="softmax")(x)

# Build final model by combining base CNN and classification head
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze all weights because we only perform inference
model.trainable = False

print("✅ InceptionResNetV2 ready")

# HAM10000 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================

# Collect all image files from the segmented images directory
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images numerically based on ISIC ID
seg_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (InceptionResNetV2)...")

# List that will store prediction results
results = []

# Iterate over the images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="InceptionResNetV2 Segmented"):

    # Select current batch of image paths
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []        # List to store processed images
    batch_filenames = []     # List to store filenames

    # Process each image in the batch
    for img_path in batch_paths:

        # Read image from disk
        image = cv2.imread(str(img_path))

        # Skip if image cannot be loaded
        if image is None:
            continue

        # Resize image to the CNN input resolution
        image = cv2.resize(image, IMG_SIZE)

        # Convert OpenCV BGR format to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing specific to InceptionResNetV2
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(
            image.astype(np.float32)
        )

        # Store processed image
        batch_images.append(image)

        # Store filename
        batch_filenames.append(img_path.name)

    # Skip batch if no valid images
    if not batch_images:
        continue

    # Convert batch list into NumPy array
    X_batch = np.array(batch_images)

    # Predict class probabilities
    probs = model.predict(X_batch, verbose=0)

    # Determine predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Determine prediction confidence (maximum probability)
    pred_probs = np.max(probs, axis=1)

    # Save prediction results
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({

            "filename": fname,                          # Image filename
            "pred_class_id": int(pred_class),           # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
            "pred_confidence": float(pred_probs[j]),    # Prediction confidence

            # Probability values for all 7 classes
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

# If predictions were generated
if results:

    # Convert results list into DataFrame
    df = pd.DataFrame(results)

    # Save results as CSV file
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ InceptionResNetV2 predictions saved to:")
    print(f"   {OUTPUT_CSV}")

    # Print number of processed images
    print(f"📊 Images processed: {len(df)}")

    # Show distribution of predicted classes
    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())

    # Show statistics for prediction confidence
    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())

else:
    print("❌ No images processed")

# Indicate completion of the pipeline
print("🎉 InceptionResNetV2 SEGMENTED PIPELINE COMPLETE!")
