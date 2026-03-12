import cv2                                  # OpenCV for reading and processing images
import numpy as np                          # NumPy for arrays and numerical operations
import pandas as pd                         # Pandas for creating DataFrames and saving CSV
from pathlib import Path                    # Pathlib for handling filesystem paths
from tqdm import tqdm                       # TQDM for progress bars
import tensorflow as tf                     # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB4  # EfficientNetB4 pretrained model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classifier
from tensorflow.keras.models import Model   # Keras Model class

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented images

OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_efficientnetb4_segmented_predictions_p2.csv"
# Path to save predictions CSV

IMG_SIZE = (380, 380)                       # Default input size for EfficientNetB4
BATCH_SIZE = 32                             # Batch size for predictions

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function to extract ISIC numeric ID
    import re                                # Import regular expressions
    m = re.search(r'ISIC_(\d+)', filename)   # Match pattern "ISIC_XXXXXX"
    return int(m.group(1)) if m else float('inf')  
    # Return integer ID if found, else infinity for sorting safety

# =====================================================
# Load EfficientNetB4
# =====================================================

print("🔄 Loading EfficientNetB4...")        # Inform user

base_model = EfficientNetB4(
    weights="imagenet",                      # Use pretrained ImageNet weights
    include_top=False,                       # Exclude original classification head
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image shape: 380x380 RGB
)

x = GlobalAveragePooling2D()(base_model.output)
# Pool features to a vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 units
x = Dense(256, activation="relu")(x)         # Fully connected layer with 256 units
outputs = Dense(7, activation="softmax")(x)  # Output layer for 7 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Create the full model

model.trainable = False                      # Freeze model for inference

print("✅ EfficientNetB4 ready")             # Notify user

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of 7 classes

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Get all .jpg and .png images

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort by ISIC ID

print(f"📁 Segmented images found: {len(seg_files)}")
# Print number of images found

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB4)...")
results = []                                 # List to store predictions

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB4"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Get batch of file paths

    batch_images = []                         # List of preprocessed images
    batch_filenames = []                      # List of filenames

    for img_path in batch_paths:              # Loop over images
        image = cv2.imread(str(img_path))     # Read image
        if image is None:                     # Skip if failed to read
            continue

        image = cv2.resize(image, IMG_SIZE)   # Resize image to 380x380
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert BGR to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        # Preprocess image for EfficientNet

        batch_images.append(image)            # Add image to batch
        batch_filenames.append(img_path.name) # Add filename to batch

    if not batch_images:                      # Skip empty batches
        continue

    X_batch = np.array(batch_images)          # Convert list to NumPy array
    probs = model.predict(X_batch, verbose=0) # Predict class probabilities

    pred_classes = np.argmax(probs, axis=1)   # Get predicted class index
    pred_probs = np.max(probs, axis=1)        # Get prediction confidence

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
# Store all predictions with probabilities

# =====================================================
# Save results
# =====================================================

if results:                                  # Check if predictions exist
    df = pd.DataFrame(results)               # Convert to DataFrame
    df.to_csv(OUTPUT_CSV, index=False)       # Save CSV

    print(f"\n✅ EfficientNetB4 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Print number of predictions per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Print statistics on confidence scores
else:
    print("❌ No images processed")          # Notify if nothing processed

print("🎉 EFFICIENTNETB4 SEGMENTED PIPELINE COMPLETE!")
# Indicate pipeline finished
