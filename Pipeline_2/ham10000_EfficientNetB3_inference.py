import cv2                                  # OpenCV for image reading and preprocessing
import numpy as np                          # NumPy for numerical arrays and calculations
import pandas as pd                         # Pandas for DataFrame handling and CSV output
from pathlib import Path                    # Pathlib to handle filesystem paths
from tqdm import tqdm                       # TQDM for progress bars in loops
import tensorflow as tf                     # TensorFlow for deep learning models
from tensorflow.keras.applications import EfficientNetB3  # Import EfficientNetB3 pretrained model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model   # Keras Model class for building full models

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing the segmented HAM10000 images

OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_efficientnetb3_segmented_predictions_p2.csv"
# Path to save the CSV with predictions

IMG_SIZE = (300, 300)                       # Default input size for EfficientNetB3
BATCH_SIZE = 32                             # Number of images processed per batch

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function to get numeric ID from filename
    import re                                # Regular expressions module
    m = re.search(r'ISIC_(\d+)', filename)   # Search for "ISIC_XXXXXX" pattern
    return int(m.group(1)) if m else float('inf')  
    # Return integer if found, else infinity for safe sorting

# =====================================================
# Load EfficientNetB3
# =====================================================

print("🔄 Loading EfficientNetB3...")        # Inform user that model is loading

base_model = EfficientNetB3(                 # Load the EfficientNetB3 base
    weights="imagenet",                      # Use pretrained ImageNet weights
    include_top=False,                       # Exclude the default classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image shape: 300x300 RGB
)

x = GlobalAveragePooling2D()(base_model.output)
# Pool convolutional features into a single vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 neurons
x = Dense(256, activation="relu")(x)         # Fully connected layer with 256 neurons
outputs = Dense(7, activation="softmax")(x)  # Output layer for 7 HAM10000 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Combine base and classifier into one model

model.trainable = False                      # Freeze weights for inference

print("✅ EfficientNetB3 ready")             # Notify that model is ready

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of 7 skin lesion classes

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all .jpg and .png images from directory

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images numerically by ISIC ID

print(f"📁 Segmented images found: {len(seg_files)}")
# Print total number of images found

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB3)...")
results = []                                 # List to store predictions

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB3"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Current batch of image paths

    batch_images = []                         # List to store preprocessed images
    batch_filenames = []                      # List to store corresponding filenames

    for img_path in batch_paths:              # Loop over each image in the batch
        image = cv2.imread(str(img_path))     # Read image using OpenCV
        if image is None:                     # Skip if failed to read
            continue

        image = cv2.resize(image, IMG_SIZE)   # Resize to 300x300
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert from BGR (OpenCV) to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        # Preprocess image for EfficientNet

        batch_images.append(image)            # Append preprocessed image
        batch_filenames.append(img_path.name) # Append filename

    if not batch_images:                      # Skip empty batches
        continue

    X_batch = np.array(batch_images)          # Convert list to NumPy array
    probs = model.predict(X_batch, verbose=0) # Predict probabilities

    pred_classes = np.argmax(probs, axis=1)   # Predicted class index
    pred_probs = np.max(probs, axis=1)        # Confidence score of prediction

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
# Store predictions and per-class probabilities

# =====================================================
# Save results
# =====================================================

if results:                                  # If predictions exist
    df = pd.DataFrame(results)               # Convert list of dicts to DataFrame
    df.to_csv(OUTPUT_CSV, index=False)       # Save DataFrame to CSV

    print(f"\n✅ EfficientNetB3 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Print number of predictions per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Print statistics on prediction confidence
else:
    print("❌ No images processed")          # Notify if no images processed

print("🎉 EFFICIENTNETB3 SEGMENTED PIPELINE COMPLETE!")
# Indicate that the entire pipeline finished successfully
