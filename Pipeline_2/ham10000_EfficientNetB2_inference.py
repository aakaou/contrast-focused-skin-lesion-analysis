import cv2                                  # OpenCV for image reading and preprocessing
import numpy as np                          # NumPy for numerical operations and array manipulation
import pandas as pd                         # Pandas for DataFrame operations and saving CSV
from pathlib import Path                    # Pathlib for handling filesystem paths
from tqdm import tqdm                       # Progress bar for loops
import tensorflow as tf                     # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB2  # EfficientNetB2 pretrained model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model   # Model class to build full Keras models

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented HAM10000 lesion images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb2_segmented_predictions_p2.csv"
# CSV file path to save prediction results

IMG_SIZE = (260, 260)                       # Input image size for EfficientNetB2
BATCH_SIZE = 32                             # Number of images to process per batch

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function to extract numeric ID from filename
    import re                                # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)   # Look for "ISIC_XXXXXX" pattern
    return int(m.group(1)) if m else float('inf')  
    # Return the numeric ID or infinity if not found (for safe sorting)

# =====================================================
# Load EfficientNetB2
# =====================================================

print("🔄 Loading EfficientNetB2...")        # Notify user that model is loading

base_model = EfficientNetB2(                 # Load EfficientNetB2 architecture
    weights="imagenet",                      # Use pretrained ImageNet weights
    include_top=False,                       # Exclude the default classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Set input shape (260x260 RGB)
)

x = GlobalAveragePooling2D()(base_model.output)
# Convert convolutional features to a single feature vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 neurons
x = Dense(256, activation="relu")(x)         # Fully connected layer with 256 neurons

outputs = Dense(7, activation="softmax")(x)  # Output layer with 7 neurons for HAM10000 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Build the full model combining base and classification head

model.trainable = False                      # Freeze all layers for inference only

print("✅ EfficientNetB2 ready")              # Confirmation that model is loaded

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of 7 skin lesion classes

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all images (jpg or png) from the directory

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images numerically by ISIC ID

print(f"📁 Segmented images found: {len(seg_files)}")
# Show how many segmented images were found

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB2)...")
results = []                                 # List to store predictions

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB2"):
# Loop through images in batches with a progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]  
    # Current batch of image paths

    batch_images = []                         # To store processed images
    batch_filenames = []                      # To store corresponding filenames

    for img_path in batch_paths:              # Loop over each image
        image = cv2.imread(str(img_path))     # Read image
        if image is None:                     # Skip if failed to read
            continue

        image = cv2.resize(image, IMG_SIZE)   # Resize image to 260x260
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert BGR (OpenCV default) to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        # Preprocess according to EfficientNet requirements

        batch_images.append(image)            # Add preprocessed image
        batch_filenames.append(img_path.name) # Add filename

    if not batch_images:                      # Skip empty batches
        continue

    X_batch = np.array(batch_images)          # Convert list to NumPy array
    probs = model.predict(X_batch, verbose=0) # Predict probabilities

    pred_classes = np.argmax(probs, axis=1)   # Predicted class index
    pred_probs = np.max(probs, axis=1)        # Confidence of prediction

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
# Store predictions and probabilities in a structured format

# =====================================================
# Save results
# =====================================================

if results:                                  # If any predictions were made
    df = pd.DataFrame(results)               # Convert to pandas DataFrame
    df.to_csv(OUTPUT_CSV, index=False)       # Save as CSV

    print(f"\n✅ EfficientNetB2 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Show how many images were predicted per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Show confidence stats (mean, min, max, etc.)
else:
    print("❌ No images processed")          # If no images were successfully predicted

print("🎉 EFFICIENTNETB2 SEGMENTED PIPELINE COMPLETE!")
# Indicate pipeline completion
