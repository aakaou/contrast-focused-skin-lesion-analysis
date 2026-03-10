import cv2                          # OpenCV library for reading and processing images
import numpy as np                  # NumPy for array and numerical operations
import pandas as pd                 # Pandas for handling dataframes and saving CSV
from pathlib import Path            # Pathlib for convenient file path operations
from tqdm import tqdm               # TQDM for progress bars
import tensorflow as tf             # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB6  # Pretrained EfficientNetB6
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for custom classifier
from tensorflow.keras.models import Model  # Model class for building the model

# =====================================================
# Paths and configuration
# =====================================================
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb6_segmented_predictions_p2.csv"
# Path to save the CSV predictions

IMG_SIZE = (528, 528)  # Default input size for EfficientNetB6
BATCH_SIZE = 16        # Batch size for inference (smaller due to large image size)

# =====================================================
# Utility function to extract ISIC number from filenames
# =====================================================
def extract_number(filename):
    import re  # Import regex module
    m = re.search(r'ISIC_(\d+)', filename)  # Look for "ISIC_<number>" in the filename
    return int(m.group(1)) if m else float('inf')  # Return number, or infinity if not found

# =====================================================
# Load EfficientNetB6
# =====================================================
print("🔄 Loading EfficientNetB6...")

base_model = EfficientNetB6(
    weights="imagenet",                      # Load pretrained weights from ImageNet
    include_top=False,                       # Remove original top classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image shape (height, width, channels)
)

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)  # Pool spatial features into a single vector
x = Dense(512, activation="relu")(x)            # Fully connected layer with 512 units and ReLU
x = Dense(256, activation="relu")(x)            # Fully connected layer with 256 units and ReLU
outputs = Dense(7, activation="softmax")(x)     # Output layer for 7 classes with softmax

# Create final model
model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False  # Freeze all weights (no training, inference only)

print("✅ EfficientNetB6 ready")

# =====================================================
# Define class names
# =====================================================
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  
# HAM10000 dataset class names

# =====================================================
# Load segmented images
# =====================================================
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  
# Find all JPG and PNG images in the directory

seg_files.sort(key=lambda x: extract_number(x.name))  
# Sort images by ISIC number for consistent order

print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (EfficientNetB6)...")
results = []  # List to store prediction results

# Iterate over images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB6"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Get current batch

    batch_images = []       # List to hold preprocessed images
    batch_filenames = []    # List to hold corresponding filenames

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image from disk
        if image is None:
            continue  # Skip if the image can't be read

        image = cv2.resize(image, IMG_SIZE)  # Resize to EfficientNetB6 input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB

        # Preprocess image for EfficientNetB6
        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)          # Add image to batch
        batch_filenames.append(img_path.name)  # Save filename for tracking

    if not batch_images:
        continue  # Skip empty batch

    X_batch = np.array(batch_images)         # Convert batch to NumPy array
    probs = model.predict(X_batch, verbose=0)  # Predict probabilities for all images in batch

    pred_classes = np.argmax(probs, axis=1)  # Get class with highest probability
    pred_probs = np.max(probs, axis=1)       # Get highest probability (confidence)

    # Store results
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

# =====================================================
# Save predictions to CSV
# =====================================================
if results:
    df = pd.DataFrame(results)           # Convert list of dicts to DataFrame
    df.to_csv(OUTPUT_CSV, index=False)   # Save DataFrame to CSV

    print(f"\n✅ EfficientNetB6 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())  # Count of each predicted class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())     # Min, max, mean, std of confidence scores
else:
    print("❌ No images processed")  # Alert if no images were processed

print("🎉 EFFICIENTNETB6 SEGMENTED PIPELINE COMPLETE!")  # Done
