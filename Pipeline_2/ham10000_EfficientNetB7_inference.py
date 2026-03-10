import cv2                          # OpenCV for reading and processing images
import numpy as np                  # NumPy for numerical operations and array handling
import pandas as pd                 # Pandas for handling tabular data (DataFrames)
from pathlib import Path            # Pathlib for convenient file path operations
from tqdm import tqdm               # TQDM for progress bars during loops
import tensorflow as tf             # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB7  # Pretrained EfficientNetB7 model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for custom classifier
from tensorflow.keras.models import Model  # Model class to create the final network

# =====================================================
# Paths and configuration
# =====================================================
PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb7_segmented_predictions_p2.csv"
# Path to save prediction results as CSV

IMG_SIZE = (600, 600)  # EfficientNetB7 default input size (height, width)
BATCH_SIZE = 8         # Batch size reduced due to large image size

# =====================================================
# Utility: extract ISIC number from filenames
# =====================================================
def extract_number(filename):
    import re  # Regular expressions module
    m = re.search(r'ISIC_(\d+)', filename)  # Match pattern "ISIC_<number>"
    return int(m.group(1)) if m else float('inf')  # Return number or infinity if not found

# =====================================================
# Load EfficientNetB7
# =====================================================
print("🔄 Loading EfficientNetB7...")

base_model = EfficientNetB7(
    weights="imagenet",                      # Load pretrained ImageNet weights
    include_top=False,                       # Remove the original top classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input shape (H, W, 3 channels)
)

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)  # Pool spatial features into a vector
x = Dense(512, activation="relu")(x)            # Dense layer with 512 units + ReLU
x = Dense(256, activation="relu")(x)            # Dense layer with 256 units + ReLU
outputs = Dense(7, activation="softmax")(x)     # Output layer with 7 classes + softmax

# Build the final model
model = Model(inputs=base_model.input, outputs=outputs)
model.trainable = False  # Freeze all layers; only inference

print("✅ EfficientNetB7 ready")

# =====================================================
# Define HAM10000 classes
# =====================================================
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # 7 skin lesion classes

# =====================================================
# Load segmented images
# =====================================================
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Find JPG & PNG images
seg_files.sort(key=lambda x: extract_number(x.name))  # Sort by ISIC number
print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================
print("🚀 Segment-aware predictions (EfficientNetB7)...")
results = []  # List to store prediction results

# Iterate through images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB7"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Current batch

    batch_images = []       # Preprocessed images for model
    batch_filenames = []    # Corresponding filenames

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image
        if image is None:
            continue  # Skip if cannot read

        image = cv2.resize(image, IMG_SIZE)           # Resize to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB

        # Preprocess image for EfficientNet
        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)            # Add to batch
        batch_filenames.append(img_path.name) # Save filename

    if not batch_images:
        continue  # Skip empty batch

    X_batch = np.array(batch_images)         # Convert batch list to NumPy array
    probs = model.predict(X_batch, verbose=0)  # Predict probabilities for batch

    pred_classes = np.argmax(probs, axis=1)  # Predicted class index
    pred_probs = np.max(probs, axis=1)       # Confidence of prediction

    # Store predictions
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
# Save results to CSV
# =====================================================
if results:
    df = pd.DataFrame(results)           # Convert results list to DataFrame
    df.to_csv(OUTPUT_CSV, index=False)   # Save DataFrame to CSV

    print(f"\n✅ EfficientNetB7 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())  # Count per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())     # Confidence statistics
else:
    print("❌ No images processed")  # Alert if no images processed

print("🎉 EFFICIENTNETB7 SEGMENTED PIPELINE COMPLETE!")  # Finished
