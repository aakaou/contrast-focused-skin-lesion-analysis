import cv2                                  # OpenCV: image reading & processing
import numpy as np                          # NumPy: array operations
import pandas as pd                         # Pandas: dataframes & CSV saving
from pathlib import Path                    # Path handling
from tqdm import tqdm                       # Progress bars
import tensorflow as tf                     # TensorFlow for DL
from tensorflow.keras.applications import EfficientNetB5  # Pretrained EfficientNetB5
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classifier
from tensorflow.keras.models import Model   # Model class for building network

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Folder containing the segmented images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb5_segmented_predictions_p2.csv"
# Output CSV file path for predictions

IMG_SIZE = (456, 456)                       # EfficientNetB5 default input size
BATCH_SIZE = 32                             # Batch size for inference

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)  # Regex to find ISIC number in filename
    return int(m.group(1)) if m else float('inf')
    # Return the numeric ID if found; else return infinity (used for sorting)

# =====================================================
# Load EfficientNetB5
# =====================================================

print("🔄 Loading EfficientNetB5...")

base_model = EfficientNetB5(
    weights="imagenet",                      # Pretrained on ImageNet
    include_top=False,                       # Remove original classification layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image shape
)

x = GlobalAveragePooling2D()(base_model.output)  # Pool features to vector
x = Dense(512, activation="relu")(x)            # Dense layer with 512 units
x = Dense(256, activation="relu")(x)            # Dense layer with 256 units
outputs = Dense(7, activation="softmax")(x)     # Output 7-class softmax

model = Model(inputs=base_model.input, outputs=outputs)  # Create final model
model.trainable = False                                 # Freeze weights

print("✅ EfficientNetB5 ready")

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Gather all JPG and PNG images

seg_files.sort(key=lambda x: extract_number(x.name))  
# Sort images by ISIC number

print(f"📁 Segmented images found: {len(seg_files)}")

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB5)...")
results = []  # List to store predictions

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB5"):
    batch_paths = seg_files[i:i + BATCH_SIZE]  # Slice batch

    batch_images = []       # List to store preprocessed images
    batch_filenames = []    # List to store filenames

    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image
        if image is None:
            continue                      # Skip if reading fails

        image = cv2.resize(image, IMG_SIZE)  # Resize image to 456x456
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # Convert from BGR to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )  
        # Preprocess image for EfficientNet

        batch_images.append(image)          # Append to batch
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # Skip empty batch

    X_batch = np.array(batch_images)       # Convert list to array
    probs = model.predict(X_batch, verbose=0)  # Predict probabilities

    pred_classes = np.argmax(probs, axis=1)  # Get predicted class index
    pred_probs = np.max(probs, axis=1)       # Get prediction confidence

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
# Append all predictions and class probabilities

# =====================================================
# Save results
# =====================================================

if results:
    df = pd.DataFrame(results)           # Convert to dataframe
    df.to_csv(OUTPUT_CSV, index=False)   # Save as CSV

    print(f"\n✅ EfficientNetB5 predictions saved:")
    print(f"   {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(df)}")

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())  # Count per class

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())     # Confidence stats
else:
    print("❌ No images processed")              # Notify if no images

print("🎉 EFFICIENTNETB5 SEGMENTED PIPELINE COMPLETE!")
