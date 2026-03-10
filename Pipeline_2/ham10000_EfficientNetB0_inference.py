import cv2                                  # OpenCV library for image loading and processing
import numpy as np                          # NumPy for numerical operations and array manipulation
import pandas as pd                         # Pandas for working with tabular data and saving CSV files
from pathlib import Path                    # Pathlib for easy handling of filesystem paths
from tqdm import tqdm                       # tqdm for progress bars during loops
import tensorflow as tf                     # TensorFlow deep learning framework
from tensorflow.keras.applications import EfficientNetB0  # Import pretrained EfficientNetB0 model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model   # Model class used to construct neural networks

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented HAM10000 skin lesion images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb0_segmented_predictions_p2.csv"
# CSV file where prediction results will be saved

IMG_SIZE = (224, 224)                       # Input image size required by EfficientNetB0
BATCH_SIZE = 32                             # Number of images processed simultaneously during prediction

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function to extract numeric identifier from image filenames
    import re                                # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)   # Search for the pattern "ISIC_XXXXXX"
    return int(m.group(1)) if m else float('inf')
    # Return numeric ID if found, otherwise return infinity to maintain sorting stability

# =====================================================
# Load EfficientNetB0
# =====================================================

print("🔄 Loading EfficientNetB0...")        # Message indicating the model is being loaded

base_model = EfficientNetB0(                 # Load EfficientNetB0 architecture
    weights="imagenet",                      # Use pretrained ImageNet weights
    include_top=False,                       # Remove the original ImageNet classification head
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Define input shape (224x224 RGB images)
)

x = GlobalAveragePooling2D()(base_model.output)
# Convert convolutional feature maps into a single feature vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 neurons

x = Dense(256, activation="relu")(x)         # Second dense layer for deeper feature representation

outputs = Dense(7, activation="softmax")(x)  # Final output layer for the 7 HAM10000 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Build the final deep learning model

model.trainable = False                      # Freeze model weights (used only for inference)

print("✅ EfficientNetB0 ready")             # Confirmation message after loading model

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of lesion classes in the HAM10000 dataset

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all segmented images with JPG or PNG extension

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images numerically according to ISIC identifier

print(f"📁 Segmented images found: {len(seg_files)}")
# Print total number of segmented images

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB0)...")
# Start prediction process

results = []                                 # List to store prediction results

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB0"):
# Iterate through dataset in batches with progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]
    # Select current batch of image paths

    batch_images = []                         # List to store processed images
    batch_filenames = []                      # List to store filenames

    for img_path in batch_paths:              # Iterate through each image in batch

        image = cv2.imread(str(img_path))     # Read image from disk using OpenCV

        if image is None:                     # Check if image loading failed
            continue                          # Skip invalid image

        image = cv2.resize(image, IMG_SIZE)   # Resize image to 224x224

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert image from BGR (OpenCV format) to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        # Apply EfficientNet preprocessing normalization

        batch_images.append(image)            # Store processed image
        batch_filenames.append(img_path.name) # Store filename

    if not batch_images:                      # If no valid images in batch
        continue                              # Skip batch

    X_batch = np.array(batch_images)          # Convert image list to NumPy array

    probs = model.predict(X_batch, verbose=0)
    # Run model inference to obtain probability predictions

    pred_classes = np.argmax(probs, axis=1)
    # Determine predicted class index (highest probability)

    pred_probs = np.max(probs, axis=1)
    # Extract confidence score for predictions

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
    # Iterate through predictions in batch

        results.append({
            "filename": fname,                       # Image filename
            "pred_class_id": int(pred_class),        # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted lesion label
            "pred_confidence": float(pred_probs[j]), # Confidence of prediction

            "prob_nv": float(prob[0]),               # Probability of class NV
            "prob_mel": float(prob[1]),              # Probability of class MEL
            "prob_bkl": float(prob[2]),              # Probability of class BKL
            "prob_bcc": float(prob[3]),              # Probability of class BCC
            "prob_akiec": float(prob[4]),            # Probability of class AKIEC
            "prob_vasc": float(prob[5]),             # Probability of class VASC
            "prob_df": float(prob[6]),               # Probability of class DF
        })

# =====================================================
# Save results
# =====================================================

if results:                                  # Check if predictions were generated

    df = pd.DataFrame(results)               # Convert results list to DataFrame

    df.to_csv(OUTPUT_CSV, index=False)       # Save predictions to CSV file

    print(f"\n✅ EfficientNetB0 predictions saved:")
    print(f"   {OUTPUT_CSV}")                 # Show location of saved CSV file

    print(f"📊 Images processed: {len(df)}")  # Show number of processed images

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Display distribution of predicted classes

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Display statistical summary of confidence scores

else:
    print("❌ No images processed")          # Display message if no predictions were generated

print("🎉 EFFICIENTNETB0 SEGMENTED PIPELINE COMPLETE!")
# Final message indicating pipeline execution completed
