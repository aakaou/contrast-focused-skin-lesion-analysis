import cv2                                  # OpenCV library for image loading and image processing
import numpy as np                          # NumPy library for numerical operations and array manipulation
import pandas as pd                         # Pandas library for tabular data handling and CSV export
from pathlib import Path                    # Pathlib for easier file and directory management
from tqdm import tqdm                       # tqdm provides progress bars for loops
import tensorflow as tf                     # TensorFlow deep learning framework
from tensorflow.keras.applications import MobileNetV3Small  # Import pretrained MobileNetV3-Small architecture
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers used for building the classifier
from tensorflow.keras.models import Model   # Keras model class for building custom neural networks

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/aakaou/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented skin lesion images from HAM10000 dataset

OUTPUT_CSV = "/aakaou/ham10000/pipeline2/ham10000_mobilenetv3_small_segmented_predictions_p2.csv"
# Output CSV file where prediction results will be saved

IMG_SIZE = (224, 224)                       # Image size expected by MobileNetV3
BATCH_SIZE = 32                             # Number of images processed simultaneously

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function used to extract numeric ID from image filename
    import re                                # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)   # Search pattern "ISIC_XXXXXX" inside filename
    return int(m.group(1)) if m else float('inf')
    # Return the extracted number for sorting; if not found return infinity

# =====================================================
# Load MobileNetV3 Small
# =====================================================

print("🔄 Loading MobileNetV3-Small...")     # Display message indicating model loading

base_model = MobileNetV3Small(               # Load MobileNetV3-Small architecture
    weights="imagenet",                      # Use pretrained ImageNet weights
    include_top=False,                       # Remove original ImageNet classification head
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Define input size: 224x224 RGB images
)

x = GlobalAveragePooling2D()(base_model.output)
# Convert convolution feature maps into a single feature vector

x = Dense(256, activation="relu")(x)         # Fully connected layer with 256 neurons

x = Dense(128, activation="relu")(x)         # Second dense layer to refine features

outputs = Dense(7, activation="softmax")(x)  # Final classification layer with 7 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Create final model combining MobileNetV3 base and classification head

model.trainable = False                      # Freeze model weights (inference only)

print("✅ MobileNetV3-Small ready")          # Confirmation that the model is loaded

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of the 7 skin lesion classes in the HAM10000 dataset

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all segmented images with JPG or PNG format

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images by their ISIC numeric identifier

print(f"📁 Segmented images found: {len(seg_files)}")
# Print the number of segmented images detected

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (MobileNetV3-Small)...")
# Start the prediction process

results = []                                 # List to store prediction outputs

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="MobileNetV3-Small"):
# Loop through images in batches with progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]
    # Select a subset of image paths corresponding to the batch

    batch_images = []                        # List to store processed images
    batch_filenames = []                     # List to store corresponding filenames

    for img_path in batch_paths:             # Iterate over each image in batch

        image = cv2.imread(str(img_path))    # Load image using OpenCV

        if image is None:                    # If loading fails
            continue                         # Skip this image

        image = cv2.resize(image, IMG_SIZE)  # Resize image to required input size

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert OpenCV image from BGR format to RGB format

        image = tf.keras.applications.mobilenet_v3.preprocess_input(
            image.astype(np.float32)
        )
        # Apply MobileNetV3 preprocessing (normalization)

        batch_images.append(image)           # Add processed image to batch list
        batch_filenames.append(img_path.name) # Store filename

    if not batch_images:                     # If batch is empty
        continue                             # Skip prediction

    X_batch = np.array(batch_images)         # Convert list of images to NumPy array

    probs = model.predict(X_batch, verbose=0)
    # Run model inference to obtain probability distributions

    pred_classes = np.argmax(probs, axis=1)
    # Determine predicted class index (highest probability)

    pred_probs = np.max(probs, axis=1)
    # Extract confidence values of predictions

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
    # Iterate through predictions and store results

        results.append({
            "filename": fname,                       # Image filename
            "pred_class_id": int(pred_class),        # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
            "pred_confidence": float(pred_probs[j]), # Prediction confidence score

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

    df = pd.DataFrame(results)               # Convert results list into a pandas DataFrame

    df.to_csv(OUTPUT_CSV, index=False)       # Save predictions to CSV file

    print(f"\n✅ MobileNetV3-Small predictions saved:")
    print(f"   {OUTPUT_CSV}")                 # Print output file path

    print(f"📊 Images processed: {len(df)}")  # Display number of processed images

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Display distribution of predicted classes

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Display statistical summary of confidence scores

else:
    print("❌ No images processed")          # Message if no images were processed

print("🎉 MOBILENETV3-SMALL SEGMENTED PIPELINE COMPLETE!")
# Final message indicating successful completion of the pipeline
