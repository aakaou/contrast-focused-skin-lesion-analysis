import cv2                                  # OpenCV library used for image reading and preprocessing
import numpy as np                          # NumPy library for numerical operations and arrays
import pandas as pd                         # Pandas library for tabular data manipulation and CSV export
from pathlib import Path                    # Pathlib allows convenient handling of file system paths
from tqdm import tqdm                       # tqdm provides a progress bar for loops
import tensorflow as tf                     # TensorFlow deep learning framework
from tensorflow.keras.applications import MobileNetV3Large   # Import pretrained MobileNetV3-Large model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers for classification head
from tensorflow.keras.models import Model   # Model class used to construct the neural network

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented HAM10000 skin lesion images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_mobilenetv3_large_segmented_predictions_p2.csv"
# Path where the prediction results will be saved

IMG_SIZE = (224, 224)                       # Input image size required by MobileNetV3-Large
BATCH_SIZE = 32                             # Number of images processed simultaneously during prediction

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function used to extract the numeric ID from image filenames
    import re                                # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)   # Search for the pattern "ISIC_XXXXXX" inside filename
    return int(m.group(1)) if m else float('inf')
    # Return the numeric ID if found, otherwise return infinity to avoid sorting issues

# =====================================================
# Load MobileNetV3 Large
# =====================================================

print("🔄 Loading MobileNetV3-Large...")     # Display message indicating that the model is loading

base_model = MobileNetV3Large(               # Load the MobileNetV3-Large CNN architecture
    weights="imagenet",                      # Use pretrained weights trained on ImageNet dataset
    include_top=False,                       # Remove the original ImageNet classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3) # Define the input image size (224x224 RGB)
)

x = GlobalAveragePooling2D()(base_model.output)
# Convert convolution feature maps into a single feature vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 neurons and ReLU activation

x = Dense(256, activation="relu")(x)         # Second dense layer with 256 neurons

outputs = Dense(7, activation="softmax")(x)  # Final output layer for 7 HAM10000 lesion classes

model = Model(inputs=base_model.input, outputs=outputs)
# Construct the final model by combining the base CNN and classification head

model.trainable = False                      # Freeze model weights (inference only)

print("✅ MobileNetV3-Large ready")          # Confirmation message after successful loading

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of the seven skin lesion classes in the HAM10000 dataset

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all segmented image files with JPG or PNG extensions

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images according to their numeric ISIC identifier

print(f"📁 Segmented images found: {len(seg_files)}")
# Display the total number of segmented images detected

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (MobileNetV3-Large)...")
# Start the prediction pipeline

results = []                                 # List used to store prediction results

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="MobileNetV3-Large"):
# Iterate through images in batches with a progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]
    # Select a subset of images corresponding to the current batch

    batch_images = []                         # List to store processed images
    batch_filenames = []                      # List to store filenames

    for img_path in batch_paths:              # Loop through each image in the batch

        image = cv2.imread(str(img_path))     # Load image using OpenCV

        if image is None:                     # Check if image loading failed
            continue                          # Skip the image if it cannot be read

        image = cv2.resize(image, IMG_SIZE)   # Resize image to 224x224 pixels

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert image color format from BGR (OpenCV default) to RGB

        image = tf.keras.applications.mobilenet_v3.preprocess_input(
            image.astype(np.float32)
        )
        # Apply MobileNetV3 preprocessing normalization

        batch_images.append(image)            # Add processed image to batch list
        batch_filenames.append(img_path.name) # Store corresponding filename

    if not batch_images:                      # If batch contains no valid images
        continue                              # Skip prediction for this batch

    X_batch = np.array(batch_images)          # Convert list of images into NumPy array

    probs = model.predict(X_batch, verbose=0)
    # Run inference to obtain probability predictions for each class

    pred_classes = np.argmax(probs, axis=1)
    # Identify predicted class index with highest probability

    pred_probs = np.max(probs, axis=1)
    # Extract the highest probability as prediction confidence

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
    # Iterate through predictions and store them

        results.append({
            "filename": fname,                       # Image filename
            "pred_class_id": int(pred_class),        # Predicted class index
            "pred_class_name": class_names[pred_class],  # Predicted class label
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

if results:                                  # Check if prediction results exist

    df = pd.DataFrame(results)               # Convert results list into a pandas DataFrame

    df.to_csv(OUTPUT_CSV, index=False)       # Save predictions to CSV file

    print(f"\n✅ MobileNetV3-Large predictions saved:")
    print(f"   {OUTPUT_CSV}")                 # Display output file location

    print(f"📊 Images processed: {len(df)}")  # Show number of processed images

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Display the distribution of predicted classes

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Display statistical summary of prediction confidence

else:
    print("❌ No images processed")          # Message if prediction list is empty

print("🎉 MOBILENETV3-LARGE SEGMENTED PIPELINE COMPLETE!")
# Final message indicating successful completion of the prediction pipeline
