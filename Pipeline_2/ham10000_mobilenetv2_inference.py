import os                                  # Provides operating system utilities (file handling, environment interaction)
import cv2                                 # OpenCV library used for image loading, resizing, and color conversion
import numpy as np                         # NumPy for numerical operations and array manipulation
import pandas as pd                        # Pandas for handling tabular data and saving results to CSV
from pathlib import Path                   # Pathlib provides object-oriented filesystem paths
from tqdm import tqdm                      # tqdm creates progress bars for loops
import tensorflow as tf                    # TensorFlow deep learning framework
from tensorflow.keras.applications import MobileNetV2   # Import pretrained MobileNetV2 CNN architecture
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers used for classification head
from tensorflow.keras.models import Model  # Keras Model class for building neural networks

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")  
# Directory containing segmented HAM10000 lesion images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_mobilenetv2_segmented_predictions_p2.csv"  
# File where prediction results will be saved

IMG_SIZE = (224, 224)                      # Input image size required by MobileNetV2
BATCH_SIZE = 32                            # Number of images processed simultaneously during prediction

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):               # Function to extract numeric ID from image filename
    import re                               # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)  # Search pattern "ISIC_XXXXXX" inside filename
    return int(m.group(1)) if m else float('inf')  
    # Return extracted number if found, otherwise return infinity for sorting safety

# =====================================================
# Load MobileNetV2
# =====================================================

print("🔄 Loading MobileNetV2...")          # Display message indicating model loading

base_model = MobileNetV2(                   # Load MobileNetV2 architecture
    weights="imagenet",                     # Use pretrained weights from ImageNet dataset
    include_top=False,                      # Remove original ImageNet classifier layer
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Define input shape: 224x224 RGB images
)

x = GlobalAveragePooling2D()(base_model.output)  
# Convert convolutional feature maps into a single feature vector

x = Dense(256, activation="relu")(x)        # Fully connected layer with 256 neurons and ReLU activation
x = Dense(128, activation="relu")(x)        # Second dense layer with 128 neurons

outputs = Dense(7, activation="softmax")(x) 
# Final classification layer with 7 outputs corresponding to HAM10000 classes

model = Model(inputs=base_model.input, outputs=outputs)  
# Build full deep learning model by connecting input to final output

model.trainable = False                     
# Freeze all weights (used only for inference, not training)

print("✅ MobileNetV2 ready")               # Confirmation that the model is ready

# HAM10000 class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  
# List of the 7 lesion categories in the HAM10000 dataset

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  
# Collect all JPG and PNG images from the segmented image directory

seg_files.sort(key=lambda x: extract_number(x.name))  
# Sort images according to their numeric ISIC identifier

print(f"📁 Segmented images found: {len(seg_files)}")  
# Print total number of segmented images detected

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (MobileNetV2)...")  
# Start prediction process

results = []                                # Initialize empty list to store prediction results

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="MobileNetV2 Segmented"):  
# Loop through images in batches with progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]  
    # Select subset of image paths corresponding to the current batch

    batch_images = []                       # List to store processed images
    batch_filenames = []                    # List to store filenames

    for img_path in batch_paths:            # Loop through each image in the batch

        image = cv2.imread(str(img_path))   # Read image from disk using OpenCV

        if image is None:                   # If image cannot be read
            continue                        # Skip and move to next image

        image = cv2.resize(image, IMG_SIZE) # Resize image to 224x224

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # Convert color format from OpenCV BGR to RGB

        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image.astype(np.float32)
        )                                   
        # Apply MobileNetV2-specific preprocessing (normalization)

        batch_images.append(image)          # Add processed image to batch list
        batch_filenames.append(img_path.name)  
        # Store filename for later reference

    if not batch_images:                    # If no images were loaded successfully
        continue                            # Skip the batch

    X_batch = np.array(batch_images)        # Convert image list to NumPy array

    probs = model.predict(X_batch, verbose=0)  
    # Perform model prediction to obtain class probability distributions

    pred_classes = np.argmax(probs, axis=1)  
    # Determine predicted class index (highest probability)

    pred_probs = np.max(probs, axis=1)      
    # Extract prediction confidence values

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):  
    # Loop through predictions in the current batch

        results.append({
            "filename": fname,                      # Image filename
            "pred_class_id": int(pred_class),       # Predicted class index
            "pred_class_name": class_names[pred_class],  # Predicted lesion label
            "pred_confidence": float(pred_probs[j]),     # Confidence of prediction

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

if results:                                  # Check if predictions exist

    df = pd.DataFrame(results)               # Convert results list to pandas DataFrame

    df.to_csv(OUTPUT_CSV, index=False)       # Save DataFrame to CSV file

    print(f"\n✅ MobileNetV2 predictions saved:")  
    print(f"   {OUTPUT_CSV}")                 # Display file location

    print(f"📊 Images processed: {len(df)}")  # Show total number of processed images

    print("\n🎯 Class distribution:")        # Display distribution of predicted classes
    print(df["pred_class_name"].value_counts())

    print("\n📈 Confidence statistics:")     # Show statistics of prediction confidence
    print(df["pred_confidence"].describe())

else:
    print("❌ No images processed")          # If no predictions were generated

print("🎉 MOBILENETV2 SEGMENTED PIPELINE COMPLETE!")  
# Final message indicating pipeline completion
