import cv2                                  # OpenCV library for image loading and image processing
import numpy as np                          # NumPy for numerical operations and array manipulation
import pandas as pd                         # Pandas for tabular data handling and saving CSV files
from pathlib import Path                    # Pathlib for easier management of file system paths
from tqdm import tqdm                       # tqdm creates progress bars for loops
import tensorflow as tf                     # TensorFlow deep learning framework
from tensorflow.keras.applications import EfficientNetB1  # Import pretrained EfficientNetB1 architecture
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense  # Layers used for classification head
from tensorflow.keras.models import Model   # Model class used to build neural networks

# =====================================================
# Paths
# =====================================================

PROC_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_segmented_images")
# Directory containing segmented HAM10000 lesion images

OUTPUT_CSV = "/home/aboubakr/Descargas/article4/ham10000/pipeline2/ham10000_efficientnetb1_segmented_predictions_p2.csv"
# File where prediction results will be saved

IMG_SIZE = (240, 240)                       # Input image size required by EfficientNetB1
BATCH_SIZE = 32                             # Number of images processed at once during prediction

# =====================================================
# Utility: extract ISIC number
# =====================================================

def extract_number(filename):                # Function to extract numeric ID from image filename
    import re                                # Regular expression module
    m = re.search(r'ISIC_(\d+)', filename)   # Search for pattern "ISIC_XXXXXX"
    return int(m.group(1)) if m else float('inf')
    # Return the numeric ID if found, otherwise return infinity (helps safe sorting)

# =====================================================
# Load EfficientNetB1
# =====================================================

print("🔄 Loading EfficientNetB1...")        # Display message indicating model loading

base_model = EfficientNetB1(                 # Load EfficientNetB1 architecture
    weights="imagenet",                      # Use pretrained weights from ImageNet
    include_top=False,                       # Remove original ImageNet classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Define input image size (240x240 RGB)
)

x = GlobalAveragePooling2D()(base_model.output)
# Convert convolution feature maps into a single feature vector

x = Dense(512, activation="relu")(x)         # Fully connected layer with 512 neurons

x = Dense(256, activation="relu")(x)         # Second dense layer to refine feature representation

outputs = Dense(7, activation="softmax")(x)  # Output layer with 7 neurons for HAM10000 classes

model = Model(inputs=base_model.input, outputs=outputs)
# Construct the final neural network by combining base model and classification head

model.trainable = False                      # Freeze all model weights (inference only)

print("✅ EfficientNetB1 ready")              # Confirmation that the model is loaded

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# List of the 7 skin lesion categories in the HAM10000 dataset

# =====================================================
# Load segmented images
# =====================================================

seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
# Collect all segmented images with JPG or PNG extensions

seg_files.sort(key=lambda x: extract_number(x.name))
# Sort images numerically based on their ISIC identifier

print(f"📁 Segmented images found: {len(seg_files)}")
# Print the number of segmented images detected

# =====================================================
# Prediction loop
# =====================================================

print("🚀 Segment-aware predictions (EfficientNetB1)...")
# Start the prediction process

results = []                                 # List to store prediction results

for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="EfficientNetB1"):
# Iterate through dataset in batches with a progress bar

    batch_paths = seg_files[i:i + BATCH_SIZE]
    # Select subset of images corresponding to the current batch

    batch_images = []                         # List to store processed images
    batch_filenames = []                      # List to store filenames

    for img_path in batch_paths:              # Loop through each image in the batch

        image = cv2.imread(str(img_path))     # Load image from disk using OpenCV

        if image is None:                     # Check if image failed to load
            continue                          # Skip this image

        image = cv2.resize(image, IMG_SIZE)   # Resize image to 240x240 pixels

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert image color format from BGR (OpenCV default) to RGB

        image = tf.keras.applications.efficientnet.preprocess_input(
            image.astype(np.float32)
        )
        # Apply EfficientNet-specific preprocessing normalization

        batch_images.append(image)            # Add processed image to batch list
        batch_filenames.append(img_path.name) # Store filename

    if not batch_images:                      # If no valid images in batch
        continue                              # Skip batch

    X_batch = np.array(batch_images)          # Convert image list to NumPy array

    probs = model.predict(X_batch, verbose=0)
    # Run model inference to obtain probability predictions

    pred_classes = np.argmax(probs, axis=1)
    # Determine predicted class index (highest probability)

    pred_probs = np.max(probs, axis=1)
    # Extract prediction confidence (maximum probability)

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
    # Iterate through predictions in the batch

        results.append({
            "filename": fname,                       # Image filename
            "pred_class_id": int(pred_class),        # Predicted class index
            "pred_class_name": class_names[pred_class], # Predicted class label
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

if results:                                  # Check if predictions exist

    df = pd.DataFrame(results)               # Convert results list to pandas DataFrame

    df.to_csv(OUTPUT_CSV, index=False)       # Save predictions to CSV file

    print(f"\n✅ EfficientNetB1 predictions saved:")
    print(f"   {OUTPUT_CSV}")                 # Show output file path

    print(f"📊 Images processed: {len(df)}")  # Print number of processed images

    print("\n🎯 Class distribution:")
    print(df["pred_class_name"].value_counts())
    # Show distribution of predicted classes

    print("\n📈 Confidence statistics:")
    print(df["pred_confidence"].describe())
    # Display statistical summary of confidence values

else:
    print("❌ No images processed")          # If no predictions were generated

print("🎉 EFFICIENTNETB1 SEGMENTED PIPELINE COMPLETE!")
# Final message indicating pipeline execution finished
