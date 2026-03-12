# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import os                              # Provides operating system utilities
import cv2                             # OpenCV library for image processing
import numpy as np                     # Numerical computations (arrays, matrices)
import pandas as pd                    # Data analysis and CSV handling
from pathlib import Path               # Object-oriented file system paths
from tqdm import tqdm                  # Progress bar for loops

# Import VGG19 architecture and preprocessing function
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

from tensorflow.keras.models import Model             # Keras model class
from tensorflow.keras.layers import Dense              # Fully connected layer
from tensorflow.keras.layers import GlobalAveragePooling2D  # Pooling layer

import tensorflow as tf                # Deep learning framework


# ===============================
# DEFINE DATASET PATHS
# ===============================

PROC_DIR = Path("/aakaou/HAM10000_images_all") 
# Folder containing processed HAM10000 images

OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") 
# Directory where segmentation outputs or processed files are stored

OUTPUT_CSV = "/aakaou/ham10000_vgg19_7class_predictions_p1.csv" 
# CSV file where model predictions will be saved

IMG_SIZE = (224, 224)                  
# Image size required by VGG networks

BATCH_SIZE = 32                        
# Number of images processed at once during inference


# ===============================
# BUILD VGG19 MODEL
# ===============================

# Load pretrained VGG19 network without the top classification layer
base_model = VGG19(
    weights="imagenet",                 # Use ImageNet pretrained weights
    include_top=False,                  # Remove original classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)  # Input image dimensions
)

# Apply global average pooling to reduce spatial dimensions
x = GlobalAveragePooling2D()(base_model.output)

# Add fully connected layer with 512 neurons
x = Dense(512, activation='relu')(x)

# Add another dense layer with 256 neurons
x = Dense(256, activation='relu')(x)

# Final output layer for 7 skin lesion classes
predictions = Dense(7, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pretrained layers (only inference, no training)
model.trainable = False

print("✅ VGG19 7-class model loaded")


# ===============================
# DEFINE HAM10000 CLASS LABELS
# ===============================

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
# These are the 7 diagnostic categories in the HAM10000 dataset


# ===============================
# FUNCTION TO SORT IMAGES BY ID
# ===============================

def extract_number(filename):
    """
    Extract numerical ID from image name.
    Example: ISIC_0001234.jpg -> 1234
    This helps sort images correctly.
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# ===============================
# GET IMAGE FILE LIST
# ===============================

# Collect all JPG and PNG images from directory
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images using extracted numerical ID
proc_files.sort(key=lambda x: extract_number(x.name))

print(f"Found {len(proc_files)} images")


# ===============================
# MODEL INFERENCE (PREDICTION)
# ===============================

results = []   # List that will store predictions for each image

# Iterate through images in batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="VGG19 Predicting"):

    batch_paths = proc_files[i:i + BATCH_SIZE]   # Select batch of image paths
    
    batch_images = []        # Store preprocessed images
    batch_filenames = []     # Store image filenames

    # Process each image in the batch
    for img_path in batch_paths:

        image = cv2.imread(str(img_path))       # Read image using OpenCV
        
        if image is None:                       # Skip corrupted images
            continue
            
        image = cv2.resize(image, IMG_SIZE)     # Resize image to 224x224
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # Convert OpenCV BGR format → RGB
        
        processed = preprocess_input(image.astype(np.float32))
        # Apply VGG19 preprocessing
        
        batch_images.append(processed)          # Add image to batch
        batch_filenames.append(img_path.name)   # Store filename
    
    if not batch_images:                        # Skip empty batches
        continue
        
    X_batch = np.array(batch_images)            # Convert batch to numpy array
    
    probs = model.predict(X_batch, verbose=0)   # Predict class probabilities
    
    pred_classes = np.argmax(probs, axis=1)     # Get predicted class index
    
    pred_probs = np.max(probs, axis=1)          # Get highest probability
    
    
    # ===============================
    # STORE RESULTS
    # ===============================

    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        
        results.append({
            'filename': fname,                       # Image filename
            
            'pred_class_id': int(pred_class),        # Predicted class index
            
            'pred_class_name': class_names[pred_class],  # Predicted label
            
            'pred_confidence': float(pred_probs[j]), # Model confidence
            
            'prob_nv': float(prob[0]),               # Probability for class NV
            'prob_mel': float(prob[1]),              # Probability for MEL
            'prob_bkl': float(prob[2]),              # Probability for BKL
            'prob_bcc': float(prob[3]),              # Probability for BCC
            'prob_akiec': float(prob[4]),            # Probability for AKIEC
            'prob_vasc': float(prob[5]),             # Probability for VASC
            'prob_df': float(prob[6])                # Probability for DF
        })


# ===============================
# SAVE RESULTS TO CSV
# ===============================

if results:
    
    results_df = pd.DataFrame(results)         # Convert results to DataFrame
    
    results_df.to_csv(OUTPUT_CSV, index=False) # Save predictions to CSV
    
    print(f"✅ VGG19 predictions saved: {OUTPUT_CSV}")
    
    print(f"📊 Processed {len(results_df)} images")
    
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())

else:
    print("❌ No images processed!")

print("✅ VGG19 prediction stage complete!")


# ============================================================
# EVALUATION SECTION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# Load prediction CSV
pred_df = pd.read_csv(OUTPUT_CSV)

# Load HAM10000 metadata containing ground truth labels
metadata = pd.read_csv("/aakaou/ham10000-dataset/HAM10000_metadata.csv")

# Create filename column to match prediction CSV
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with ground truth labels
df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')

print("Merged dataset:", df.shape)


# ===============================
# CLASSIFICATION REPORT
# ===============================

print("\n📊 Classification Report")

print(classification_report(
    df['dx'],                    # True labels
    df['pred_class_name'],       # Predicted labels
    labels=class_names,
    zero_division=0,
    digits=4
))


# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()


# ===============================
# ROC CURVE ANALYSIS
# ===============================

print("\n🎯 ROC Curve Analysis")

lb = LabelBinarizer()                     # Convert labels to binary format
y_true_bin = lb.fit_transform(df['dx'])   # One-hot encoding

plt.figure(figsize=(12,9))

colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))

auc_dict = {}
valid_classes = []

for i, cls in enumerate(class_names):

    y_true_cls = y_true_bin[:, i]         # True binary labels
    y_prob_cls = df[f'prob_{cls}'].values # Predicted probabilities
    
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)  # Compute ROC curve
    
    roc_auc = auc(fpr, tpr)               # Area under curve
    
    auc_dict[cls] = roc_auc
    valid_classes.append(cls)
    
    plt.plot(
        fpr,
        tpr,
        color=colors[i],
        lw=3,
        label=f'{cls} (AUC = {roc_auc:.3f})'
    )


# Plot random classifier baseline
plt.plot([0,1],[0,1],'k--',label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - VGG19 HAM10000")

plt.legend(loc="lower right")

plt.grid(True)

plt.tight_layout()

plt.show()


# ===============================
# SUMMARY RESULTS
# ===============================

macro_auc = np.mean(list(auc_dict.values()))

print("\n🏆 MACRO-AUC:", round(macro_auc,3))

print("Best class:", max(auc_dict, key=auc_dict.get))
