# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os                        # Operating system utilities (file management)
import cv2                       # OpenCV for reading and processing images
import numpy as np               # Numerical operations and arrays
import pandas as pd              # Data manipulation and CSV handling
from pathlib import Path         # Easier file path handling
from tqdm import tqdm            # Progress bar for loops

import tensorflow as tf          # Deep learning framework
from tensorflow.keras.applications import ResNet50  # Pretrained ResNet50 architecture
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


# ==========================================================
# 2. DEFINE PATHS AND PARAMETERS
# ==========================================================

# Directory containing segmented skin lesion images
PROC_DIR = Path("/aakaou/HAM10000_segmented_p1")

# Output CSV file where predictions will be saved
OUTPUT_CSV = "/aakaou/ham10000_resnet50_segmented_predictions_p1.csv"

# CNN input image size required by ResNet
IMG_SIZE = (224, 224)

# Batch size for prediction (efficient GPU usage)
BATCH_SIZE = 32


# ==========================================================
# 3. BUILD RESNET50 MODEL (IMAGENET PRETRAINED)
# ==========================================================

print("🔄 Loading ResNet50 (ImageNet pretrained)...")

# Load official ResNet50 model pretrained on ImageNet
base_model = ResNet50(
    weights='imagenet',         # Use ImageNet pretrained weights
    include_top=False,          # Remove original 1000-class classifier
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add Global Average Pooling layer to convert feature maps → vector
x = GlobalAveragePooling2D()(base_model.output)

# Fully connected layer
x = Dense(512, activation='relu')(x)

# Second dense layer
x = Dense(256, activation='relu')(x)

# Final classification layer (7 skin lesion classes)
predictions = Dense(7, activation='softmax')(x)

# Build final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pretrained layers (inference only)
model.trainable = False

print("✅ ResNet50 ready (ImageNet pretrained)")


# ==========================================================
# 4. HAM10000 CLASS LABELS
# ==========================================================

class_names = [
    'nv',     # Melanocytic nevi
    'mel',    # Melanoma
    'bkl',    # Benign keratosis-like lesions
    'bcc',    # Basal cell carcinoma
    'akiec',  # Actinic keratoses
    'vasc',   # Vascular lesions
    'df'      # Dermatofibroma
]


# ==========================================================
# 5. HELPER FUNCTION TO SORT FILES
# ==========================================================

def extract_number(filename):
    """
    Extract numeric ID from filenames like:
    ISIC_0001234.jpg -> 1234
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


# ==========================================================
# 6. LOAD SEGMENTED IMAGES
# ==========================================================

# Collect all jpg and png images
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images based on ISIC numeric ID
seg_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Segmented images found: {len(seg_files)}")


# ==========================================================
# 7. RUN MODEL PREDICTIONS
# ==========================================================

print("🚀 Running ResNet50 predictions...")

results = []   # List to store prediction results

# Process images in batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE), desc="ResNet50 Segmented"):

    # Select batch paths
    batch_paths = seg_files[i:i + BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    # Load images
    for img_path in batch_paths:

        # Read image
        image = cv2.imread(str(img_path))

        # Skip corrupted images
        if image is None:
            continue

        # Resize image to CNN input size
        image = cv2.resize(image, IMG_SIZE)

        # Convert BGR (OpenCV) → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply ResNet preprocessing
        processed = tf.keras.applications.resnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    # Skip empty batch
    if not batch_images:
        continue

    # Convert list → numpy array
    X_batch = np.array(batch_images)

    # Predict probabilities
    probs = model.predict(X_batch, verbose=0)

    # Get predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Highest probability (confidence)
    pred_probs = np.max(probs, axis=1)

    # Store results
    for j, (fname, prob, pred_class) in enumerate(
        zip(batch_filenames, probs, pred_classes)
    ):

        results.append({
            'filename': fname,
            'pred_class_id': int(pred_class),
            'pred_class_name': class_names[pred_class],
            'pred_confidence': float(pred_probs[j]),

            # Save probability for each class
            'prob_nv': float(prob[0]),
            'prob_mel': float(prob[1]),
            'prob_bkl': float(prob[2]),
            'prob_bcc': float(prob[3]),
            'prob_akiec': float(prob[4]),
            'prob_vasc': float(prob[5]),
            'prob_df': float(prob[6])
        })


# ==========================================================
# 8. SAVE PREDICTIONS
# ==========================================================

if results:

    # Convert results list → DataFrame
    results_df = pd.DataFrame(results)

    # Save predictions to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Predictions saved: {OUTPUT_CSV}")

    print(f"📊 Images processed: {len(results_df)}")

    # Show most frequent predicted classes
    print("\n🎯 Top predictions:")
    print(results_df['pred_class_name'].value_counts().head())

    # Confidence statistics
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())

else:
    print("❌ No segmented images found!")

print("\n🎉 ResNet50 prediction pipeline completed!")

# ==========================================================
# 9. IMPORT EVALUATION LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# 10. LOAD PREDICTIONS
# ==========================================================

csv_path = "/aakaou/ham10000_resnet50_segmented_predictions_p1.csv"

pred_df = pd.read_csv(csv_path)

print("Predictions loaded:", pred_df.shape)


# ==========================================================
# 11. LOAD METADATA (GROUND TRUTH)
# ==========================================================

metadata = pd.read_csv("/aakaou/ham10000-dataset/HAM10000_metadata.csv")

# Convert image_id → filename
metadata['filename'] = metadata['image_id'] + '.jpg'


# ==========================================================
# 12. MERGE PREDICTIONS + TRUE LABELS
# ==========================================================

df = pred_df.merge(
    metadata[['filename','dx']],
    on='filename',
    how='inner'
)

print("Merged dataset:", df.shape)


# ==========================================================
# 13. CLASSIFICATION REPORT
# ==========================================================

print("\n📊 Classification Report\n")

print(classification_report(
    df['dx'],
    df['pred_class_name'],
    labels=class_names,
    zero_division=0,
    digits=4
))


# ==========================================================
# 14. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    df['dx'],
    df['pred_class_name'],
    labels=class_names
)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================================
# 15. ROC CURVES
# ==========================================================

# Convert labels to binary (One-vs-Rest)
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)

plt.figure(figsize=(12,9))

auc_dict = {}

for i, cls in enumerate(class_names):

    y_true_cls = y_true_bin[:,i]

    y_prob_cls = df[f'prob_{cls}'].values

    # Skip if insufficient positive samples
    if y_true_cls.sum() < 2:
        continue

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)

    roc_auc = auc(fpr, tpr)

    auc_dict[cls] = roc_auc

    plt.plot(
        fpr,
        tpr,
        lw=3,
        label=f'{cls} (AUC={roc_auc:.3f})'
    )

# Random baseline
plt.plot([0,1],[0,1],'k--',label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - ResNet50")
plt.legend()

plt.grid(True)

plt.show()


# ==========================================================
# 16. MACRO AUC
# ==========================================================

macro_auc = np.mean(list(auc_dict.values()))

print("\n🏆 Macro AUC:", round(macro_auc,4))

print("\nAUC per class:")

for cls,val in auc_dict.items():
    print(f"{cls}: {val:.3f}")
