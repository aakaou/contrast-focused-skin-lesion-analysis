# ==========================================================
# 1️⃣ IMPORT LIBRARIES
# ==========================================================
import os                          # For file and path operations
import re                          # Regular expressions for extracting image IDs
import cv2                         # OpenCV for image reading/resizing/converting
import numpy as np                 # Numerical operations
import pandas as pd                # DataFrames
from pathlib import Path           # Cross-platform path handling
from tqdm import tqdm              # Progress bars

import matplotlib.pyplot as plt    # Plotting
import seaborn as sns              # Heatmaps for confusion matrix

import tensorflow as tf            # Deep learning framework
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ==========================================================
# 2️⃣ PATHS AND PARAMETERS
# ==========================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")  # Folder containing all images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional output folder for segmented images
OUTPUT_CSV = "/aakaou/ham10000_mobilenetv3small_7class_predictions_p1.csv"  # CSV output path

IMG_SIZE = (224, 224)  # Input size for MobileNetV3-Small
BATCH_SIZE = 32        # Number of images per batch

# HAM10000 7-class labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ==========================================================
# 3️⃣ HELPER FUNCTION TO EXTRACT NUMERIC IMAGE ID
# ==========================================================
def extract_number(filename):
    """
    Extract numeric ID from filenames like ISIC_12345.jpg
    Used for sorting images numerically
    """
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')  # float('inf') ensures non-matching filenames are sorted last

# ==========================================================
# 4️⃣ LOAD MOBILENETV3-SMALL MODEL WITH CUSTOM HEAD
# ==========================================================
# Load MobileNetV3Small pretrained on ImageNet without top layer
base_model = MobileNetV3Small(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add a global average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# Fully connected layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Output layer: 7 classes, softmax
predictions = Dense(7, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers for inference
model.trainable = False

print("✅ MobileNetV3-Small 7-class model loaded (SMALLEST + FASTEST!)")
# ==========================================================
# 5️⃣ GET IMAGE FILES
# ==========================================================
# Get all .jpg and .png images
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort numerically using ISIC ID
proc_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Found {len(proc_files)} images")

# ==========================================================
# 6️⃣ BATCH PREDICTION LOOP
# ==========================================================
results = []  # List to store results

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="MobileNetV3-Small Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    batch_images = []    # Store preprocessed images
    batch_filenames = [] # Store corresponding filenames

    for img_path in batch_paths:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue  # Skip unreadable images

        # Resize to 224x224
        image = cv2.resize(image, IMG_SIZE)
        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for MobileNetV3
        processed = preprocess_input(image.astype(np.float32))

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:
        continue  # Skip empty batch

    # Convert list to array
    X_batch = np.array(batch_images)

    # Predict probabilities
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # Class with max probability
    pred_probs = np.max(probs, axis=1)       # Max probability (confidence)

    # Store results for each image
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            'filename': fname,
            'pred_class_id': int(pred_class),
            'pred_class_name': class_names[pred_class],
            'pred_confidence': float(pred_probs[j]),
            'prob_nv': float(prob[0]),
            'prob_mel': float(prob[1]),
            'prob_bkl': float(prob[2]),
            'prob_bcc': float(prob[3]),
            'prob_akiec': float(prob[4]),
            'prob_vasc': float(prob[5]),
            'prob_df': float(prob[6])
        })

# ==========================================================
# 7️⃣ SAVE PREDICTIONS TO CSV
# ==========================================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)  # Save CSV
    print(f"✅ MobileNetV3-Small 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

# ==========================================================
# 8️⃣ MERGE PREDICTIONS WITH GROUND TRUTH METADATA
# ==========================================================
metadata = pd.read_csv("/aakaou/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'].astype(str) + '.jpg'

# Merge predictions with true labels
df = results_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')
print(f"✅ Merged dataset: {df.shape[0]} images")

# ==========================================================
# 9️⃣ CLASSIFICATION REPORT
# ==========================================================
print("📊 CLASSIFICATION REPORT")
print(classification_report(df['dx'], df['pred_class_name'],
                            labels=class_names, zero_division=0, digits=4))

# ==========================================================
# 🔟 CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================================
# 1️⃣1️⃣ ROC CURVES AND MACRO-AUC
# ==========================================================
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)  # One-hot encode true labels
prob_cols = [f'prob_{c}' for c in class_names]  # Probability columns

plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

auc_dict = {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[prob_cols[i]].values

    # Filter invalid probabilities
    valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
    y_true_cls = y_true_cls[valid_mask]
    y_prob_cls = y_prob_cls[valid_mask]

    # Skip if insufficient positives
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue

    # Compute ROC
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)

    auc_dict[cls] = roc_auc
    valid_classes.append(cls)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{cls} (AUC={roc_auc:.3f})")

# Random baseline
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
plt.title("ROC Curves - MobileNetV3-Small 7-class", fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================================
# 1️⃣2️⃣ RESULTS SUMMARY
# ==========================================================
macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🎯 Per-class AUC:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
