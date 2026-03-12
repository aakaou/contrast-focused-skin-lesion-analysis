# ==========================================================
# 1️⃣ IMPORT LIBRARIES
# ==========================================================
import os                          # Operating system operations
import re                          # Regular expressions (for extracting image IDs)
import cv2                         # OpenCV for reading & processing images
import numpy as np                 # Array manipulations
import pandas as pd                # DataFrames for tabular data
from pathlib import Path           # Path handling
from tqdm import tqdm              # Progress bars for loops

import matplotlib.pyplot as plt    # Plotting
import seaborn as sns              # Advanced plotting (heatmaps)

import tensorflow as tf            # Deep learning
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Sklearn metrics for evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ==========================================================
# 2️⃣ PATHS AND PARAMETERS
# ==========================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")  # Folder containing all images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional output folder
OUTPUT_CSV = "/aakaou/ham10000_mobilenetv1_7class_predictions_p1.csv"  # CSV output path

IMG_SIZE = (224, 224)  # MobileNetV1 requires 224x224 input images
BATCH_SIZE = 32        # Number of images per batch during prediction

# 7 classes in HAM10000 dataset
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ==========================================================
# 3️⃣ HELPER FUNCTION TO EXTRACT IMAGE ID
# ==========================================================
def extract_number(filename):
    """
    Extract numeric ID from filenames like ISIC_12345.jpg
    Used to sort files numerically
    """
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# ==========================================================
# 4️⃣ LOAD MOBILENETV1 MODEL WITH CUSTOM HEAD
# ==========================================================
# Load pretrained MobileNet, excluding top classifier layers
base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)  # Flatten features globally
x = Dense(512, activation='relu')(x)             # Dense layer with 512 units
x = Dense(256, activation='relu')(x)             # Dense layer with 256 units
predictions = Dense(7, activation='softmax')(x)  # Softmax output for 7 classes

# Combine base model + custom head
model = Model(inputs=base_model.input, outputs=predictions)
model.trainable = False  # Freeze base model weights for inference

print("✅ MobileNetV1 7-class model loaded (lightweight!)")

# ==========================================================
# 5️⃣ GET ALL IMAGE FILES
# ==========================================================
# List all .jpg and .png images in the folder
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images numerically based on ISIC ID
proc_files.sort(key=lambda x: extract_number(x.name))

print(f"📁 Found {len(proc_files)} images")

# ==========================================================
# 6️⃣ BATCH PREDICTION LOOP
# ==========================================================
results = []  # List to store prediction results

# Iterate over images in batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="MobileNetV1 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    batch_images = []    # Preprocessed images for this batch
    batch_filenames = [] # Filenames corresponding to batch images

    for img_path in batch_paths:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:  # Skip if image is unreadable
            continue

        # Resize to 224x224 for MobileNetV1
        image = cv2.resize(image, IMG_SIZE)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for MobileNetV1
        processed = preprocess_input(image.astype(np.float32))

        batch_images.append(processed)
        batch_filenames.append(img_path.name)

    if not batch_images:  # Skip empty batch
        continue

    # Convert list to numpy array
    X_batch = np.array(batch_images)

    # Predict probabilities for 7 classes
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # Index of predicted class
    pred_probs = np.max(probs, axis=1)       # Maximum probability (confidence)

    # Store each prediction in results
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
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ MobileNetV1 7-class predictions saved: {OUTPUT_CSV}")
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
print(classification_report(df['dx'], df['pred_class_name'], labels=class_names, zero_division=0, digits=4))

# ==========================================================
# 🔟 CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)

plt.figure(figsize=(12, 10))
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
prob_cols = [f'prob_{c}' for c in class_names]  # Columns with predicted probabilities

plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

auc_dict = {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[prob_cols[i]].values

    # Skip classes with too few positives
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)

    auc_dict[cls] = roc_auc
    valid_classes.append(cls)

    # Plot ROC curve
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f"{cls} (AUC={roc_auc:.3f})")

# Plot random baseline
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
plt.title("ROC Curves - MobileNetV1 7-class", fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Macro-AUC summary
macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🎯 Per-class AUC:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")
print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
