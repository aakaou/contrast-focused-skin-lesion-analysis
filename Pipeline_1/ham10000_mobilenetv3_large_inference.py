# ============================================================
# MobileNetV3-Large HAM10000 7-Class Prediction + Evaluation
# ============================================================

# === 1️⃣ Import required libraries ===
import os                          # File path operations
import cv2                         # Image reading and resizing
import numpy as np                 # Numerical operations
import pandas as pd                # DataFrames for results
from pathlib import Path           # OS-independent path handling
from tqdm import tqdm              # Progress bars for loops
import tensorflow as tf            # Deep learning framework
from tensorflow.keras.applications import MobileNetV3Large  # Pretrained MobileNetV3-Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input  # Preprocessing
from tensorflow.keras.models import Model  # Build custom model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc  # Evaluation
from sklearn.preprocessing import LabelBinarizer  # One-hot encoding for ROC
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2️⃣ Paths and constants
# ============================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")   # Folder containing images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional folder for segmented outputs
OUTPUT_CSV = "/aakaou/ham10000_mobilenetv3large_7class_predictions_p1.csv"  # Output CSV
IMG_SIZE = (224, 224)        # MobileNetV3-Large input size
BATCH_SIZE = 32              # Larger batch for smaller 224x224 images

# ============================================================
# 3️⃣ Build MobileNetV3-Large model
# ============================================================
base_model = MobileNetV3Large(
    weights="imagenet",        # Pretrained weights
    include_top=False,         # Remove classification head
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add global average pooling
x = GlobalAveragePooling2D()(base_model.output)

# Fully connected layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Output layer with 7 classes
predictions = Dense(7, activation='softmax')(x)

# Combine into a model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers for inference
model.trainable = False
print("✅ MobileNetV3-Large 7-class model loaded (V3-Small → Large POWER!)")

# HAM10000 class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ============================================================
# 4️⃣ Helper: extract numeric ID from filename
# ============================================================
def extract_number(filename):
    """
    Extracts numeric ID from filenames like ISIC_12345.jpg
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')  # Sort unmatched files at the end

# ============================================================
# 5️⃣ Collect and sort image files
# ============================================================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Support JPG + PNG
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"Found {len(proc_files)} images")

# ============================================================
# 6️⃣ Batch prediction
# ============================================================
results = []  # List to store prediction results

for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="MobileNetV3-Large Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    batch_images, batch_filenames = [], []
    
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Resize & convert to RGB
        image = cv2.resize(image, IMG_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for MobileNetV3
        processed = preprocess_input(image.astype(np.float32))
        batch_images.append(processed)
        batch_filenames.append(img_path.name)
    
    if not batch_images:
        continue
    
    X_batch = np.array(batch_images)
    
    # Predict probabilities
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)
    pred_probs = np.max(probs, axis=1)
    
    # Store results in a list
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

# ============================================================
# 7️⃣ Save predictions to CSV
# ============================================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ MobileNetV3-Large 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

print("✅ MobileNetV3-Large Complete!")
print("⚡ V3-Small → V3-Large: lightweight accuracy KING!")

# ============================================================
# 8️⃣ Load predictions + metadata for evaluation
# ============================================================
pred_df = pd.read_csv(OUTPUT_CSV)
metadata = pd.read_csv("/aakaou/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with ground truth
df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')
print("Merged:", df.shape)

# ============================================================
# 9️⃣ Classification report
# ============================================================
print("📊 CLASSIFICATION REPORT")
print(classification_report(df['dx'], df['pred_class_name'], 
                            labels=class_names, zero_division=0, digits=4))

# ============================================================
# 10️⃣ Confusion matrix
# ============================================================
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=class_names)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================================
# 11️⃣ ROC-AUC analysis
# ============================================================
print("\n🎯 ROC-AUC Analysis")
prob_cols = [col for col in df.columns if col.startswith('prob_')]
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)
print("Classes in order:", lb.classes_)

valid_classes = []
auc_scores = {}

for i, class_name in enumerate(class_names):
    if f'prob_{class_name}' in df.columns:
        class_mask = y_true_bin[:, i].sum() > 0
        if class_mask.sum() > 1:
            y_prob = df[f'prob_{class_name}'].values[class_mask]
            y_true = y_true_bin[class_mask, i]
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                auc_scores[class_name] = roc_auc
                valid_classes.append(class_name)
                print(f"  {class_name}: AUC = {roc_auc:.4f}")
            except ValueError as e:
                print(f"  {class_name}: SKIPPED ({e})")

if auc_scores:
    macro_auc = np.mean(list(auc_scores.values()))
    print(f"\n🎯 MACRO AUC: {macro_auc:.4f}")
else:
    print("\n⚠️ No valid ROC curves (missing samples per class)")

# ============================================================
# 12️⃣ ROC Plot (One-vs-Rest)
# ============================================================
plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
fpr_dict, tpr_dict, auc_dict = {}, {}, {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[f'prob_{cls}'].values
    
    # Remove NaN/Inf
    valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
    y_true_cls = y_true_cls[valid_mask]
    y_prob_cls = y_prob_cls[valid_mask]
    
    # Skip insufficient data
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue
    
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)
    
    fpr_dict[cls] = fpr
    tpr_dict[cls] = tpr
    auc_dict[cls] = roc_auc
    valid_classes.append(cls)
    
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{cls} (AUC = {roc_auc:.3f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random (AUC=0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - MobileNetV3-Large 7-class (predictions_p1)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 13️⃣ ROC-AUC results summary
# ============================================================
print(f"📊 Valid classes plotted: {len(valid_classes)}/7")
print("\n🎯 AUC Scores:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
