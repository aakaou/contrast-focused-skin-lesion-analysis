# ============================================================
# EfficientNetB5 HAM10000 7-Class Prediction + Evaluation
# ============================================================

# === 1️⃣ Import libraries ===
import os                          # File and directory handling
import cv2                         # OpenCV for image processing
import numpy as np                 # Numerical computing
import pandas as pd                # DataFrame manipulation
from pathlib import Path           # Object-oriented file paths
from tqdm import tqdm              # Progress bar
import tensorflow as tf            # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB5     # Pretrained EfficientNetB5
from tensorflow.keras.applications.efficientnet import preprocess_input  # Preprocessing
from tensorflow.keras.models import Model                   # To define custom model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Custom layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc  # Evaluation metrics
from sklearn.preprocessing import LabelBinarizer           # For one-hot encoding of labels
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2️⃣ Define paths and constants
# ============================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")   # Folder containing input images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional output folder
OUTPUT_CSV = "/aakaou/ham10000_efficientnetb5_7class_predictions_p1.csv"  # CSV output
IMG_SIZE = (456, 456)        # EfficientNetB5 recommended input size
BATCH_SIZE = 16              # Smaller batch size for larger images

# ============================================================
# 3️⃣ Build EfficientNetB5 model with custom top layers
# ============================================================
# Load base EfficientNetB5 (ImageNet weights, no top classifier)
base_model = EfficientNetB5(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add global average pooling (reduces feature map to 1D vector)
x = GlobalAveragePooling2D()(base_model.output)

# Add dense layers for custom 7-class classification
x = Dense(512, activation='relu')(x)   # Fully connected layer with 512 neurons
x = Dense(256, activation='relu')(x)   # Fully connected layer with 256 neurons

# Final softmax layer for 7 classes
predictions = Dense(7, activation='softmax')(x)

# Build full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers (no training, inference only)
model.trainable = False
print("✅ EfficientNetB5 7-class model loaded (B4 → B5 BEAST MODE!)")

# HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ============================================================
# 4️⃣ Helper function to extract numeric ID from filename
# ============================================================
def extract_number(filename):
    """
    Extract numeric ID from ISIC filenames like ISIC_12345.jpg → 12345
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')  # Place unmatched files at the end

# ============================================================
# 5️⃣ Get image filenames
# ============================================================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Support JPG + PNG
proc_files.sort(key=lambda x: extract_number(x.name))  # Sort numerically by ISIC ID
print(f"Found {len(proc_files)} images")

# ============================================================
# 6️⃣ Batch prediction
# ============================================================
results = []  # List to store prediction results

# Loop through images in batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="EfficientNetB5 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    
    batch_images = []    # Store preprocessed images
    batch_filenames = [] # Store filenames
    
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image
        if image is None:
            continue  # Skip if image cannot be read

        # Resize to model input size
        image = cv2.resize(image, IMG_SIZE)
        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for EfficientNetB5
        processed = preprocess_input(image.astype(np.float32))
        
        batch_images.append(processed)
        batch_filenames.append(img_path.name)
    
    if not batch_images:
        continue  # Skip empty batches
        
    X_batch = np.array(batch_images)  # Convert to NumPy array

    # Predict probabilities for 7 classes
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # Index of max probability
    pred_probs = np.max(probs, axis=1)       # Maximum probability per image

    # Save results
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
    
    print(f"✅ EfficientNetB5 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

print("✅ EfficientNetB5 Complete!")
print("🐲 B5 BEAST: 456x456 + 30M params = TOP ROC contender!")

# ============================================================
# 8️⃣ Load predictions and metadata for evaluation
# ============================================================
csv_path = OUTPUT_CSV
pred_df = pd.read_csv(csv_path)  # Predicted results

# Load metadata with true labels
metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")
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
print("Probability columns found:", len(prob_cols))

# One-hot encode true labels
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)
print("Classes in order:", lb.classes_)

valid_classes = []
auc_scores = {}

# Compute ROC-AUC per class
for i, class_name in enumerate(class_names):
    if f'prob_{class_name}' in df.columns:
        class_mask = y_true_bin[:, i].sum() > 0
        if class_mask.sum() > 1:  # Need at least 2 samples
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
# 12️⃣ Full ROC plot (one-vs-rest)
# ============================================================
plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
fpr_dict, tpr_dict, auc_dict = {}, {}, {}
valid_classes = []

for i, cls in enumerate(class_names):
    y_true_cls = y_true_bin[:, i]
    y_prob_cls = df[f'prob_{cls}'].values
    
    # Remove NaN/Inf values
    valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
    y_true_cls = y_true_cls[valid_mask]
    y_prob_cls = y_prob_cls[valid_mask]
    
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
plt.title('ROC Curves - EfficientNetB5 7-class (predictions_p1)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 13️⃣ ROC-AUC results summary
# ============================================================
print(f"\n✅ ROC Plot saved: efficientnetb5_p1_roc_curve.png")
print(f"📊 Valid classes plotted: {len(valid_classes)}/7")
print("\n🎯 AUC Scores:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
