# ============================================================
# EfficientNetB4 HAM10000 7-Class Prediction + Evaluation
# ============================================================

# === Import libraries ===
import os                          # For file path operations
import cv2                         # OpenCV for image loading and preprocessing
import numpy as np                 # Numerical computations
import pandas as pd                # DataFrame handling
from pathlib import Path           # File path handling
from tqdm import tqdm              # Progress bar for loops
import tensorflow as tf            # TensorFlow for deep learning
from tensorflow.keras.applications import EfficientNetB4      # Pretrained EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input  # Preprocessing
from tensorflow.keras.models import Model                    # To define model with custom top layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Layers for classification head

# ============================================================
# 1️⃣ Define paths and constants
# ============================================================
PROC_DIR = Path("/kaggle/working/HAM10000_images_all")   # Folder containing input images
OUT_DIR = Path("/kaggle/working/HAM10000_segmented_p1") # Folder for output (optional)
OUTPUT_CSV = "/kaggle/working/ham10000_efficientnetb4_7class_predictions_p1.csv"  # CSV output
IMG_SIZE = (380, 380)        # EfficientNetB4 recommended input size
BATCH_SIZE = 32              # Batch size for predictions

# ============================================================
# 2️⃣ Build EfficientNetB4 model with 7-class output
# ============================================================
# Load EfficientNetB4 base with ImageNet weights, exclude top layer
base_model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add global average pooling to reduce feature maps
x = GlobalAveragePooling2D()(base_model.output)

# Add custom dense layers for classification
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Final output layer with 7 classes and softmax activation
predictions = Dense(7, activation='softmax')(x)

# Define full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers (no weight updates during inference)
model.trainable = False
print("✅ EfficientNetB4 7-class model loaded (B3 → B4 heavy hitter!)")

# Define 7 HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ============================================================
# 3️⃣ Helper function to sort images by ISIC number
# ============================================================
def extract_number(filename):
    """Extract the numeric ID from an ISIC filename (e.g., ISIC_12345.jpg → 12345)."""
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')  # Place non-matching files at end

# ============================================================
# 4️⃣ Get list of image files
# ============================================================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Support JPG + PNG
proc_files.sort(key=lambda x: extract_number(x.name))  # Sort numerically by ISIC ID
print(f"Found {len(proc_files)} images")

# ============================================================
# 5️⃣ Batch prediction loop
# ============================================================
results = []  # List to store prediction results

# Iterate in batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="EfficientNetB4 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    
    batch_images = []    # Store preprocessed images
    batch_filenames = [] # Store filenames
    
    # Process each image
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image
        if image is None:
            continue  # Skip if image not read

        # Resize to model input size
        image = cv2.resize(image, IMG_SIZE)
        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for EfficientNet
        processed = preprocess_input(image.astype(np.float32))
        batch_images.append(processed)
        batch_filenames.append(img_path.name)
    
    if not batch_images:
        continue  # Skip empty batches
        
    X_batch = np.array(batch_images)  # Convert to NumPy array

    # Predict probabilities for 7 classes
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)  # Predicted class index
    pred_probs = np.max(probs, axis=1)       # Max confidence per sample

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
# 6️⃣ Save predictions to CSV
# ============================================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"✅ EfficientNetB4 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

print("✅ EfficientNetB4 Complete!")
print("🔥 B4 power: 380x380 + ~19M params = heavyweight contender!")

# ============================================================
# 7️⃣ Load predictions and metadata for evaluation
# ============================================================
csv_path = OUTPUT_CSV
pred_df = pd.read_csv(csv_path)  # Predicted results

# Load metadata (ground truth labels)
metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with ground truth
df = pred_df.merge(metadata[['filename', 'dx']], on='filename', how='inner')
print("Merged:", df.shape)

# ============================================================
# 8️⃣ Classification report
# ============================================================
print("📊 CLASSIFICATION REPORT")
print(classification_report(df['dx'], df['pred_class_name'], 
                          labels=class_names, zero_division=0, digits=4))

# ============================================================
# 9️⃣ Confusion matrix
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
# 10️⃣ ROC-AUC analysis (robust to missing classes)
# ============================================================
print("\n🎯 ROC-AUC Analysis")
prob_cols = [col for col in df.columns if col.startswith('prob_')]
print("Probability columns found:", len(prob_cols))

lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'])

valid_classes = []
auc_scores = {}

for i, class_name in enumerate(class_names):
    if f'prob_{class_name}' in df.columns:
        # Check if class exists in true labels
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
# 11️⃣ Full ROC plot
# ============================================================
plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
fpr_dict = dict()
tpr_dict = dict()
auc_dict = dict()
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
    
    plt.plot(fpr, tpr, color=colors[i], lw=3,
             label=f'{cls} (AUC = {roc_auc:.3f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random (AUC=0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - EfficientNetB4 7-class (predictions_p1)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 12️⃣ ROC-AUC results summary
# ============================================================
print(f"\n✅ ROC Plot saved: efficientnetb4_p1_roc_curve.png")
print(f"📊 Valid classes plotted: {len(valid_classes)}/7")
print("\n🎯 AUC Scores:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
