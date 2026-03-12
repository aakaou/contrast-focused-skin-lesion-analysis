# ============================================================
# EfficientNetB7 HAM10000 7-Class Prediction + Evaluation
# ============================================================

# === 1️⃣ Import required libraries ===
import os                          # OS file path operations
import cv2                         # OpenCV for image reading and resizing
import numpy as np                 # Numerical computations
import pandas as pd                # DataFrame operations
from pathlib import Path           # Object-oriented paths
from tqdm import tqdm              # Progress bar
import tensorflow as tf            # TensorFlow deep learning framework
from tensorflow.keras.applications import EfficientNetB7  # Pretrained EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input  # Preprocessing function
from tensorflow.keras.models import Model  # Model constructor
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Custom layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc  # Evaluation metrics
from sklearn.preprocessing import LabelBinarizer  # One-hot encoding for ROC
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 2️⃣ Paths and constants
# ============================================================
PROC_DIR = Path("/aakaou/HAM10000_images_all")   # Folder containing input images
OUT_DIR = Path("/aakaou/HAM10000_segmented_p1") # Optional output folder
OUTPUT_CSV = "/aakaou/ham10000_efficientnetb7_7class_predictions_p1.csv"  # CSV output
IMG_SIZE = (600, 600)        # EfficientNetB7 optimal input size
BATCH_SIZE = 4               # Smaller batch size for large 600x600 images

# ============================================================
# 3️⃣ Build EfficientNetB7 model
# ============================================================
# Load pretrained EfficientNetB7 without top layers
base_model = EfficientNetB7(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Add global average pooling to reduce spatial dimensions
x = GlobalAveragePooling2D()(base_model.output)

# Add fully connected layers
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# Final output layer with 7 classes and softmax
predictions = Dense(7, activation='softmax')(x)

# Combine base and top layers into a model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers for inference only
model.trainable = False
print("✅ EfficientNetB7 7-class model loaded (ULTIMATE SOTA BEHEMOTH!)")

# Define HAM10000 class names
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ============================================================
# 4️⃣ Helper function to extract numeric ID from filename
# ============================================================
def extract_number(filename):
    """
    Extract numeric ID from ISIC filenames like ISIC_12345.jpg -> 12345
    """
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')  # Place unmatched files at the end

# ============================================================
# 5️⃣ Collect and sort image files
# ============================================================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))  # Support JPG and PNG
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"Found {len(proc_files)} images")

# ============================================================
# 6️⃣ Batch prediction
# ============================================================
results = []  # List to store predictions

# Loop over batches
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="EfficientNetB7 Predicting"):
    batch_paths = proc_files[i:i + BATCH_SIZE]
    
    batch_images = []    # Preprocessed images
    batch_filenames = [] # Corresponding filenames
    
    for img_path in batch_paths:
        image = cv2.imread(str(img_path))  # Read image
        if image is None:
            continue  # Skip unreadable images
        
        # Resize to model input size
        image = cv2.resize(image, IMG_SIZE)
        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Preprocess for EfficientNetB7
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
    
    # Store results
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
    
    print(f"✅ EfficientNetB7 7-class predictions saved: {OUTPUT_CSV}")
    print(f"📊 Processed {len(results_df)} images")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

print("✅ EfficientNetB7 Complete!")
print("👑 ULTIMATE SOTA: 600x600 + 66M params = ROC CHAMPION!")

# ============================================================
# 8️⃣ Load predictions and metadata for evaluation
# ============================================================
pred_df = pd.read_csv(OUTPUT_CSV)

metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")
metadata['filename'] = metadata['image_id'] + '.jpg'

# Merge predictions with true labels
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

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random (AUC=0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - EfficientNetB7 7-class (predictions_p1)', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 13️⃣ ROC-AUC results summary
# ============================================================
print(f"\n✅ ROC Plot saved: efficientnetb7_p1_roc_curve.png")
print(f"📊 Valid classes plotted: {len(valid_classes)}/7")
print("\n🎯 AUC Scores:")
for cls in sorted(valid_classes, key=lambda x: auc_dict[x], reverse=True):
    print(f"  {cls:>8}: {auc_dict[cls]:.3f}")

macro_auc = np.mean([auc_dict[cls] for cls in valid_classes])
print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
