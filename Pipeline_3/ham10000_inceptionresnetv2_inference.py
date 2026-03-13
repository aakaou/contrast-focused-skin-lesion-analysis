"""
HAM10000 InceptionResNetV2 7-Class Skin Lesion Classification Pipeline
- Predicts 7-class probabilities for HAM10000 images
- Saves CSV of predictions
- Generates classification report and confusion matrix
- Plots ROC curves with AUC
- Uses ImageNet-pretrained InceptionResNetV2
"""

# ======================
# 1️⃣ IMPORTS
# ======================
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# ======================
# 2️⃣ SETTINGS
# ======================
PROC_DIR = Path("/aakaou/HAM10000_images_all")
OUTPUT_CSV = "/aakaou/ham10000_inceptionresnetv2_7class_predictions_p3.csv"
METADATA_CSV = "/aakaou/ham10000-dataset/HAM10000_metadata.csv"
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# ======================
# 3️⃣ HELPER FUNCTIONS
# ======================
def extract_number(filename):
    """Extract numeric ID from filename, e.g., ISIC_12345.jpg -> 12345"""
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

def load_images(image_paths):
    """Load images, resize, convert to RGB and preprocess"""
    images, filenames = [], []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32))
        images.append(img)
        filenames.append(img_path.name)
    return np.array(images), filenames

# ======================
# 4️⃣ MODEL SETUP
# ======================
print("🔄 Loading InceptionResNetV2 (ImageNet pretrained)...")
base_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.trainable = False
print("✅ InceptionResNetV2 7-class model loaded")

# ======================
# 5️⃣ LOAD IMAGE FILES
# ======================
proc_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))
proc_files.sort(key=lambda x: extract_number(x.name))
print(f"📁 Found {len(proc_files)} images")

# ======================
# 6️⃣ PREDICTIONS
# ======================
results = []
print("🚀 InceptionResNetV2 predictions...")
for i in tqdm(range(0, len(proc_files), BATCH_SIZE), desc="InceptionResNetV2 Predicting"):
    batch_paths = proc_files[i:i+BATCH_SIZE]
    X_batch, batch_filenames = load_images(batch_paths)
    if X_batch.size == 0:
        continue
    probs = model.predict(X_batch, verbose=0)
    pred_classes = np.argmax(probs, axis=1)
    pred_confidences = np.max(probs, axis=1)
    for j, (fname, prob, pred_class) in enumerate(zip(batch_filenames, probs, pred_classes)):
        results.append({
            'filename': fname,
            'pred_class_id': int(pred_class),
            'pred_class_name': CLASS_NAMES[pred_class],
            'pred_confidence': float(pred_confidences[j]),
            **{f'prob_{c}': float(prob[idx]) for idx, c in enumerate(CLASS_NAMES)}
        })

# Save CSV
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Predictions saved: {OUTPUT_CSV}")
    print(f"📊 Images processed: {len(results_df)}")
    print("\n🎯 Prediction distribution:")
    print(results_df['pred_class_name'].value_counts())
    print("\n📈 Confidence statistics:")
    print(results_df['pred_confidence'].describe())
else:
    print("❌ No images processed!")

# ======================
# 7️⃣ METRICS & EVALUATION
# ======================
meta_df = pd.read_csv(METADATA_CSV)
meta_df['filename'] = meta_df['image_id'].astype(str) + '.jpg'
df = results_df.merge(meta_df[['filename','dx']], on='filename', how='inner')
print(f"\n✅ Merged dataset: {df.shape[0]} images")

# Classification report
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(df['dx'], df['pred_class_name'], labels=CLASS_NAMES, zero_division=0, digits=4))

# Confusion matrix
cm = confusion_matrix(df['dx'], df['pred_class_name'], labels=CLASS_NAMES)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======================
# 8️⃣ ROC CURVES & AUC
# ======================
prob_cols = [f'prob_{c}' for c in CLASS_NAMES]
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'].values)
print("Classes in order:", lb.classes_)

plt.figure(figsize=(12,9))
colors = plt.cm.tab10(np.linspace(0,1,len(CLASS_NAMES)))
auc_dict = {}
valid_classes = []

for i, cls in enumerate(CLASS_NAMES):
    y_true_cls = y_true_bin[:,i]
    y_prob_cls = df[prob_cols[i]].values
    valid_mask = ~(np.isnan(y_prob_cls) | np.isinf(y_prob_cls))
    y_true_cls = y_true_cls[valid_mask]
    y_prob_cls = y_prob_cls[valid_mask]
    if len(y_true_cls) < 10 or y_true_cls.sum() < 2:
        print(f"⚠️ Skip {cls}: {y_true_cls.sum()} positives")
        continue
    fpr, tpr, _ = roc_curve(y_true_cls, y_prob_cls)
    roc_auc = auc(fpr, tpr)
    auc_dict[cls] = roc_auc
    valid_classes.append(cls)
    plt.plot(fpr, tpr, color=colors[i], lw=3, label=f'{cls} (AUC={roc_auc:.3f})')

plt.plot([0,1],[0,1], color='black', lw=2, linestyle='--', label='Random (AUC=0.5)')
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - InceptionResNetV2 7-class', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary
if auc_dict:
    macro_auc = np.mean(list(auc_dict.values()))
    print(f"\n🏆 MACRO-AUC: {macro_auc:.3f}")
    print(f"📈 Best class: {max(valid_classes, key=lambda x: auc_dict[x])}")
