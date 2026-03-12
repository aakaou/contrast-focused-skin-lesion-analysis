import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
prediction_folder = Path(r"/aakaou/ham10000/pipeline4/seg_masks1")
ground_truth_folder = Path(r"/aakaou/ham10000/pipeline4/seg_overlays1")
csv_output = Path(r"/aakaou/ham10000/pipeline3/segmentation_metrics_vgg19.csv")
metadata_file = Path(r"/aakaou/ham10000/HAM10000_metadata.csv")
original_images_folder = Path(r"/aakaou/ham10000/HAM10000_images_all")  # adjust path

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# =========================
# LOAD METADATA
# =========================
meta_df = pd.read_csv(metadata_file)

# 7 HAM10000 classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {name: i for i, name in enumerate(class_names)}
meta_df['class_id'] = meta_df['dx'].map(class_map)

# =========================
# GET IMAGE FILES
# =========================
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

image_files = list(original_images_folder.glob("*.jpg")) + list(original_images_folder.glob("*.png"))
image_files.sort(key=lambda x: extract_number(x.name))
print(f"Found {len(image_files)} images")

# =========================
# CREATE DATAFRAME FOR TRAINING
# =========================
df = meta_df[meta_df['image_id'].isin([f.stem for f in image_files])].copy()
df['filename'] = df['image_id'] + '.jpg'
df['label'] = df['class_id']

# Split train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=0)

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                               horizontal_flip=True, zoom_range=0.2)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',  # sparse integer labels
    batch_size=BATCH_SIZE
)

val_data = val_gen.flow_from_dataframe(
    val_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# VGG19 MODEL
# =========================
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ VGG19 7-class model ready")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop])

# =========================
# PREDICTION ON VALIDATION SET
# =========================
val_data.reset()
y_true = val_df['label'].values
probs = model.predict(val_data, verbose=1)
y_pred = np.argmax(probs, axis=1)
pred_conf = np.max(probs, axis=1)

# =========================
# SAVE PREDICTIONS CSV
# =========================
results_df = pd.DataFrame({
    'filename': val_df['filename'].values,
    'actual_class_id': y_true,
    'actual_class_name': [class_names[i] for i in y_true],
    'pred_class_id': y_pred,
    'pred_class_name': [class_names[i] for i in y_pred],
    'pred_confidence': pred_conf
})

# Add individual class probabilities
for i, name in enumerate(class_names):
    results_df[f'prob_{name}'] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Predictions saved to {csv_output}")

# =========================
# EVALUATION
# =========================
print("\n🎯 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

roc_auc = roc_auc_score(label_binarize(y_true, classes=range(7)), probs, multi_class='ovr')
print(f"\n📈 ROC AUC Score (one-vs-rest): {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("Actual", fontweight='bold')
plt.title("Confusion Matrix", fontweight='bold')
plt.show()

# =========================
# ROC CURVES FOR ALL 7 CLASSES
# =========================
y_true_bin = label_binarize(y_true, classes=range(7))

plt.figure(figsize=(10,8))
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curve for 7-class HAM10000 Classification (VGG19)", fontweight='bold')
plt.legend(loc="lower right", prop={'weight':'bold'})
plt.grid(alpha=0.3)
plt.show()

