import os                          # File/directory handling
import cv2                         # Image loading and preprocessing
import numpy as np                 # Numerical operations
import pandas as pd                # Data handling
from tqdm import tqdm              # Progress bars
from pathlib import Path           # Path handling
import tensorflow as tf            # Deep learning framework
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns               # Visualization
import matplotlib.pyplot as plt

# =========================
# PATHS
# =========================
prediction_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_masks1")
ground_truth_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_overlays1")
csv_output = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/segmentation_metrics.csv")
metadata_file = Path(r"/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")
original_images_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/HAM10000_images_all")

IMG_SIZE = (224, 224)  # Resize all images to 224x224
BATCH_SIZE = 16         # Batch size for training
EPOCHS = 20             # Max epochs

# =========================
# LOAD METADATA
# =========================
meta_df = pd.read_csv(metadata_file)   # Load CSV metadata
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # HAM10000 classes
class_map = {name: i for i, name in enumerate(class_names)}        # Map class names to integer labels
meta_df['class_id'] = meta_df['dx'].map(class_map)                 # Add class ID column

# =========================
# GET IMAGE FILES
# =========================
def extract_number(filename):
    """Extract ISIC number from filename for sorting."""
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

# Collect all jpg/png images
image_files = list(original_images_folder.glob("*.jpg")) + list(original_images_folder.glob("*.png"))
# Sort images numerically by ISIC number
image_files.sort(key=lambda x: extract_number(x.name))
print(f"Found {len(image_files)} images")

# =========================
# CREATE DATAFRAME
# =========================
# Filter metadata to only existing images
df = meta_df[meta_df['image_id'].isin([f.stem for f in image_files])].copy()
df['filename'] = df['image_id'] + '.jpg'  # Full filename
df['label'] = df['class_id']              # Label column

# Split into train/validation sets (stratified)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=0)

# =========================
# DATA GENERATORS
# =========================
# Training data augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,  # Normalize for ResNet
    horizontal_flip=True,  # Random horizontal flips
    zoom_range=0.2         # Random zoom
)
# Validation generator (no augmentation)
val_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet.preprocess_input)

# Flow training data from DataFrame
train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',  # raw integer labels
    batch_size=BATCH_SIZE
)

# Flow validation data
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
# RESNET18 MODEL (Keras)
# =========================
def conv_block(x, filters, kernel_size=3, stride=1):
    """Residual block for ResNet18."""
    x_shortcut = x
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # Shortcut projection if needed
    if stride != 1 or x_shortcut.shape[-1] != filters:
        x_shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x_shortcut)
        x_shortcut = layers.BatchNormalization()(x_shortcut)
    x = layers.Add()([x, x_shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet18(input_shape=(224,224,3), num_classes=7):
    """Build a ResNet18-like model from scratch."""
    inputs = layers.Input(shape=input_shape)
    # Initial conv + maxpool
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    # Residual blocks
    filters_list = [64, 128, 256, 512]
    for filters in filters_list:
        x = conv_block(x, filters)
        x = conv_block(x, filters)
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

# Instantiate model
model = build_resnet18(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=7)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ ResNet18 7-class model ready")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Fit model with early stopping
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=[early_stop])

# =========================
# PREDICTION ON VALIDATION SET
# =========================
val_data.reset()                     # Reset generator
y_true = val_df['label'].values      # True labels
probs = model.predict(val_data, verbose=1)  # Predicted probabilities
y_pred = np.argmax(probs, axis=1)          # Predicted class IDs
pred_conf = np.max(probs, axis=1)          # Prediction confidence

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

# Add probability columns for each class
for i, name in enumerate(class_names):
    results_df[f'prob_{name}'] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Predictions saved to {csv_output}")

# =========================
# EVALUATION
# =========================
print("\n🎯 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ROC AUC
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

# ROC curves for all classes
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
plt.title("ROC Curve for 7-class HAM10000 Classification (ResNet18)", fontweight='bold')
plt.legend(loc="lower right", prop={'weight':'bold'})
plt.grid(alpha=0.3)
plt.show()
