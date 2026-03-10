# =========================
# IMPORTS
# =========================
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PATHS
# =========================
prediction_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_masks1")
ground_truth_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_overlays1")
csv_output = Path(r"/home/aboubakr/Descargas/article4/ham10000/pipeline3/segmentation_metrics.csv")
metadata_file = Path(r"/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")
original_images_folder = Path(r"/home/aboubakr/Descargas/article4/ham10000/HAM10000_images_all")

# =========================
# CONSTANTS
# =========================
IMG_SIZE = (224, 224)  # Resize images to 224x224 for ResNet101
BATCH_SIZE = 16         # Number of images per training batch
EPOCHS = 10             # Maximum number of epochs to train

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)  # Load HAM10000 metadata

# Define 7 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

# Map textual labels to numeric labels
df['label'] = df['dx'].map(class_map)
df['filename'] = df['image_id'] + ".jpg"

# Keep only images that exist in folder
df = df[df['filename'].apply(lambda x: (original_images_folder / x).exists())]
print(f"✅ Images found: {len(df)}")

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,          # 20% for validation
    stratify=df['label'],   # Keep class distribution
    random_state=42         # Reproducible split
)

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalize like ResNet
    horizontal_flip=True,                     # Data augmentation
    zoom_range=0.2
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Training data
train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',  # Integer labels
    batch_size=BATCH_SIZE
)

# Validation data
val_data = val_gen.flow_from_dataframe(
    val_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False  # Keep order for evaluation
)

# =========================
# RESNET101 MODEL
# =========================
base_model = ResNet101(
    weights="imagenet",    # Pretrained on ImageNet
    include_top=False,     # Remove top layers
    input_shape=(224, 224, 3)
)

# Freeze all layers (transfer learning)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool feature maps
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)  # 7-class softmax

# Complete model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ ResNet101 loaded (ImageNet pretrained)")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# PREDICTIONS
# =========================
val_data.reset()
y_true = val_df['label'].values

probs = model.predict(val_data, verbose=1)   # Probabilities
y_pred = np.argmax(probs, axis=1)           # Predicted class
confidence = np.max(probs, axis=1)         # Max probability

# =========================
# SAVE RESULTS CSV
# =========================
results_df = pd.DataFrame({
    "filename": val_df['filename'].values,
    "actual_class_id": y_true,
    "actual_class_name": [class_names[i] for i in y_true],
    "pred_class_id": y_pred,
    "pred_class_name": [class_names[i] for i in y_pred],
    "pred_confidence": confidence
})

# Add individual class probabilities
for i, cls in enumerate(class_names):
    results_df[f"prob_{cls}"] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Predictions saved to: {csv_output}")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n📊 Classification Report\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted", fontweight="bold")
plt.ylabel("Actual", fontweight="bold")
plt.title("Confusion Matrix – ResNet101", fontweight="bold")
plt.tight_layout()
plt.show()

# =========================
# ROC CURVES (MULTI-CLASS)
# =========================
y_true_bin = label_binarize(y_true, classes=range(7))  # One-hot encode labels

plt.figure(figsize=(10,8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], "k--", lw=2)  # Random classifier line
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.title("ROC Curve – ResNet101 (HAM10000)", fontweight="bold")
plt.legend(prop={"weight": "bold"})
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
