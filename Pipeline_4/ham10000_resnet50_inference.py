# =========================
# IMPORTS
# =========================
from pathlib import Path                    # Path handling for files
import os                                  # File system operations
import numpy as np                          # Numerical operations
import pandas as pd                         # Dataframes for metadata/results
import tensorflow as tf                     # Deep learning framework
from tensorflow.keras.applications import ResNet50
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
prediction_folder = Path(r"/aakaou/ham10000/pipeline4/seg_masks1")
ground_truth_folder = Path(r"/aakaou/ham10000/pipeline4/seg_overlays1")
csv_output = Path(r"/aakaou/ham10000/pipeline4/segmentation_metrics_resnet50.csv")

metadata_file = Path(r"/aakaou/ham10000/HAM10000_metadata.csv")
original_images_folder = Path(r"/aakaou/ham10000/HAM10000_images_all")

# =========================
# CONSTANTS
# =========================
IMG_SIZE = (224, 224)       # Resize images for ResNet50 input
BATCH_SIZE = 16             # Batch size for training
EPOCHS = 10                 # Maximum number of epochs

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)   # Load metadata CSV (contains image IDs and labels)

# Define 7 HAM10000 lesion classes
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}  # Map class names to integers

# Map textual labels to integer IDs
df['label'] = df['dx'].map(class_map)
df['filename'] = df['image_id'] + ".jpg"   # Generate filenames for images

# Keep only rows where images exist
df = df[df['filename'].apply(lambda x: (original_images_folder / x).exists())]
print(f"✅ Images found: {len(df)}")

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,            # 20% for validation
    stratify=df['label'],     # Keep class distribution balanced
    random_state=42           # For reproducibility
)

# =========================
# DATA GENERATORS
# =========================
# Training data generator with augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalize images like ResNet50
    horizontal_flip=True,                     # Random horizontal flips
    zoom_range=0.2                            # Random zoom
)

# Validation data generator (no augmentation)
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Flow training data from dataframe
train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',        # Use integer labels directly
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
    shuffle=False            # Important to preserve order
)

# =========================
# RESNET50 MODEL
# =========================
base_model = ResNet50(
    weights="imagenet",       # Use pretrained ImageNet weights
    include_top=False,        # Remove fully connected top layers
    input_shape=(224, 224, 3)
)

# Freeze the pretrained backbone to avoid updating weights
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool feature maps to vector
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)              # Dropout for regularization
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)  # 7-class softmax output

# Create final model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ ResNet50 loaded (ImageNet pretrained)")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",      # Stop training if validation loss stops improving
    patience=3,              # Wait 3 epochs before stopping
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# PREDICTIONS ON VALIDATION SET
# =========================
val_data.reset()                       # Reset generator
y_true = val_df['label'].values        # True labels

probs = model.predict(val_data, verbose=1)     # Probabilities for each class
y_pred = np.argmax(probs, axis=1)             # Predicted class IDs
confidence = np.max(probs, axis=1)           # Maximum probability for each prediction

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

# Save individual class probabilities
for i, cls in enumerate(class_names):
    results_df[f"prob_{cls}"] = probs[:, i]

# Write CSV
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
plt.title("Confusion Matrix – ResNet50", fontweight="bold")
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

plt.plot([0,1], [0,1], "k--", lw=2)  # Diagonal line (random classifier)
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.title("ROC Curve – ResNet50 (HAM10000)", fontweight="bold")
plt.legend(prop={"weight": "bold"})
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
