# =========================
# IMPORTS
# =========================
from pathlib import Path            # Path handling for files
import numpy as np                 # Numerical computations
import pandas as pd                # DataFrames for metadata & results
import tensorflow as tf            # Deep learning framework
import keras_cv                     # KerasCV library for pretrained backbones

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt     # Plotting
import seaborn as sns              # Advanced plotting (confusion matrix & heatmaps)

# =========================
# PATHS
# =========================
prediction_folder = Path("/aakaou/ham10000/pipeline3/seg_masks1")
ground_truth_folder = Path("/aakaou/ham10000/pipeline3/seg_overlays1")
csv_output = Path("/aakaou/ham10000/pipeline3/segmentation_metrics.csv")

metadata_file = Path("/aakaou/ham10000/HAM10000_metadata.csv")
original_images_folder = Path("/aakaou/ham10000/HAM10000_images_all")

# =========================
# CONSTANTS
# =========================
IMG_SIZE = (224, 224)   # Image size for ResNet34
BATCH_SIZE = 8           # Smaller batch due to GPU memory constraints
EPOCHS = 10              # Number of training epochs

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)             # Load CSV containing HAM10000 labels

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']   # 7 skin lesion classes
class_map = {cls: i for i, cls in enumerate(class_names)}          # Map class names -> integer IDs

df["label"] = df["dx"].map(class_map)    # Create numerical label column
df["filename"] = df["image_id"] + ".jpg" # Generate filenames for images

# Filter to keep only images that exist
df = df[df["filename"].apply(lambda x: (original_images_folder / x).exists())]
print(f"✅ Images found: {len(df)}")

# =========================
# TRAIN / VAL SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,              # 20% for validation
    stratify=df["label"],       # Ensure class distribution remains balanced
    random_state=0
)

# =========================
# DATA GENERATORS
# =========================
# Training generator with augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Normalize inputs for ResNet
    horizontal_flip=True,                     # Random horizontal flips
    zoom_range=0.2                            # Random zoom
)

# Validation generator (no augmentation)
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Flow training data from DataFrame
train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",       # Use integer labels
    batch_size=BATCH_SIZE
)

# Flow validation data
val_data = val_gen.flow_from_dataframe(
    val_df,
    directory=original_images_folder,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False            # No shuffle to preserve order
)

# =========================
# RESNET34 BACKBONE (KerasCV)
# =========================
base_model = keras_cv.models.ResNetBackbone.from_preset(
    "resnet34_imagenet",  # Pretrained ResNet34 on ImageNet
    load_weights=True
)
base_model.trainable = False      # Freeze backbone

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool feature maps to vector
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)              # Dropout to reduce overfitting
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)  # 7-class softmax output

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ ResNet34 loaded successfully (keras-cv preset)")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",      # Stop if val_loss doesn't improve
    patience=3,              # Wait 3 epochs
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
val_data.reset()                   # Reset generator
y_true = val_df["label"].values    # True labels

probs = model.predict(val_data, verbose=1)       # Probabilities
y_pred = np.argmax(probs, axis=1)               # Predicted class IDs
confidence = np.max(probs, axis=1)             # Max probability per sample

# =========================
# SAVE RESULTS CSV
# =========================
results_df = pd.DataFrame({
    "filename": val_df["filename"].values,
    "actual_class_id": y_true,
    "actual_class_name": [class_names[i] for i in y_true],
    "pred_class_id": y_pred,
    "pred_class_name": [class_names[i] for i in y_pred],
    "pred_confidence": confidence
})

# Add class probabilities
for i, cls in enumerate(class_names):
    results_df[f"prob_{cls}"] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Results saved to: {csv_output}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
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
plt.title("Confusion Matrix – ResNet34 (HAM10000)", fontweight="bold")
plt.tight_layout()
plt.show()

# =========================
# ROC CURVES (7-CLASS)
# =========================
y_true_bin = label_binarize(y_true, classes=range(7))  # One-hot encode

plt.figure(figsize=(10, 8))

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)  # Diagonal line
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.title("ROC Curves – ResNet34 (HAM10000)", fontweight="bold")
plt.legend(prop={"weight": "bold"})
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
