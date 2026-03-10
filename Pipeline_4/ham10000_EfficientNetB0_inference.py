# =========================
# 1️⃣ Imports
# =========================
from pathlib import Path      # For handling file paths
import numpy as np           # Numerical computations
import pandas as pd          # Data manipulation (metadata CSV)
import tensorflow as tf      # Deep learning framework

# EfficientNetB0 and preprocessing
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# Layers and model API
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Evaluation and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2️⃣ Paths & hyperparameters
# =========================
overlay_dir = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Overlay images folder
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")  # Metadata CSV
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/efficientnetb0_results.csv")  # Output CSV

IMG_SIZE = (224, 224)  # Input image size for EfficientNetB0
BATCH_SIZE = 16        # Batch size
EPOCHS = 15            # Training epochs (stage 1)
NUM_CLASSES = 7        # Number of skin lesion classes

# =========================
# 3️⃣ Load metadata
# =========================
df = pd.read_csv(metadata_file)  # Read CSV

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {c: i for i, c in enumerate(class_names)}  # Map class name → int

df['label'] = df['dx'].map(class_map)  # Numeric labels
df['filename'] = df['image_id'] + ".png"  # Overlay filenames

# Keep only images that actually exist
df = df[df['filename'].apply(lambda x: (overlay_dir / x).exists())]

print(f"✅ Segmented images found: {len(df)}")
if len(df) < 50:
    raise RuntimeError("❌ Too few images — check seg_overlays1 filenames")

# =========================
# 4️⃣ Train / Validation split
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,        # 20% validation
    stratify=df['label'], # Keep class distribution
    random_state=42
)

# =========================
# 5️⃣ Data generators
# =========================
# Training generator with augmentation
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
    horizontal_flip=True,                     # Random horizontal flip
    rotation_range=15,                        # Random rotation
    zoom_range=0.2                             # Random zoom
).flow_from_dataframe(
    train_df,
    directory=overlay_dir,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",  # Integer labels
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Validation generator (no augmentation)
val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_dataframe(
    val_df,
    directory=overlay_dir,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="raw",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# 6️⃣ Model definition
# =========================
base_model = EfficientNetB0(
    include_top=False,      # Remove classifier
    weights="imagenet",     # Pretrained weights
    input_shape=(224,224,3)
)
base_model.trainable = False  # Freeze base model initially

# Add custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = Dense(512, activation="relu")(x)  # Fully connected layer
x = Dropout(0.5)(x)                  # Regularization
x = Dense(256, activation="relu")(x) # Additional dense layer
outputs = Dense(NUM_CLASSES, activation="softmax")(x)  # Final output

model = Model(base_model.input, outputs)

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ EfficientNetB0 compiled")

# =========================
# 7️⃣ Train stage 1 (frozen backbone)
# =========================
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# 8️⃣ Fine-tuning stage 2
# =========================
print("🔓 Fine-tuning...")

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Smaller LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop]
)

# =========================
# 9️⃣ Predictions
# =========================
val_gen.reset()                    # Reset generator
probs = model.predict(val_gen, verbose=1)  # Probabilities
y_pred = np.argmax(probs, axis=1)         # Predicted class
y_true = val_df['label'].values           # True class
confidence = np.max(probs, axis=1)       # Max confidence per sample

# =========================
# 🔟 Save results CSV
# =========================
results_df = pd.DataFrame({
    "filename": val_df["filename"].values,
    "actual_class": y_true,
    "pred_class": y_pred,
    "confidence": confidence
})

# Add probability columns
for i, cls in enumerate(class_names):
    results_df[f"prob_{cls}"] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Results saved to {csv_output}")

# =========================
# 1️⃣1️⃣ Classification report
# =========================
print("\n📊 Classification Report\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# =========================
# 1️⃣2️⃣ Confusion matrix
# =========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – EfficientNetB0")
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC–AUC
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # One-hot encode

plt.figure(figsize=(10,8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # ROC curve
    roc_auc = auc(fpr, tpr)                                 # AUC
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0,1], [0,1], 'k--')  # Diagonal reference
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – EfficientNetB0 (HAM10000)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
