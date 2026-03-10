# ==========================
# 1️⃣ Imports
# =========================
from pathlib import Path             # For file path handling
import numpy as np                   # Numerical operations
import pandas as pd                  # Working with CSVs / DataFrames
import tensorflow as tf              # Deep learning
from tensorflow.keras.applications import MobileNet          # Pretrained MobileNetV1
from tensorflow.keras.applications.mobilenet import preprocess_input  # MobileNet preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Layers for top classifier
from tensorflow.keras.models import Model                   # For creating custom model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image generator
from tensorflow.keras.callbacks import EarlyStopping       # Stop training early if no improvement
from sklearn.model_selection import train_test_split       # Split dataset into train/val
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Metrics
from sklearn.preprocessing import label_binarize           # One-hot encoding for ROC
import matplotlib.pyplot as plt                             # Plotting library
import seaborn as sns                                       # Enhanced plotting

# =========================
# 2️⃣ Paths & Hyperparameters
# =========================
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Folder with overlay images
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")         # CSV file with labels
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/mobilenetv1_results.csv")  # CSV for predictions

IMG_SIZE = (224, 224)  # MobileNetV1 input size
BATCH_SIZE = 16        # Number of samples per batch
EPOCHS = 5             # Training epochs
NUM_CLASSES = 7        # Number of skin lesion classes

# =========================
# 3️⃣ Load Metadata
# =========================
df = pd.read_csv(metadata_file)  # Load CSV file with metadata

# Define class names and map to integer labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

df['label'] = df['dx'].map(class_map)                  # Map lesion type to integer
df['overlay_filename'] = df['image_id'] + ".png"      # Overlay filename

# =========================
# 4️⃣ Filter existing files
# =========================
# Keep only rows where overlay file exists in the folder
df = df[df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())]

print(f"✅ Segmented overlay images found: {len(df)}")

# Raise error if no files are found
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check overlay folder path or extensions.")

# =========================
# 5️⃣ Train / Validation Split
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,           # 20% validation split
    stratify=df['label'],    # Preserve class distribution
    random_state=42          # Reproducible split
)

# =========================
# 6️⃣ Data Generators
# =========================
def create_generator(df, folder, augment=False):
    """
    Creates a Keras ImageDataGenerator for training or validation
    
    Args:
        df: DataFrame containing filenames and labels
        folder: Path containing the images
        augment: Whether to apply augmentation (for training)
    Returns:
        A Keras generator yielding batches of images and labels
    """
    if augment:
        # Apply augmentation to training data
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # Preprocess images
            horizontal_flip=True,                     # Random horizontal flip
            rotation_range=15,                        # Random rotations
            zoom_range=0.2                            # Random zoom
        )
    else:
        # Validation data: only preprocessing
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    # Create generator from DataFrame
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',  # Filename column
        y_col='label',             # Label column
        target_size=IMG_SIZE,      # Resize images
        class_mode='raw',          # Return integer labels
        batch_size=BATCH_SIZE,
        shuffle=augment            # Shuffle only for training
    )

# Instantiate generators
train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# 7️⃣ Build MobileNetV1 Model
# =========================
base_model = MobileNet(
    weights='imagenet',       # Pretrained ImageNet weights
    include_top=False,        # Exclude final classification layer
    input_shape=(224, 224, 3), # Input shape matches overlay images
    alpha=1.0                  # Standard width multiplier
)

base_model.trainable = False  # Freeze pretrained weights

# Build classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool features to 1D
x = Dense(256, activation='relu')(x)  # Dense layer
x = Dropout(0.5)(x)                   # Dropout for regularization
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax

# Combine base and top into a model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Labels are integers
    metrics=['accuracy']
)

print("✅ MobileNetV1 compiled successfully")

# =========================
# 8️⃣ Train Model
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=3,                # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Restore best model weights
)

history = model.fit(
    train_data,               # Training generator
    validation_data=val_data, # Validation generator
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# 9️⃣ Predictions
# =========================
val_data.reset()                  # Reset validation generator
y_true = val_df['label'].values   # True labels
probs = model.predict(val_data, verbose=1)  # Predicted probabilities
y_pred = np.argmax(probs, axis=1)          # Predicted classes
confidence = np.max(probs, axis=1)        # Prediction confidence

# =========================
# 🔟 Save Results to CSV
# =========================
results_df = pd.DataFrame({
    'filename': val_df['overlay_filename'].values,
    'actual_class_id': y_true,
    'actual_class_name': [class_names[i] for i in y_true],
    'pred_class_id': y_pred,
    'pred_class_name': [class_names[i] for i in y_pred],
    'pred_confidence': confidence
})

# Add probability columns
for i, cls in enumerate(class_names):
    results_df[f'prob_{cls}'] = probs[:, i]

# Save to CSV
results_df.to_csv(csv_output, index=False)
print(f"✅ Results saved to: {csv_output}")

# =========================
# 1️⃣1️⃣ Classification Report
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# 1️⃣2️⃣ Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("Actual", fontweight='bold')
plt.title("Confusion Matrix – MobileNetV1", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC Curves
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # One-hot encoding

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # Compute ROC curve
    roc_auc = auc(fpr, tpr)                                 # Compute AUC
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – MobileNetV1 (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
