# =========================
# 1️⃣ Imports
# =========================
from pathlib import Path             # Handle file paths
import numpy as np                   # Numerical operations
import pandas as pd                  # DataFrame manipulation
import tensorflow as tf              # Deep learning framework
from tensorflow.keras.applications import InceptionV3          # Pretrained model
from tensorflow.keras.applications.inception_v3 import preprocess_input  # Preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Layers
from tensorflow.keras.models import Model                         # Model API
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation
from tensorflow.keras.callbacks import EarlyStopping               # Early stopping
from sklearn.model_selection import train_test_split               # Split data
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Metrics
from sklearn.preprocessing import label_binarize                   # For multi-class ROC
import matplotlib.pyplot as plt                                     # Plotting
import seaborn as sns                                               # Enhanced plotting

# =========================
# 2️⃣ Paths & Parameters
# =========================
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Overlay folder
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")         # Metadata CSV
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/inceptionv3_results.csv")  # CSV output

IMG_SIZE = (299, 299)  # InceptionV3 default input size
BATCH_SIZE = 16        # Number of samples per batch
EPOCHS = 5             # Number of training epochs
NUM_CLASSES = 7        # Number of lesion classes

# =========================
# 3️⃣ Load Metadata
# =========================
df = pd.read_csv(metadata_file)  # Read metadata CSV

# Define lesion classes and mapping to numeric labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}  # {'nv':0, 'mel':1,...}

df['label'] = df['dx'].map(class_map)  # Map lesion type to numeric label

# Create overlay filenames (PNG)
df['overlay_filename'] = df['image_id'] + ".png"

# =========================
# 4️⃣ Filter existing files
# =========================
# Keep only rows where the overlay file exists
df = df[df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())]

print(f"✅ Segmented overlay images found: {len(df)}")

# Raise error if no files are found
if len(df) == 0:
    raise RuntimeError(
        "❌ No images found. Check overlay folder path and PNG extensions."
    )

# =========================
# 5️⃣ Train/Validation Split
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,           # 20% validation
    stratify=df['label'],    # Keep class distribution
    random_state=42           # Reproducible split
)

# =========================
# 6️⃣ Data Generators
# =========================
def create_generator(df, folder, augment=False):
    """
    Creates an ImageDataGenerator from a dataframe.
    
    Args:
        df: DataFrame containing filenames and labels
        folder: folder containing the images
        augment: whether to apply augmentation
    Returns:
        A Keras ImageDataGenerator iterator
    """
    if augment:
        # Apply augmentation for training
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # Normalize for InceptionV3
            horizontal_flip=True,                      # Flip images
            rotation_range=15,                         # Rotate ±15°
            zoom_range=0.2                             # Random zoom
        )
    else:
        # No augmentation for validation
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    # Generate batches from dataframe
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',
        y_col='label',
        target_size=IMG_SIZE,
        class_mode='raw',   # Use numeric labels
        batch_size=BATCH_SIZE,
        shuffle=augment     # Shuffle if training
    )

# Create training and validation generators
train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# 7️⃣ Build InceptionV3 Model
# =========================
base_model = InceptionV3(
    weights='imagenet',      # Load pretrained ImageNet weights
    include_top=False,       # Exclude default classifier
    input_shape=(299, 299, 3) # Input shape matches overlay RGB
)

base_model.trainable = False  # Freeze pretrained layers

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)    # Global average pooling
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)                 # Dropout for regularization
x = Dense(256, activation='relu')(x) # Fully connected layer
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Softmax output for 7 classes

# Define full model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Multi-class integer labels
    metrics=['accuracy']
)

print("✅ InceptionV3 compiled successfully")

# =========================
# 8️⃣ Train Model
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=3,           # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Restore best model
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# 9️⃣ Predictions
# =========================
val_data.reset()  # Reset generator for prediction
y_true = val_df['label'].values   # True labels
probs = model.predict(val_data, verbose=1)  # Predicted probabilities
y_pred = np.argmax(probs, axis=1)           # Predicted classes
confidence = np.max(probs, axis=1)         # Max probability per sample

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

# Add probability columns for each class
for i, cls in enumerate(class_names):
    results_df[f'prob_{cls}'] = probs[:, i]

# Save CSV
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
plt.title("Confusion Matrix – InceptionV3", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC Curves
# =========================
# Convert true labels to one-hot encoding
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # Compute ROC
    roc_auc = auc(fpr, tpr)                                 # Compute AUC
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – InceptionV3 (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
