# =========================
# 1️⃣ Imports
# =========================
from pathlib import Path             # File path handling
import numpy as np                   # Numerical operations
import pandas as pd                  # CSV and DataFrame handling
import tensorflow as tf              # Deep learning
from tensorflow.keras.applications import MobileNetV2          # Pretrained MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Preprocess images
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Layers for classifier
from tensorflow.keras.models import Model                   # Custom model wrapper
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data generator
from tensorflow.keras.callbacks import EarlyStopping       # Early stopping
from sklearn.model_selection import train_test_split       # Split dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Metrics
from sklearn.preprocessing import label_binarize           # One-hot encoding for ROC
import matplotlib.pyplot as plt                             # Plotting
import seaborn as sns                                       # Enhanced plots

# =========================
# 2️⃣ Paths & Hyperparameters
# =========================
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Overlay images
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")         # Metadata CSV
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/mobilenetv2_results.csv")  # CSV output

IMG_SIZE = (224, 224)  # MobileNetV2 input size
BATCH_SIZE = 16        # Samples per batch
EPOCHS = 5             # Training epochs
NUM_CLASSES = 7        # Skin lesion categories

# =========================
# 3️⃣ Load Metadata
# =========================
df = pd.read_csv(metadata_file)  # Load CSV file

# Define class names and map to integers
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

df['label'] = df['dx'].map(class_map)                  # Map lesion type to integer
df['overlay_filename'] = df['image_id'] + ".png"      # Construct overlay filename

# =========================
# 4️⃣ Filter existing overlay files
# =========================
df = df[df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())]

print(f"✅ Segmented overlay images found: {len(df)}")

if len(df) == 0:
    raise RuntimeError("❌ No images found. Check overlay folder path.")

# =========================
# 5️⃣ Train / Validation Split
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,           # 20% validation
    stratify=df['label'],    # Maintain class distribution
    random_state=42          # Reproducibility
)

# =========================
# 6️⃣ Data Generators
# =========================
def create_generator(df, folder, augment=False):
    """
    Creates a Keras ImageDataGenerator from DataFrame
    
    Args:
        df: DataFrame containing filenames and labels
        folder: Folder containing overlay images
        augment: Apply augmentation (for training)
    Returns:
        Generator yielding batches of images and labels
    """
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # Preprocess images
            horizontal_flip=True,                     # Random horizontal flip
            rotation_range=15,                        # Random rotation
            zoom_range=0.2                            # Random zoom
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input   # Only preprocess
        )

    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',  # Filename column
        y_col='label',             # Label column
        target_size=IMG_SIZE,      # Resize images
        class_mode='raw',          # Integer labels
        batch_size=BATCH_SIZE,
        shuffle=augment            # Shuffle only for training
    )

# Create train & validation generators
train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# 7️⃣ Build MobileNetV2 Model
# =========================
base_model = MobileNetV2(
    weights='imagenet',       # Pretrained weights
    include_top=False,        # Exclude classifier
    input_shape=(224, 224, 3) # Input size
)

base_model.trainable = False  # Freeze base layers

# Add custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Pool features to 1D
x = Dense(128, activation='relu')(x)  # Dense layer
x = Dropout(0.4)(x)                   # Dropout
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer

model = Model(inputs=base_model.input, outputs=outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ MobileNetV2 compiled successfully")

# =========================
# 8️⃣ Train Model
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,                # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Keep best model
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
val_data.reset()                  # Reset generator
y_true = val_df['label'].values   # True labels
probs = model.predict(val_data, verbose=1)  # Probabilities
y_pred = np.argmax(probs, axis=1)          # Predicted class
confidence = np.max(probs, axis=1)        # Confidence

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
plt.title("Confusion Matrix – MobileNetV2", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC Curves
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # One-hot encode labels

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # ROC curve
    roc_auc = auc(fpr, tpr)                                 # AUC score
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Reference line
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – MobileNetV2 (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
