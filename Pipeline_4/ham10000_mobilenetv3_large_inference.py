# =========================
# 1️⃣ Imports
# =========================
from pathlib import Path  # To handle file paths easily
import numpy as np        # Numerical computations and arrays
import pandas as pd       # For working with tabular data (metadata)
import tensorflow as tf   # Deep learning framework

# Pretrained model and preprocessing
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Layers for building classifier on top of base model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model  # Keras Model wrapper
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image loader & augmentation
from tensorflow.keras.callbacks import EarlyStopping  # Stop training if no improvement

# Evaluation
from sklearn.model_selection import train_test_split  # Split dataset into train/validation
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Metrics
from sklearn.preprocessing import label_binarize  # One-hot encode labels for ROC

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2️⃣ Paths & Hyperparameters
# =========================
ground_truth_folder = Path("/aakaou/ham10000/pipeline4/seg_overlays1")  # Overlay images folder
metadata_file = Path("/aakaou/ham10000/HAM10000_metadata.csv")         # CSV with metadata
csv_output = Path("/aakaou/ham10000/pipeline4/mobilenetv3large_results.csv")  # Output CSV

IMG_SIZE = (224, 224)  # Input size required for MobileNetV3
BATCH_SIZE = 16        # Number of images per batch
EPOCHS = 5             # Maximum number of epochs for training
NUM_CLASSES = 7        # Number of skin lesion classes

# =========================
# 3️⃣ Load metadata
# =========================
df = pd.read_csv(metadata_file)  # Read CSV into DataFrame

# Define class names and map them to integers
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

# Map 'dx' column to integers and create overlay filename column
df['label'] = df['dx'].map(class_map)          # Numeric labels
df['overlay_filename'] = df['image_id'] + ".png"  # Overlay image filename

# =========================
# 4️⃣ Filter existing overlay images
# =========================
df = df[df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())]  # Keep only files that exist
print(f"✅ Segmented overlay images found: {len(df)}")

if len(df) == 0:  # Safety check
    raise RuntimeError("❌ No images found. Check overlay folder path.")

# =========================
# 5️⃣ Train / Validation split
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,           # 20% of data for validation
    stratify=df['label'],    # Preserve class distribution
    random_state=42          # Reproducible split
)

# =========================
# 6️⃣ Data generators
# =========================
def create_generator(df, folder, augment=False):
    """
    Returns a Keras ImageDataGenerator that loads images from a DataFrame.
    
    Args:
        df: DataFrame with filenames and labels
        folder: Path to overlay images
        augment: If True, apply data augmentation
    Returns:
        generator yielding batches of images and labels
    """
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # MobileNetV3 preprocessing
            horizontal_flip=True,                     # Random horizontal flips
            rotation_range=20,                        # Random rotations
            zoom_range=0.25                            # Random zoom
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input  # Only preprocessing, no augmentation
        )

    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',  # Column with filenames
        y_col='label',             # Column with labels
        target_size=IMG_SIZE,      # Resize images
        class_mode='raw',          # Keep integer labels
        batch_size=BATCH_SIZE,
        shuffle=augment            # Shuffle only for training
    )

# Create train and validation generators
train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# 7️⃣ Build MobileNetV3 Large model
# =========================
base_model = MobileNetV3Large(
    weights='imagenet',       # Pretrained weights
    include_top=False,        # Remove default classifier
    input_shape=(224, 224, 3) # Input size
)
base_model.trainable = False   # Freeze base layers

# Add custom classifier on top
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = BatchNormalization()(x)       # Normalize activations
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)                   # Dropout for regularization
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax

model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # Adam optimizer
    loss='sparse_categorical_crossentropy',                 # Multi-class integer labels
    metrics=['accuracy']
)
print("✅ MobileNetV3 Large compiled successfully")

# =========================
# 8️⃣ Train the model
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=3,                # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Keep best weights
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
val_data.reset()                  # Reset generator for predictions
y_true = val_df['label'].values   # True labels
probs = model.predict(val_data, verbose=1)  # Model output probabilities
y_pred = np.argmax(probs, axis=1)          # Predicted class
confidence = np.max(probs, axis=1)        # Confidence of predictions

# =========================
# 🔟 Save results to CSV
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
# 1️⃣1️⃣ Classification report
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# 1️⃣2️⃣ Confusion matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,                 # Show numbers
    fmt='d',                    # Integer format
    cmap='Blues',
    xticklabels=class_names,    # X-axis labels
    yticklabels=class_names     # Y-axis labels
)
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("Actual", fontweight='bold')
plt.title("Confusion Matrix – MobileNetV3 Large", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC curves
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # One-hot encode labels

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # ROC curve
    roc_auc = auc(fpr, tpr)                                 # AUC score
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Reference diagonal
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – MobileNetV3 Large (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
