# =========================
# IMPORTS
# =========================
from pathlib import Path                # Path handling for folders/files
import numpy as np                      # Numerical operations
import pandas as pd                     # CSV reading/writing
import tensorflow as tf                 # TensorFlow for deep learning
from tensorflow.keras.applications import DenseNet121  # DenseNet121 backbone
from tensorflow.keras.applications.densenet import preprocess_input  # Preprocessing for DenseNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Concatenate, Conv2D  # Layers
from tensorflow.keras.models import Model  # Keras Model API
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Standard generators (not used here)
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping
from sklearn.model_selection import train_test_split  # Train/val split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Evaluation metrics
from sklearn.preprocessing import label_binarize  # One-hot encoding for ROC
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns            # Heatmaps
import cv2                       # Image I/O and resizing

# =========================
# PATHS
# =========================
original_images_folder = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_images_all")  # Original RGB images
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Overlay masks
prediction_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_masks1")       # Predicted masks
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")         # Metadata CSV
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/densenet121_results.csv")  # Output CSV

# =========================
# CONSTANTS
# =========================
IMG_SIZE = (224, 224)  # Resize images to 224x224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 7         # Number of lesion classes

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)  # Load metadata
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # Classes
class_map = {cls: i for i, cls in enumerate(class_names)}  # Map class name -> int

# Add numeric label
df['label'] = df['dx'].map(class_map)
# Filename for original RGB images
df['filename'] = df['image_id'] + ".jpg"
# Filename for overlay masks
df['overlay_filename'] = df['image_id'] + ".png"
# Filename for predicted masks
df['mask_filename'] = df['image_id'] + ".png"

# Keep only rows where all required files exist
df = df[
    df['filename'].apply(lambda x: (original_images_folder / x).exists()) &
    df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists()) &
    df['mask_filename'].apply(lambda x: (prediction_folder / x).exists())
]
print(f"✅ Total usable images: {len(df)}")
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check filenames and extensions in all folders")

# =========================
# TRAIN/VALIDATION SPLIT
# =========================
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# =========================
# CUSTOM IMAGE LOADER
# =========================
def load_images(original_file, overlay_file, mask_file):
    """
    Load original RGB image, overlay RGB image, and mask.
    Combine them into a single 7-channel array (RGB+overlay+mask).
    """
    # Original RGB image
    img = cv2.imread(str(original_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # Overlay image (RGB)
    overlay = cv2.imread(str(overlay_file))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, IMG_SIZE)

    # Mask image (grayscale), scale to [0,1]
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = mask[..., np.newaxis] / 255.0

    # Concatenate RGB + overlay + mask into one array (224x224x7)
    combined = np.concatenate([img, overlay, mask], axis=-1)
    # Preprocess first 6 channels (RGB + overlay) like DenseNet
    combined[..., :6] = preprocess_input(combined[..., :6])

    return combined

# =========================
# CUSTOM GENERATOR
# =========================
class CustomGenerator(tf.keras.utils.Sequence):
    """
    Custom Keras Sequence generator to yield batches of combined images and labels.
    """
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        # Generate one batch
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

        # Load images for this batch
        X = np.array([
            load_images(
                original_images_folder / row['filename'],
                ground_truth_folder / row['overlay_filename'],
                prediction_folder / row['mask_filename']
            ) for _, row in batch_df.iterrows()
        ])
        y = batch_df['label'].values
        return X, y

    def on_epoch_end(self):
        # Shuffle indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create train and validation generators
train_gen = CustomGenerator(train_df)
val_gen = CustomGenerator(val_df, shuffle=False)

# =========================
# DENSENET121 MODEL
# =========================
input_layer = Input(shape=(224,224,7))  # 7-channel input (RGB + overlay + mask)

# Split channels: first 6 for DenseNet, last 1 for mask branch
rgb_overlay_input = input_layer[..., :6]
mask_input = input_layer[..., 6:]

# DenseNet121 backbone (6 channels, no pretrained weights)
base_model = DenseNet121(weights=None, include_top=False, input_shape=(224,224,6))
x = base_model(rgb_overlay_input)
x = GlobalAveragePooling2D()(x)

# Simple mask branch
mask_branch = Conv2D(16, 3, activation='relu', padding='same')(mask_input)
mask_branch = GlobalAveragePooling2D()(mask_branch)

# Concatenate DenseNet features and mask features
x = Concatenate()([x, mask_branch])
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Complete model
model = Model(inputs=input_layer, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ DenseNet121 ready")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop])

# =========================
# PREDICTIONS
# =========================
y_true = val_df['label'].values
probs = model.predict(val_gen)
y_pred = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)

# =========================
# SAVE RESULTS
# =========================
results_df = pd.DataFrame({
    'filename': val_df['filename'].values,
    'actual_class_id': y_true,
    'actual_class_name': [class_names[i] for i in y_true],
    'pred_class_id': y_pred,
    'pred_class_name': [class_names[i] for i in y_pred],
    'pred_confidence': confidence
})
for i, cls in enumerate(class_names):
    results_df[f'prob_{cls}'] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Predictions saved to: {csv_output}")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - DenseNet121")
plt.show()

# =========================
# ROC CURVES
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
plt.figure(figsize=(10,8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - DenseNet121")
plt.legend()
plt.show()
