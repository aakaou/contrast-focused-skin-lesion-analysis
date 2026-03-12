from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Concatenate, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# =========================
# PATHS
# =========================
original_images_folder = Path("/aakaou/ham10000/HAM10000_images_all")
ground_truth_folder = Path("/aakaou/ham10000/pipeline4/seg_overlays1")
prediction_folder = Path("/aakaou/ham10000/pipeline4/seg_masks1")
metadata_file = Path("/aakaou/ham10000/HAM10000_metadata.csv")
csv_output = Path("/aakaou/ham10000/pipeline4/densenet169_results.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 7

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

df['label'] = df['dx'].map(class_map)
df['filename'] = df['image_id'] + ".jpg"
df['overlay_filename'] = df['image_id'] + ".png"  # overlays are .png
df['mask_filename'] = df['image_id'] + ".png"     # masks are .png

# Keep only images where all files exist
df = df[
    df['filename'].apply(lambda x: (original_images_folder / x).exists()) &
    df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists()) &
    df['mask_filename'].apply(lambda x: (prediction_folder / x).exists())
]

print(f"✅ Total usable images: {len(df)}")
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check filenames and extensions in all folders")

# =========================
# SPLIT TRAIN/VALIDATION
# =========================
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# =========================
# CUSTOM IMAGE LOADER
# =========================
def load_images(original_file, overlay_file, mask_file):
    # Original RGB
    img = cv2.imread(str(original_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    # Overlay RGB
    overlay = cv2.imread(str(overlay_file))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, IMG_SIZE)

    # Mask grayscale
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = mask[..., np.newaxis] / 255.0

    # Preprocess RGB + overlay (first 6 channels)
    combined = np.concatenate([img, overlay, mask], axis=-1)
    combined[..., :6] = preprocess_input(combined[..., :6])

    return combined

# =========================
# CUSTOM GENERATOR
# =========================
class CustomGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]

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
        if self.shuffle:
            np.random.shuffle(self.indexes)

train_gen = CustomGenerator(train_df)
val_gen = CustomGenerator(val_df, shuffle=False)

# =========================
# DENSENET169 MODEL
# =========================
input_layer = Input(shape=(224,224,7))  # RGB + overlay + mask

# Split RGB+overlay and mask channels
rgb_overlay_input = input_layer[..., :6]
mask_input = input_layer[..., 6:]

# DenseNet169 on 6 channels (cannot load pretrained ImageNet weights for 6 channels)
base_model = DenseNet169(weights=None, include_top=False, input_shape=(224,224,6))
x = base_model(rgb_overlay_input)
x = GlobalAveragePooling2D()(x)

# Mask branch
mask_branch = Conv2D(16, 3, activation='relu', padding='same')(mask_input)
mask_branch = GlobalAveragePooling2D()(mask_branch)

# Concatenate features
x = Concatenate()([x, mask_branch])
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ DenseNet169 ready")

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
plt.title("Confusion Matrix - DenseNet169")
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
plt.title("ROC Curve - DenseNet169")
plt.legend()
plt.show()


