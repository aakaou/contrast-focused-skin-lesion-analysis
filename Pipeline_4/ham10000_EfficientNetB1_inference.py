rom pathlib import Path
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Conv2D, Concatenate
from tensorflow.keras.models import Model
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
original_images_folder = Path(r"/aakaou/ham10000/HAM10000_images_all")
csv_output = Path(r"/aakaou/ham10000/pipeline4/efficientnetb1_results.csv")
metadata_file = Path(r"/aakaou/ham10000/HAM10000_metadata.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 7

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}
df['label'] = df['dx'].map(class_map)
df['filename'] = df['image_id'] + ".jpg"
df['mask_filename'] = df['image_id'] + ".png"
df['overlay_filename'] = df['image_id'] + ".png"

# Keep only rows where all three files exist
df = df[
    df['filename'].apply(lambda x: (original_images_folder / x).exists()) &
    df['mask_filename'].apply(lambda x: (prediction_folder / x).exists()) &
    df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())
]
print(f"✅ Images found for training: {len(df)}")
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check filenames in all three folders")

# =========================
# SPLIT TRAIN/VALIDATION
# =========================
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=0)

# =========================
# CUSTOM LOADER
# =========================
def load_image_pair(original_file, mask_file, overlay_file):
    # Original RGB
    img = cv2.imread(str(original_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    
    # Mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, IMG_SIZE)
    mask = mask[..., np.newaxis]  # add channel
    
    # Overlay
    overlay = cv2.imread(str(overlay_file))
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    overlay = cv2.resize(overlay, IMG_SIZE)
    
    # Concatenate as 7 channels: RGB + overlay(3) + mask(1)
    combined = np.concatenate([img, overlay, mask], axis=-1)
    # Scale only RGB + overlay channels using preprocess_input
    combined[..., :6] = preprocess_input(combined[..., :6])
    
    return combined

# =========================
# DATA GENERATOR
# =========================
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        X = np.array([load_image_pair(
            original_images_folder / row['filename'],
            prediction_folder / row['mask_filename'],
            ground_truth_folder / row['overlay_filename']
        ) for _, row in batch_df.iterrows()])
        y = batch_df['label'].values
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

train_gen = CustomDataGenerator(train_df)
val_gen = CustomDataGenerator(val_df, shuffle=False)

# =========================
# MODEL
# =========================
input_layer = Input(shape=(224,224,7))  # RGB + overlay + mask

# Split first 3 channels (RGB+overlay) to pre-trained EfficientNetB1
rgb_overlay_input = input_layer[..., :6]  # first 6 channels
mask_input = input_layer[..., 6:]  # last channel

# EfficientNetB1 on first 6 channels (we will replicate weights for 6 channels)
base_model = EfficientNetB1(include_top=False, weights=None, input_shape=(224,224,6))  # weights=None because channels !=3

x = base_model(rgb_overlay_input)
x = GlobalAveragePooling2D()(x)

# Process mask input
mask_branch = Conv2D(16, (3,3), activation='relu', padding='same')(mask_input)
mask_branch = GlobalAveragePooling2D()(mask_branch)

# Concatenate both
x = Concatenate()([x, mask_branch])
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("✅ EfficientNetB1 with RGB+overlay+mask ready")

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[early_stop])

# =========================
# PREDICTIONS & CSV
# =========================
y_true = val_df['label'].values
probs = model.predict(val_gen)
y_pred = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)

results_df = pd.DataFrame({
    "filename": val_df['filename'].values,
    "actual_class_id": y_true,
    "actual_class_name": [class_names[i] for i in y_true],
    "pred_class_id": y_pred,
    "pred_class_name": [class_names[i] for i in y_pred],
    "pred_confidence": confidence
})
for i, cls in enumerate(class_names):
    results_df[f"prob_{cls}"] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Predictions saved to {csv_output}")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EfficientNetB1")
plt.show()

# =========================
# ROC CURVES
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
plt.figure(figsize=(10,8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - EfficientNetB1 (HAM10000)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\n📊 Classification Report (EfficientNetB1):\n")
print(report)

