from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
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
ground_truth_folder = Path("/aakaou/ham10000/pipeline4/seg_overlays1")
metadata_file = Path("/aakaou/ham10000/HAM10000_metadata.csv")
csv_output = Path("/aakaou/ham10000/pipeline4/efficientnetb3_results.csv")

IMG_SIZE = (300, 300)   # EfficientNetB3 input
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 7

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

df['label'] = df['dx'].map(class_map)
df['overlay_filename'] = df['image_id'] + ".png"

# =========================
# FILTER EXISTING FILES
# =========================
df = df[df['overlay_filename'].apply(
    lambda x: (ground_truth_folder / x).exists()
)]

print(f"✅ Segmented overlay images found: {len(df)}")

if len(df) == 0:
    raise RuntimeError("❌ No images found. Check overlay folder path or extension.")

# =========================
# TRAIN / VALIDATION SPLIT
# =========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# =========================
# DATA GENERATORS
# =========================
def create_generator(df, folder, augment=False):
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=30,
            zoom_range=0.35,
            width_shift_range=0.07,
            height_shift_range=0.07
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',
        y_col='label',
        target_size=IMG_SIZE,
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=augment
    )

train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# EFFICIENTNETB3 MODEL
# =========================
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ EfficientNetB3 compiled successfully")

# =========================
# TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# =========================
# PREDICTIONS
# =========================
val_data.reset()

y_true = val_df['label'].values
probs = model.predict(val_data, verbose=1)

y_pred = np.argmax(probs, axis=1)
confidence = np.max(probs, axis=1)

# =========================
# SAVE RESULTS CSV
# =========================
results_df = pd.DataFrame({
    'filename': val_df['overlay_filename'].values,
    'actual_class_id': y_true,
    'actual_class_name': [class_names[i] for i in y_true],
    'pred_class_id': y_pred,
    'pred_class_name': [class_names[i] for i in y_pred],
    'pred_confidence': confidence
})

for i, cls in enumerate(class_names):
    results_df[f'prob_{cls}'] = probs[:, i]

results_df.to_csv(csv_output, index=False)
print(f"✅ Results saved to: {csv_output}")

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
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
plt.title("Confusion Matrix – EfficientNetB3", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# MULTI-CLASS ROC CURVES
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – EfficientNetB3 (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
