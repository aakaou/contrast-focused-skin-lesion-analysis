from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
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
prediction_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_masks1")
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline3/seg_overlays1")
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline3/segmentation_metrics.csv")

metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")
original_images_folder = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_images_all")

IMG_SIZE = (224, 224)
BATCH_SIZE = 8        # ⬅️ Reduced for ResNet152
EPOCHS = 10

# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(metadata_file)

class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
class_map = {cls: i for i, cls in enumerate(class_names)}

df['label'] = df['dx'].map(class_map)
df['filename'] = df['image_id'] + ".jpg"

df = df[df['filename'].apply(lambda x: (original_images_folder / x).exists())]
print(f"✅ Images found: {len(df)}")

# =========================
# SPLIT
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
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    zoom_range=0.2
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE
)

val_data = val_gen.flow_from_dataframe(
    val_df,
    directory=original_images_folder,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================
# RESNET152 MODEL
# =========================
base_model = ResNet152(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("✅ ResNet152 loaded (ImageNet pretrained)")

# =========================
# TRAIN
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
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
# SAVE CSV
# =========================
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
print(f"✅ Predictions saved to: {csv_output}")

# =========================
# REPORT & PLOTS
# =========================
print("\n📊 Classification Report\n")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted", fontweight="bold")
plt.ylabel("Actual", fontweight="bold")
plt.title("Confusion Matrix – ResNet152", fontweight="bold")
plt.tight_layout()
plt.show()

y_true_bin = label_binarize(y_true, classes=range(7))

plt.figure(figsize=(10,8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1], "k--", lw=2)
plt.xlabel("False Positive Rate", fontweight="bold")
plt.ylabel("True Positive Rate", fontweight="bold")
plt.title("ROC Curve – ResNet152 (HAM10000)", fontweight="bold")
plt.legend(prop={"weight": "bold"})
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
