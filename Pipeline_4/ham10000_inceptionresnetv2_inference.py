# =========================
# 1️⃣ Imports
# =========================
from pathlib import Path             # For handling file paths easily
import numpy as np                   # For numerical operations
import pandas as pd                  # For working with dataframes
import tensorflow as tf              # Deep learning framework
from tensorflow.keras.applications import InceptionResNetV2  # Pretrained model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input  # Preprocessing function
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout  # Layers to build top model
from tensorflow.keras.models import Model                       # For defining custom models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.callbacks import EarlyStopping           # Stop training early if needed
from sklearn.model_selection import train_test_split           # Split dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # Evaluation metrics
from sklearn.preprocessing import label_binarize               # One-hot encoding for ROC
import matplotlib.pyplot as plt                                 # For plotting
import seaborn as sns                                           # Enhanced plots

# =========================
# 2️⃣ Paths & Parameters
# =========================
ground_truth_folder = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/seg_overlays1")  # Overlay folder path
metadata_file = Path("/home/aboubakr/Descargas/article4/ham10000/HAM10000_metadata.csv")         # Metadata CSV path
csv_output = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline4/inceptionresnetv2_results.csv")  # Output CSV

IMG_SIZE = (299, 299)  # Input size for InceptionResNetV2
BATCH_SIZE = 16        # Number of samples per batch
EPOCHS = 5             # Number of training epochs
NUM_CLASSES = 7        # Number of lesion classes

# =========================
# 3️⃣ Load Metadata
# =========================
df = pd.read_csv(metadata_file)  # Load the metadata CSV file

# Define lesion classes and map to numeric labels
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # Class names
class_map = {cls: i for i, cls in enumerate(class_names)}          # {'nv':0, 'mel':1,...}

df['label'] = df['dx'].map(class_map)          # Map lesion type to integer label
df['overlay_filename'] = df['image_id'] + ".png"  # Overlay filename (PNG)

# =========================
# 4️⃣ Filter existing files
# =========================
# Keep only rows where the overlay file exists in the folder
df = df[df['overlay_filename'].apply(lambda x: (ground_truth_folder / x).exists())]

print(f"✅ Segmented overlay images found: {len(df)}")

# Raise error if no files are found
if len(df) == 0:
    raise RuntimeError("❌ No images found. Check overlay folder path or extensions.")

# =========================
# 5️⃣ Train / Validation Split
# =========================
# Split data into training and validation while keeping class distribution
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
    Creates a Keras ImageDataGenerator for training or validation
    
    Args:
        df: DataFrame containing filenames and labels
        folder: Path containing the images
        augment: Whether to apply augmentation (for training)
    Returns:
        A Keras generator yielding batches of images and labels
    """
    if augment:
        # Training data augmentation
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # Normalize images
            horizontal_flip=True,                      # Random horizontal flip
            rotation_range=15,                         # Random rotation
            zoom_range=0.2                             # Random zoom
        )
    else:
        # Validation data, only preprocess
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    # Generate batches from dataframe
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='overlay_filename',  # Column with filenames
        y_col='label',             # Column with labels
        target_size=IMG_SIZE,      # Resize images
        class_mode='raw',          # Use numeric labels (not one-hot)
        batch_size=BATCH_SIZE,
        shuffle=augment            # Shuffle for training only
    )

# Create training and validation generators
train_data = create_generator(train_df, ground_truth_folder, augment=True)
val_data = create_generator(val_df, ground_truth_folder, augment=False)

# =========================
# 7️⃣ Build InceptionResNetV2 Model
# =========================
base_model = InceptionResNetV2(
    weights='imagenet',      # Pretrained ImageNet weights
    include_top=False,       # Exclude final classifier
    input_shape=(299, 299, 3)  # RGB overlay input
)

base_model.trainable = False  # Freeze pretrained layers

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)    # Pool features spatially
x = Dense(512, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)                 # Dropout to reduce overfitting
x = Dense(256, activation='relu')(x) # Another dense layer
outputs = Dense(NUM_CLASSES, activation='softmax')(x)  # Softmax for 7 classes

# Combine base and top into a model
model = Model(inputs=base_model.input, outputs=outputs)

# Compile model with optimizer and loss
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Integer labels
    metrics=['accuracy']
)

print("✅ InceptionResNetV2 compiled successfully")

# =========================
# 8️⃣ Train Model
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=3,                # Stop if no improvement for 3 epochs
    restore_best_weights=True  # Restore best weights at end
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
val_data.reset()               # Reset generator to start
y_true = val_df['label'].values  # True labels
probs = model.predict(val_data, verbose=1)  # Predicted probabilities
y_pred = np.argmax(probs, axis=1)           # Predicted classes
confidence = np.max(probs, axis=1)         # Max probability per prediction

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

# Save predicted probability for each class
for i, cls in enumerate(class_names):
    results_df[f'prob_{cls}'] = probs[:, i]

# Write to CSV
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
plt.title("Confusion Matrix – InceptionResNetV2", fontweight='bold')
plt.tight_layout()
plt.show()

# =========================
# 1️⃣3️⃣ Multi-class ROC Curves
# =========================
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))  # One-hot encoding

plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])  # ROC curve
    roc_auc = auc(fpr, tpr)                                 # Compute AUC
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line for reference
plt.xlabel("False Positive Rate", fontweight='bold')
plt.ylabel("True Positive Rate", fontweight='bold')
plt.title("ROC Curves – InceptionResNetV2 (HAM10000)", fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
