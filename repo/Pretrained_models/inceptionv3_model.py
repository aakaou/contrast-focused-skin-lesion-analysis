# ======================================================
# inceptionv3_all_pipelines.py
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (299, 299)  # InceptionV3 default input size
BATCH_SIZE = 16        # Number of samples per batch
EPOCHS = 5             # Number of training epochs
NUM_CLASSES = 7        # Number of lesion classes

metadata_file = "/aakaou/HAM10000_metadata.csv"

# =========================
# PIPELINES
# =========================

pipelines_info = [

    {
        "images": r"/aakaou/pipeline1_seg_overlays",
        "csv": r"/aakaou/pipeline1_inceptionv3_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline2_seg_overlays",
        "csv": r"/aakaou/pipeline2_inceptionv3_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline3_seg_overlays",
        "csv": r"/aakaou/pipeline3_inceptionv3_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline4_overlays",
        "csv": r"/aakaou/pipeline4_inceptionv3_predictions.csv"
    }

]

# =========================
# LOAD METADATA
# =========================

meta_df = pd.read_csv(metadata_file)

class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']
class_map = {cls: i for i, cls in enumerate(class_names)}

meta_df["label"] = meta_df["dx"].map(class_map)
meta_df["filename"] = meta_df["image_id"] + ".jpg"  # Keep .jpg for consistency

# ======================================================
# LOOP THROUGH PIPELINES
# ======================================================

def create_generator(df, folder, augment=False):
    """Creates an ImageDataGenerator from a dataframe."""
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            rotation_range=15,
            zoom_range=0.2
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    
    return datagen.flow_from_dataframe(
        dataframe=df,
        directory=folder,
        x_col='filename',
        y_col='label',
        target_size=IMG_SIZE,
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=augment
    )

for pipe in pipelines_info:

    print(f"\n🚀 Training InceptionV3 on {pipe['images']}")

    image_folder = Path(pipe["images"])

    # =========================
    # FILTER EXISTING IMAGES
    # =========================

    df = meta_df.copy()
    df = df[df["filename"].apply(lambda x: (image_folder / x).exists())]

    print(f"✅ Images found: {len(df)}")

    if len(df) == 0:
        print(f"❌ No images found in {image_folder}")
        continue

    # =========================
    # TRAIN / VAL SPLIT
    # =========================

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # =========================
    # DATA GENERATORS
    # =========================

    train_data = create_generator(train_df, image_folder, augment=True)
    val_data = create_generator(val_df, image_folder, augment=False)

    # =========================
    # INCEPTIONV3 BACKBONE
    # =========================

    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False

    # Classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ InceptionV3 model ready")

    # =========================
    # TRAIN MODEL
    # =========================

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model.fit(
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
    # SAVE RESULTS
    # =========================

    results = pd.DataFrame({
        "filename": val_df["filename"].values,
        "actual_class_id": y_true,
        "actual_class_name": [class_names[i] for i in y_true],
        "pred_class_id": y_pred,
        "pred_class_name": [class_names[i] for i in y_pred],
        "confidence": confidence
    })

    for i, name in enumerate(class_names):
        results[f"prob_{name}"] = probs[:, i]

    results.to_csv(pipe["csv"], index=False)
    print(f"✅ Predictions saved → {pipe['csv']}")

print("\n🎯 All pipelines completed")

