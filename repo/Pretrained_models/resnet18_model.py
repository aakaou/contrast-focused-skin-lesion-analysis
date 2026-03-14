# ======================================================
# resnet18_all_pipelines.py
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

metadata_file = "/aakaou/HAM10000_metadata.csv"

# =========================
# PIPELINES
# =========================

pipelines_info = [

    {
        "images": r"/aakaou/pipeline1_seg_overlays",
        "csv": r"/aakaou/pipeline1_resnet18_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline2_seg_overlays",
        "csv": r"/aakaou/pipeline2_resnet18_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline3_seg_overlays",
        "csv": r"/aakaou/pipeline3_resnet18_predictions.csv"
    },

    {
        "images": r"/aakaou/pipeline4_overlays",
        "csv": r"/aakaou/pipeline4_resnet18_predictions.csv"
    }

]

# =========================
# LOAD METADATA
# =========================

meta_df = pd.read_csv(metadata_file)

class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']
class_map = {name:i for i,name in enumerate(class_names)}

meta_df['class_id'] = meta_df['dx'].map(class_map)

# ======================================================
# RESNET18 MODEL
# ======================================================

def conv_block(x, filters, kernel_size=3, stride=1):

    x_shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or x_shortcut.shape[-1] != filters:

        x_shortcut = layers.Conv2D(filters, 1,
                                   strides=stride,
                                   use_bias=False)(x_shortcut)

        x_shortcut = layers.BatchNormalization()(x_shortcut)

    x = layers.Add()([x, x_shortcut])
    x = layers.ReLU()(x)

    return x


def build_resnet18(input_shape=(224,224,3), num_classes=7):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64,7,strides=2,padding='same',use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3,strides=2,padding='same')(x)

    filters_list = [64,128,256,512]

    for filters in filters_list:
        x = conv_block(x,filters)
        x = conv_block(x,filters)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512,activation='relu')(x)
    x = layers.Dense(256,activation='relu')(x)

    outputs = layers.Dense(num_classes,activation='softmax')(x)

    model = models.Model(inputs,outputs)

    return model

# ======================================================
# RUN FOR EACH PIPELINE
# ======================================================

for pipe in pipelines_info:

    print(f"\n🚀 Training ResNet18 on {pipe['images']}")

    image_folder = Path(pipe["images"])

    # =========================
    # GET IMAGE FILES
    # =========================

    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    image_ids = [f.stem for f in image_files]

    print(f"Found {len(image_files)} images")

    # =========================
    # CREATE DATAFRAME
    # =========================

    df = meta_df[meta_df['image_id'].isin(image_ids)].copy()

    df['filename'] = df['image_id'] + ".jpg"
    df['label'] = df['class_id']

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=0
    )

    # =========================
    # DATA GENERATORS
    # =========================

    train_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        horizontal_flip=True,
        zoom_range=0.2
    )

    val_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )

    train_data = train_gen.flow_from_dataframe(
        train_df,
        directory=image_folder,
        x_col='filename',
        y_col='label',
        target_size=IMG_SIZE,
        class_mode='raw',
        batch_size=BATCH_SIZE
    )

    val_data = val_gen.flow_from_dataframe(
        val_df,
        directory=image_folder,
        x_col='filename',
        y_col='label',
        target_size=IMG_SIZE,
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # =========================
    # BUILD MODEL
    # =========================

    model = build_resnet18(
        input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
        num_classes=7
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("✅ ResNet18 model ready")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    # =========================
    # TRAIN MODEL
    # =========================

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

    probs = model.predict(val_data)
    y_pred = np.argmax(probs, axis=1)
    y_true = val_df['label'].values
    pred_conf = np.max(probs, axis=1)

    results = pd.DataFrame({

        "filename": val_df["filename"].values,

        "actual_class_id": y_true,
        "actual_class_name": [class_names[i] for i in y_true],

        "pred_class_id": y_pred,
        "pred_class_name": [class_names[i] for i in y_pred],

        "confidence": pred_conf
    })

    for i,name in enumerate(class_names):
        results[f"prob_{name}"] = probs[:,i]

    results.to_csv(pipe["csv"], index=False)

    print(f"✅ Predictions saved → {pipe['csv']}")

print("\n🎯 All pipelines completed")
