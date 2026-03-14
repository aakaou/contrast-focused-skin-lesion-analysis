# ======================================================
# vgg19_pretrained_all_pipelines.py
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (224, 224)
metadata_file = "/aakaou/HAM10000_metadata.csv"

# =========================
# PIPELINES
# =========================

pipelines_info = [
    {
        "pred_masks": r"/aakaou/pipeline1_seg_masks",
        "gt_masks": r"/aakaou/pipeline1_seg_overlays",
        "csv": r"/aakaou/pipeline1_vgg19_predictions.csv"
    },
    {
        "pred_masks": r"/aakaou/pipeline2_seg_masks",
        "gt_masks": r"/aakaou/pipeline2_seg_overlays",
        "csv": r"/aakaou/pipeline2_vgg19_predictions.csv"
    },
    {
        "pred_masks": r"/aakaou/pipeline3_seg_masks",
        "gt_masks": r"/aakaou/pipeline3_seg_overlays",
        "csv": r"/aakaou/pipeline3_vgg19_predictions.csv"
    },
    {
        "pred_masks": r"/aakaou/pipeline4_seg_masks",
        "gt_masks": r"/aakaou/pipeline4_overlays",
        "csv": r"/aakaou/pipeline4_vgg19_predictions.csv"
    }
]

# =========================
# LOAD METADATA
# =========================

meta_df = pd.read_csv(metadata_file)

class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']
class_map = {name:i for i,name in enumerate(class_names)}

meta_df["class_id"] = meta_df["dx"].map(class_map)

# =========================
# BUILD VGG19 MODEL
# =========================

base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

print("✅ Pretrained VGG19 model ready")

# ======================================================
# PROCESS EACH PIPELINE
# ======================================================

for pipe in pipelines_info:

    print(f"\n🚀 Running VGG19 on pipeline → {pipe['csv']}")

    image_folder = Path(pipe["gt_masks"])
    image_files = list(image_folder.glob("*.png")) + list(image_folder.glob("*.jpg"))

    results = []

    for img_path in image_files:

        img_id = img_path.stem

        # Load image
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        probs = model.predict(img, verbose=0)[0]

        pred_id = np.argmax(probs)
        confidence = np.max(probs)

        # Get ground truth
        row = meta_df[meta_df["image_id"] == img_id]
        if len(row) == 0:
            continue

        true_id = int(row["class_id"].values[0])

        result = {
            "image_id": img_id,
            "actual_class_id": true_id,
            "actual_class_name": class_names[true_id],
            "pred_class_id": pred_id,
            "pred_class_name": class_names[pred_id],
            "confidence": confidence
        }

        for i, name in enumerate(class_names):
            result[f"prob_{name}"] = probs[i]

        results.append(result)

    df = pd.DataFrame(results)

    df.to_csv(pipe["csv"], index=False)

    print(f"✅ Predictions saved → {pipe['csv']}")

print("\n🎯 All pipelines completed")
