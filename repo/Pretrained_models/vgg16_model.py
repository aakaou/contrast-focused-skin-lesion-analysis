# ======================================================
# vgg16_pretrained_all_pipelines.py
# ======================================================

import pandas as pd
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =========================
# PARAMETERS
# =========================

IMG_SIZE = (224,224)

metadata_file = "/aakaou/HAM10000_metadata.csv"

# =========================
# PIPELINES
# =========================

pipelines_info = [

    {
        "pred_masks": r"/aakaou/pipeline1_seg_masks",
        "gt_masks": r"/aakaou/pipeline1_seg_overlays",
        "csv": r"/aakaou/pipeline1_vgg16_predictions.csv"
    },

    {
        "pred_masks": r"/aakaou/pipeline2_seg_masks",
        "gt_masks": r"/aakaou/pipeline2_seg_overlays",
        "csv": r"/aakaou/pipeline2_vgg16_predictions.csv"
    },

    {
        "pred_masks": r"/aakaou/pipeline3_seg_masks",
        "gt_masks": r"/aakaou/pipeline3_seg_overlays",
        "csv": r"/aakaou/pipeline3_vgg16_predictions.csv"
    },

    {
        "pred_masks": r"/aakaou/pipeline4_seg_masks",
        "gt_masks": r"/aakaou/pipeline4_overlays",
        "csv": r"/aakaou/pipeline4_vgg16_predictions.csv"
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
# VGG16 MODEL
# =========================

base_model = VGG16(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512,activation="relu")(x)
x = Dense(256,activation="relu")(x)
outputs = Dense(7,activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

print("✅ Pretrained VGG16 ready")

# ======================================================
# PROCESS EACH PIPELINE
# ======================================================

for pipe in pipelines_info:

    print(f"\n🚀 Running VGG16 on pipeline: {pipe['csv']}")

    overlay_folder = Path(pipe["gt_masks"])

    image_files = list(overlay_folder.glob("*.png")) + list(overlay_folder.glob("*.jpg"))

    results = []

    for img_path in image_files:

        img_id = img_path.stem

        # Load image
        img = load_img(img_path, target_size=IMG_SIZE)
        img = img_to_array(img)
        img = preprocess_input(img)

        img = np.expand_dims(img, axis=0)

        # Prediction
        probs = model.predict(img, verbose=0)[0]

        pred_id = np.argmax(probs)
        confidence = np.max(probs)

        # Ground truth
        row = meta_df[meta_df["image_id"] == img_id]

        if len(row) == 0:
            continue

        true_id = int(row["class_id"].values[0])

        results.append({

            "image_id": img_id,

            "actual_class_id": true_id,
            "actual_class_name": class_names[true_id],

            "pred_class_id": pred_id,
            "pred_class_name": class_names[pred_id],

            "confidence": confidence
        })

        # Save class probabilities
        for i,name in enumerate(class_names):
            results[-1][f"prob_{name}"] = probs[i]

    df = pd.DataFrame(results)

    df.to_csv(pipe["csv"], index=False)

    print(f"✅ Saved predictions: {pipe['csv']}")

print("\n🎯 All pipelines completed")
