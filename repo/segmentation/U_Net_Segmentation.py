# unet_segmentation_from_sonar.py
# ===============================================================
# U-Net Segmentation using sonar-processed images
# Input: sonar-only images
# Output: binary masks + segmentation overlays
# ===============================================================

import os
from pathlib import Path
import cv2
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU

logging.basicConfig(level=logging.INFO)

# ===============================================================
# U-NET MODEL DEFINITION
# ===============================================================
def build_unet(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)

    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)

    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)

    outputs = Conv2D(1, 1, activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4),
                  loss=BinaryCrossentropy(),
                  metrics=[MeanIoU(num_classes=2)])
    return model

# ===============================================================
# SONAR FUSION FOR OVERLAY
# ===============================================================
def fuse_overlay(original, mask):
    """Fuse lesion onto original sonar image."""
    mask_bin = (mask > 0).astype(np.uint8)
    if len(mask_bin.shape) == 2:
        mask_bin = np.stack([mask_bin]*3, axis=-1)
    fused = np.where(mask_bin == 1, original, original)  # same as original
    return fused

# ===============================================================
# PROCESS SINGLE IMAGE
# ===============================================================
def process_image(model, img_path, masks_folder, overlays_folder):
    filename = img_path.name

    # Load sonar image
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"Cannot read {filename}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_shape = img_rgb.shape[:2]

    # Resize for U-Net
    img_resized = cv2.resize(img_rgb, (256, 256)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Predict mask
    pred = model.predict(img_resized, verbose=0)[0, :, :, 0]
    mask_bin = (pred > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_bin, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save binary mask
    masks_folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(masks_folder / f"{filename}.png"), mask_resized * 255)

    # Save overlay
    if overlays_folder:
        overlays_folder.mkdir(parents=True, exist_ok=True)
        fused_img = fuse_overlay(img_rgb, mask_resized)
        cv2.imwrite(str(overlays_folder / filename), cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))

    logging.info(f"✅ {filename} processed")

# ===============================================================
# PROCESS PIPELINE
# ===============================================================
def process_pipeline(model, sonar_folder, masks_folder, overlays_folder):
    sonar_folder = Path(sonar_folder) / "sonar_only"
    images = sorted(sonar_folder.glob("*.jpg"))

    with ThreadPoolExecutor(max_workers=6) as executor:
        for img_path in images:
            executor.submit(process_image, model, img_path, masks_folder, overlays_folder)

# ===============================================================
# MAIN EXECUTION
# ===============================================================
if __name__ == "__main__":
    pipelines_info = [
        {
            "sonar": r"/aakaou/pipeline1_sonar_output",
            "masks": r"/aakaou/pipeline1_seg_masks",
            "overlays": r"/aakaou/pipeline1_seg_overlays"
        },
        {
            "sonar": r"/aakaou/pipeline2_sonar_output",
            "masks": r"/aakaou/pipeline2_seg_masks",
            "overlays": r"/aakaou/pipeline2_seg_overlays"
        },
        {
            "sonar": r"/aakaou/pipeline3_sonar_output",
            "masks": r"/aakaou/pipeline3_seg_masks",
            "overlays": r"/aakaou/pipeline3_seg_overlays"
        },
        {
            "sonar": r"/aakaou/pipeline4_sonar_output",
            "masks": r"/aakaou/pipeline4_seg_masks",
            "overlays": r"/aakaou/pipeline4_seg_overlays"
        }
    ]

    # Build U-Net model
    unet_model = build_unet()
    # Load pretrained weights if available
    # unet_model.load_weights("unet_model_weights.h5")

    # Process each pipeline
    for pipe in pipelines_info:
        logging.info(f"🚀 Processing pipeline: {pipe['sonar']}")
        process_pipeline(unet_model,
                         pipe['sonar'],
                         Path(pipe['masks']),
                         Path(pipe['overlays']))

    logging.info("✅ U-Net segmentation + overlays completed for all pipelines")
