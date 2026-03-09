"""This Python script implements a U-Net-based segmentation pipeline for skin lesion images in the HAM10000 dataset. 
It loads preprocessed and sonar-enhanced images, builds a U-Net model, and attempts to load pretrained weights for accurate lesion segmentation. 
The script automatically pairs original images with their sonar-enhanced counterparts, processes them in batches, and predicts segmentation masks. 
Each mask is then applied to the original image with a visual enhancement: the background blends the sonar colors, the lesion remains highlighted, 
and optional contours are drawn around the segmented region. The final images are saved to an output directory, 
producing visually informative segmentation results ready for analysis or presentation. 
The pipeline is designed for efficiency, using batch processing and careful memory management."""

import os  # Operating system functions (paths, file checks)
from pathlib import Path  # Modern path handling
from PIL import Image  # Image loading, resizing, and saving
import numpy as np  # Numerical operations
import cv2  # OpenCV for image processing

import tensorflow as tf  # TensorFlow library
from tensorflow.keras.layers import (  # Import layers for U-Net
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, concatenate
)
from tensorflow.keras.models import Model  # For building the model
from tensorflow.keras.optimizers import Adam  # Optimizer
from tensorflow.keras.losses import BinaryCrossentropy  # Loss for segmentation
from tensorflow.keras.metrics import MeanIoU  # Metric for segmentation quality

# -----------------------------
# PATHS & CONSTANTS
# -----------------------------
IMG_SIZE   = 256  # Resize images to 256x256
BATCH_SIZE = 8    # Number of images processed per batch

BASE_DIR  = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2")  # Base folder
PROC_DIR  = BASE_DIR / "HAM10000_processed_images"  # Preprocessed images
SONAR_DIR = BASE_DIR / "HAM10000_sonar_images"      # Sonar-effect images
OUT_DIR   = BASE_DIR / "HAM10000_segmented_images"  # Segmentation results

WEIGHTS_DEFAULT = BASE_DIR / "unet_sonar.h5"  # Default weights file

# Create output directory if missing
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# U-NET MODEL
# -----------------------------
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(input_size)  # Input layer (256x256 RGB)

    # --- Encoder ---
    c1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D(2)(c1)  # Downsample

    c2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(256, 3, activation="relu", padding="same")(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    c4 = Conv2D(512, 3, activation="relu", padding="same")(c4)
    p4 = MaxPooling2D(2)(c4)

    # --- Bottleneck ---
    c5 = Conv2D(1024, 3, activation="relu", padding="same")(p4)
    c5 = Conv2D(1024, 3, activation="relu", padding="same")(c5)

    # --- Decoder ---
    u6 = UpSampling2D(2)(c5)
    c6 = concatenate([u6, c4])
    c6 = Conv2D(512, 3, activation="relu", padding="same")(c6)
    c6 = Conv2D(512, 3, activation="relu", padding="same")(c6)

    u7 = UpSampling2D(2)(c6)
    c7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation="relu", padding="same")(c7)
    c7 = Conv2D(256, 3, activation="relu", padding="same")(c7)

    u8 = UpSampling2D(2)(c7)
    c8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation="relu", padding="same")(c8)
    c8 = Conv2D(128, 3, activation="relu", padding="same")(c8)

    u9 = UpSampling2D(2)(c8)
    c9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation="relu", padding="same")(c9)
    c9 = Conv2D(64, 3, activation="relu", padding="same")(c9)

    outputs = Conv2D(1, 1, activation="sigmoid")(c9)  # Output mask

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-5),
        loss=BinaryCrossentropy(),
        metrics=[MeanIoU(num_classes=2)],  # Evaluate overlap
    )
    return model

# -----------------------------
# AUTO-FIND WEIGHTS
# -----------------------------
def find_weight_file():
    if WEIGHTS_DEFAULT.exists():  # Check default location
        return WEIGHTS_DEFAULT

    # If not, search common Kaggle paths
    search_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    patterns = [
        "unet_sonar*.h5",
        "unet_sonar*.weights.h5",
        "*.weights.h5",
        "unet_sonar*.keras",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = list(root.rglob(pat))
            if hits:
                return hits[0]
    return None

# -----------------------------
# LOAD MODEL + WEIGHTS
# -----------------------------
model = unet_model()  # Build model

weight_path = find_weight_file()
if weight_path is None:
    print("⚠️ No U-Net weights found. Using untrained model.")
else:
    print(f"✅ Loading weights: {weight_path}")
    model.load_weights(str(weight_path))  # Load pretrained weights

# -----------------------------
# VALIDATE DIRECTORIES
# -----------------------------
if not PROC_DIR.exists():
    raise FileNotFoundError(f"Missing: {PROC_DIR}")
if not SONAR_DIR.exists():
    raise FileNotFoundError(f"Missing: {SONAR_DIR}")

# -----------------------------
# COLLECT IMAGE PAIRS
# -----------------------------
orig_paths, sonar_paths = [], []
for op in PROC_DIR.iterdir():  # Loop through preprocessed images
    if op.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        sp = SONAR_DIR / op.name
        if sp.exists():  # Ensure matching sonar image exists
            orig_paths.append(op)
            sonar_paths.append(sp)

print(f"Found {len(orig_paths)} image pairs")

# -----------------------------
# VISUAL STYLE
# -----------------------------
ALPHA_BG = 0.80      # Blend factor for background
DRAW_CONTOUR = True  # Whether to draw lesion contours

# -----------------------------
# SEGMENT & SAVE FINAL OUTPUT
# -----------------------------
total = len(orig_paths)

for i in range(0, total, BATCH_SIZE):
    batch_orig  = orig_paths[i:i + BATCH_SIZE]  # Original images
    batch_sonar = sonar_paths[i:i + BATCH_SIZE]  # Sonar images
    names = [p.name for p in batch_orig]  # Filenames

    X, ORIG = [], []

    for op, sp in zip(batch_orig, batch_sonar):
        orig = Image.open(op).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        sonar = Image.open(sp).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        ORIG.append(np.asarray(orig, dtype=np.float32) / 255.0)  # Normalize
        X.append(np.asarray(sonar, dtype=np.float32) / 255.0)

    X = np.stack(X, axis=0)    # Batch tensor for sonar
    ORIG = np.stack(ORIG, axis=0)  # Batch tensor for original

    preds = model.predict(X, verbose=0)[..., 0]  # Predict masks

    for orig_arr, sonar_arr, pred, name in zip(ORIG, X, preds, names):
        mask = (pred > 0.5).astype(np.uint8) * 255  # Binary mask
        bg_mask = cv2.bitwise_not(mask)             # Invert mask for background

        orig_u8  = (orig_arr * 255).astype(np.uint8)
        sonar_u8 = (sonar_arr * 255).astype(np.uint8)

        # Highlight red channel for sonar background
        sonar_red = np.zeros_like(sonar_u8)
        sonar_red[:, :, 0] = np.clip(sonar_u8[:, :, 0] * 2.2, 0, 255)
        sonar_red[:, :, 1] = np.clip(sonar_u8[:, :, 1] * 0.3, 0, 255)
        sonar_red[:, :, 2] = np.clip(sonar_u8[:, :, 2] * 0.3, 0, 255)

        # Blend sonar background with original image
        bg_blend = cv2.addWeighted(
            orig_u8, 1.0 - ALPHA_BG,
            sonar_red, ALPHA_BG, 0
        )

        lesion_part = cv2.bitwise_and(orig_u8, orig_u8, mask=mask)  # Lesion
        bg_part     = cv2.bitwise_and(bg_blend, bg_blend, mask=bg_mask)  # Background
        combined = cv2.add(lesion_part, bg_part)  # Combine lesion + background

        # Draw contour if enabled
        if DRAW_CONTOUR:
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(combined, cnts, -1, (255, 255, 0), 2)

        Image.fromarray(combined).save(OUT_DIR / name)  # Save final image

    print(f"Processed {min(i + BATCH_SIZE, total)}/{total}")

print(f"🎉 Done. Results saved in: {OUT_DIR}")
