# ================================================================
# SONAR + U-NET SEGMENTATION PIPELINE FOR HAM10000
# ================================================================
# This script performs the following steps:
#
# 1. Convert processed dermoscopy images to RED-dominant SONAR images
# 2. Load U-Net segmentation model (if weights available)
# 3. Segment lesion from sonar images
# 4. Combine:
#       lesion  → original processed image
#       background → red sonar image
# 5. Save final segmented images
#
# ================================================================


# ================================
# IMPORT LIBRARIES
# ================================

import os                          # file operations
import cv2                         # computer vision operations
import numpy as np                 # numerical operations
from pathlib import Path           # path handling
import logging                     # logging system
from concurrent.futures import ThreadPoolExecutor  # parallel processing
from PIL import Image              # image loading/saving
import tensorflow as tf            # deep learning framework


# ================================
# DEFINE DIRECTORY PATHS
# ================================

# Folder containing processed dermoscopy images
PROCESSED_DIR = Path("/aakaou/HAM10000_processed_images")

# Folder where sonar images will be saved
SONAR_DIR = Path("/aakaou/HAM10000_sonar_red_images")

# Folder where final segmented images will be saved
OUT_DIR = Path("/aakaou/sonar_segmented_images")

# Create output folder if it does not exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to trained U-Net weights
MODEL_PATH = Path("/aakaou/unet_sonar_weights.h5")


# ================================
# CONFIGURATION PARAMETERS
# ================================

IMG_SIZE = 256         # input size for model
BATCH_SIZE = 4         # batch size for inference
ALPHA_BG = 0.75        # blending strength (unused but adjustable)
DRAW_CONTOUR = True    # draw lesion contour in final output


# ================================
# LOGGING CONFIGURATION
# ================================

logging.basicConfig(level=logging.INFO)


# ===============================================================
# RED DOMINANT SONAR FUNCTION
# ===============================================================

def red_sonar(image, red_boost=2.0, green_blue_down=0.35):

    """
    Convert image to red-dominant sonar style.

    Dark structures become bright red.
    Green/blue channels are suppressed.
    """

    # Convert input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize grayscale values between 0-255
    norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply OpenCV JET colormap
    sonar = cv2.applyColorMap(norm_gray.astype(np.uint8), cv2.COLORMAP_JET)

    # Convert BGR → RGB
    sonar = cv2.cvtColor(sonar, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Boost RED channel intensity
    sonar[:, :, 0] = np.clip(sonar[:, :, 0] * red_boost, 0, 255)

    # Reduce GREEN channel
    sonar[:, :, 1] = np.clip(sonar[:, :, 1] * green_blue_down, 0, 255)

    # Reduce BLUE channel
    sonar[:, :, 2] = np.clip(sonar[:, :, 2] * green_blue_down, 0, 255)

    # Convert back to uint8
    return sonar.astype(np.uint8)


# ===============================================================
# PROCESS SINGLE IMAGE → CREATE SONAR IMAGE
# ===============================================================

def process_single_sonar(filename):

    # Construct full image path
    img_path = PROCESSED_DIR / filename

    # Skip invalid files
    if not img_path.exists() or not filename.lower().endswith(('.jpg','.png','.jpeg')):
        return

    try:

        # Load image using OpenCV
        image = cv2.imread(str(img_path))

        # Generate red sonar version
        sonar_img = red_sonar(image)

        # Create sonar folder if needed
        SONAR_DIR.mkdir(parents=True, exist_ok=True)

        # Save sonar image
        save_path = SONAR_DIR / filename

        cv2.imwrite(
            str(save_path),
            cv2.cvtColor(sonar_img, cv2.COLOR_RGB2BGR)
        )

        logging.info(f"✅ Saved sonar: {filename}")

    except Exception as e:

        logging.error(f"❌ {filename}: {e}")


# ===============================================================
# CREATE SONAR IMAGES FOR ALL DATASET
# ===============================================================

def process_all_images():

    # Ensure processed images folder exists
    if not PROCESSED_DIR.exists():
        logging.error("❌ Processed images folder not found!")
        return

    # Collect all image filenames
    files = [
        f.name for f in PROCESSED_DIR.iterdir()
        if f.suffix.lower() in {'.jpg','.png','.jpeg'}
    ]

    logging.info(f"Processing {len(files)} images for red-dominant sonar...")

    # Run multi-threaded processing
    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(process_single_sonar, f)
            for f in files
        ]

        # Wait for all threads
        for future in futures:
            future.result()

    logging.info(f"🎉 All sonar images saved to: {SONAR_DIR}")


# ===============================================================
# U-NET MODEL ARCHITECTURE
# ===============================================================

def build_unet(input_shape=(256,256,3)):

    # Define input layer
    inputs = tf.keras.Input(input_shape)

    # ----- Encoder -----

    c1 = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(inputs)
    c1 = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(128,3,activation="relu",padding="same")(p1)
    c2 = tf.keras.layers.Conv2D(128,3,activation="relu",padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(256,3,activation="relu",padding="same")(p2)
    c3 = tf.keras.layers.Conv2D(256,3,activation="relu",padding="same")(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # ----- Bottleneck -----

    b = tf.keras.layers.Conv2D(512,3,activation="relu",padding="same")(p3)
    b = tf.keras.layers.Conv2D(512,3,activation="relu",padding="same")(b)

    # ----- Decoder -----

    u3 = tf.keras.layers.UpSampling2D()(b)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = tf.keras.layers.Conv2D(256,3,activation="relu",padding="same")(u3)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = tf.keras.layers.Conv2D(128,3,activation="relu",padding="same")(u2)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(u1)

    # Output segmentation mask
    outputs = tf.keras.layers.Conv2D(1,1,activation="sigmoid")(c6)

    return tf.keras.Model(inputs, outputs)


# ===============================================================
# LOAD MODEL
# ===============================================================

model = build_unet()

# If weights exist → load trained model
if MODEL_PATH.exists():

    model.load_weights(MODEL_PATH)

    print("✅ Loaded trained U-Net weights")

    USE_UNET = True

# Otherwise use simple threshold segmentation
else:

    print("⚠️ No U-Net weights found → DEMO MODE")

    USE_UNET = False


# ===============================================================
# COLLECT IMAGE PATHS
# ===============================================================

orig_paths  = sorted(PROCESSED_DIR.glob("*.jpg"))
sonar_paths = sorted(SONAR_DIR.glob("*.jpg"))

# Ensure both folders have same number of images
assert len(orig_paths) == len(sonar_paths), "Mismatch between processed and sonar images"


# ===============================================================
# SEGMENTATION + IMAGE COMPOSITION
# ===============================================================

for i in range(0, len(orig_paths), BATCH_SIZE):

    batch_orig  = orig_paths[i:i+BATCH_SIZE]
    batch_sonar = sonar_paths[i:i+BATCH_SIZE]

    X, ORIG, names = [], [], []

    # Load images
    for op, sp in zip(batch_orig, batch_sonar):

        orig  = Image.open(op).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        sonar = Image.open(sp).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        ORIG.append(np.array(orig, dtype=np.float32) / 255.0)
        X.append(np.array(sonar, dtype=np.float32) / 255.0)

        names.append(op.name)

    X = np.stack(X)
    ORIG = np.stack(ORIG)


    # ===========================================================
    # SEGMENTATION
    # ===========================================================

    if USE_UNET:

        preds = model.predict(X, verbose=0)[...,0]

        masks = (preds > 0.5).astype(np.uint8) * 255

    else:

        masks = []

        for sonar in X:

            gray = cv2.cvtColor((sonar*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

            _, m = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

            masks.append(m)

        masks = np.array(masks)


    # ===========================================================
    # COMBINE LESION + RED SONAR BACKGROUND
    # ===========================================================

    for orig_arr, sonar_arr, mask, name in zip(ORIG, X, masks, names):

        mask_u8 = mask.astype(np.uint8)

        bg_mask  = cv2.bitwise_not(mask_u8)

        orig_u8  = (orig_arr * 255).astype(np.uint8)

        sonar_u8 = (sonar_arr * 255).astype(np.uint8)


        # Create stronger RED sonar background
        sonar_red = np.zeros_like(sonar_u8)

        sonar_red[:,:,0] = np.clip(sonar_u8[:,:,0] * 2.2, 0, 255)

        sonar_red[:,:,1] = np.clip(sonar_u8[:,:,1] * 0.3, 0, 255)

        sonar_red[:,:,2] = np.clip(sonar_u8[:,:,2] * 0.3, 0, 255)


        # Extract lesion area
        lesion_part = cv2.bitwise_and(orig_u8, orig_u8, mask=mask_u8)

        # Extract sonar background
        bg_part = cv2.bitwise_and(sonar_red, sonar_red, mask=bg_mask)

        # Merge both
        final_img = cv2.add(lesion_part, bg_part)


        # Draw contour around lesion
        if DRAW_CONTOUR:

            cnts,_ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(final_img, cnts, -1, (255,255,0), 2)


        # Save result
        Image.fromarray(final_img).save(OUT_DIR / name)


print("🎉 DONE → Saved segmented images to:", OUT_DIR)
