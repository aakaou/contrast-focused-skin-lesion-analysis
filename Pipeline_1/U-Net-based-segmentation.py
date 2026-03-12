# ============================================================
# HAM10000 SONAR + U-NET SEGMENTATION PIPELINE
# ============================================================
# This script performs two main tasks:
#
# 1) Generate SONAR-style images using a JET colormap
# 2) Run U-Net segmentation using the SONAR images
# 3) Combine original image + sonar image using predicted mask
#
# Output: combined images saved in HAM10000_combined
# ============================================================


# ============================================================
# 1. Import required libraries
# ============================================================

from pathlib import Path                  # For OS-independent file paths
from PIL import Image                     # Image loading/saving
import numpy as np                        # Numerical operations
from matplotlib import cm                 # Colormap (JET)
from tqdm.auto import tqdm                # Progress bar
import tensorflow as tf                   # Deep learning framework
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    UpSampling2D, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
import os


# ============================================================
# 2. Define base directories (Kaggle working folder)
# ============================================================

BASE_DIR = Path("/aakaou")

# Folder containing normalized images
PROC_DIR = BASE_DIR / "HAM10000_processed"

# Folder where sonar images will be saved
SONAR_DIR = BASE_DIR / "HAM10000_sonar"

# Folder for final combined output
OUT_DIR = BASE_DIR / "HAM10000_segmented"

# U-Net weight file (if available)
WEIGHTS = str(BASE_DIR / "unet_sonar.h5")

# Create directories if they do not exist
SONAR_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. Image processing parameters
# ============================================================

IMG_SIZE = 256
BATCH_SIZE = 2


# ============================================================
# 4. Function: create SONAR image using matplotlib colormap
# ============================================================

def sonar_effect_matplotlib(pil_img):
    """
    Convert an RGB image into a SONAR-style visualization
    using a JET colormap.
    """

    # Convert image to grayscale
    gray = pil_img.convert("L")

    # Convert grayscale image to numpy array
    arr = np.array(gray).astype("float32")

    # Normalize pixel values to range [0,1]
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    # Apply JET colormap from matplotlib
    cmap = cm.get_cmap("jet")

    # Result is RGBA values in range [0,1]
    colored = cmap(arr_norm)

    # Keep only RGB channels
    colored_rgb = (colored[..., :3] * 255).astype("uint8")

    # Convert numpy array back to PIL image
    return Image.fromarray(colored_rgb)


# ============================================================
# 5. Generate SONAR images for all processed images
# ============================================================

print("Creating SONAR images...")

# Get all processed images
image_paths = sorted(PROC_DIR.glob("*.jpg"))

# Loop through each image
for img_path in tqdm(image_paths):

    # Load image
    orig = Image.open(img_path).convert("RGB")

    # Apply sonar effect
    sonar_img = sonar_effect_matplotlib(orig)

    # Save with prefix sonar_
    out_name = f"sonar_{img_path.name}"

    # Save image
    sonar_img.save(SONAR_DIR / out_name)

print("SONAR images created.")


# ============================================================
# 6. U-Net architecture definition
# ============================================================

def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    """
    Standard U-Net architecture for image segmentation.
    """

    # Input layer
    inputs = Input(input_size)

    # ----- Encoder -----

    c1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D(2)(c1)

    c2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D(2)(c2)

    c3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(256, 3, activation="relu", padding="same")(c3)
    p3 = MaxPooling2D(2)(c3)

    c4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    c4 = Conv2D(512, 3, activation="relu", padding="same")(c4)
    p4 = MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation="relu", padding="same")(p4)
    c5 = Conv2D(1024, 3, activation="relu", padding="same")(c5)

    # ----- Decoder -----

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

    # Output mask (binary segmentation)
    outputs = Conv2D(1, 1, activation="sigmoid")(c9)

    # Build model
    model = Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=Adam(1e-5),
        loss=BinaryCrossentropy(),
        metrics=[MeanIoU(num_classes=2)]
    )

    return model


# ============================================================
# 7. Load U-Net model
# ============================================================

model = unet_model()

# Load trained weights if available
if os.path.exists(WEIGHTS):

    print("Loading trained weights:", WEIGHTS)
    model.load_weights(WEIGHTS)

else:

    print("WARNING: No trained weights found.")
    print("Model will run with random weights.")


# ============================================================
# 8. Collect image pairs (original + sonar)
# ============================================================

orig_paths = []
sonar_paths = []

# Iterate through processed images
for orig_path in PROC_DIR.iterdir():

    if orig_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # Find corresponding sonar image
    sonar_path = SONAR_DIR / f"sonar_{orig_path.name}"

    if sonar_path.exists():
        orig_paths.append(orig_path)
        sonar_paths.append(sonar_path)

print("Total image pairs found:", len(orig_paths))


# ============================================================
# 9. Run segmentation + combine images
# ============================================================

for i in range(0, len(orig_paths), BATCH_SIZE):

    batch_orig = orig_paths[i:i + BATCH_SIZE]
    batch_sonar = sonar_paths[i:i + BATCH_SIZE]

    X = []
    ORIG = []
    NAMES = []

    # Load batch
    for op, sp in zip(batch_orig, batch_sonar):

        orig_img = Image.open(op).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        sonar_img = Image.open(sp).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        orig_arr = np.array(orig_img).astype("float32") / 255.0
        sonar_arr = np.array(sonar_img).astype("float32") / 255.0

        ORIG.append(orig_arr)
        X.append(sonar_arr)
        NAMES.append(op.name)

    X = np.stack(X)
    ORIG = np.stack(ORIG)

    # Predict segmentation mask
    preds = model.predict(X, verbose=0)[..., 0]

    # Process each prediction
    for idx, (orig_arr, pred, name) in enumerate(zip(ORIG, preds, NAMES)):

        # Binary mask
        mask = (pred > 0.5).astype("float32")

        # Convert mask to 3 channels
        mask3 = np.repeat(mask[..., None], 3, axis=-1)

        sonar_arr = X[idx]

        # Combine original lesion with sonar background
        combined = orig_arr * mask3 + sonar_arr * (1.0 - mask3)

        # Convert to uint8
        combined_uint8 = (combined * 255).clip(0, 255).astype("uint8")

        # Save output image
        out_img = Image.fromarray(combined_uint8)

        out_path = OUT_DIR / name
        out_img.save(out_path)

        print("Saved:", out_path)


# ============================================================
# Pipeline finished
# ============================================================

print("Pipeline finished successfully.")
