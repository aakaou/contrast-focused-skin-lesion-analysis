
# ============================================================
# HAM10000 IMAGE MERGING + PREPROCESSING PIPELINE
# No augmentation (only orientation fix, resize, normalization)
# ============================================================

# ----------- 1. Import required libraries -----------

from pathlib import Path          # Platform-independent file path handling
import shutil                     # Copy files from one folder to another
from PIL import Image, ImageOps   # Image loading and orientation correction
import numpy as np                # Numerical operations for normalization
from tqdm.auto import tqdm        # Progress bar for loops


# ============================================================
# 2. Merge HAM10000 image folders (part_1 + part_2)
# ============================================================

# Root dataset directory
root = Path("/aakaou/ham10000-dataset")

# Two original folders containing images
part1 = root / "HAM10000_images_part_1"
part2 = root / "HAM10000_images_part_2"

# Destination folder that will contain ALL images
dest = Path("/aakaou/HAM10000_images_all")

# Create the folder if it does not exist
dest.mkdir(exist_ok=True)

# Loop through both folders
for sub in [part1, part2]:

    # Iterate through all jpg images
    for img_path in sub.glob("*.jpg"):

        # Copy the image into the new combined folder
        shutil.copy2(img_path, dest / img_path.name)

# Print confirmation
print("✅ Done! All images are now in:", dest)


# ============================================================
# 3. Preprocessing settings
# ============================================================

# Folder containing merged images
RAW_DIR = Path("/aakaou/HAM10000_images_all")

# Folder that will contain processed images
OUT_DIR = Path("/aakaou/HAM10000_processed")

# Create output directory if it does not exist
OUT_DIR.mkdir(exist_ok=True)

# Target size for all images
IMG_SIZE = 256


# ============================================================
# 4. Collect all images
# ============================================================

# Get all JPG images and sort them
image_paths = sorted(RAW_DIR.glob("*.jpg"))

# Print number of images found
print("Found images:", len(image_paths))


# ============================================================
# 5. Process each image
# ============================================================

# Loop through images with progress bar
for img_path in tqdm(image_paths):

    # Load image using PIL
    img = Image.open(img_path)

    # Fix orientation using EXIF metadata
    # Some HAM10000 images are rotated depending on camera
    img = ImageOps.exif_transpose(img)

    # Convert image to RGB (remove possible grayscale)
    img = img.convert("RGB")

    # Resize image to fixed size (256x256)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Convert image to numpy array
    arr = np.array(img)

    # Convert to float and normalize pixel range [0,1]
    arr = arr.astype("float32") / 255.0

    # Compute mean per channel
    mean = arr.mean(axis=(0, 1), keepdims=True)

    # Compute standard deviation per channel
    std = arr.std(axis=(0, 1), keepdims=True) + 1e-6

    # Standardization (z-score normalization)
    arr_norm = (arr - mean) / std

    # Rescale values back to range [0,1]
    arr_norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min() + 1e-6)

    # Convert normalized image back to uint8 [0,255]
    arr_uint8 = (arr_norm * 255).clip(0, 255).astype("uint8")

    # Convert numpy array back to PIL image
    img_out = Image.fromarray(arr_uint8)

    # Save processed image using the same filename
    img_out.save(OUT_DIR / img_path.name)


# ============================================================
# 6. Finish message
# ============================================================

print("✅ Preprocessing complete!")
print("📂 Processed images saved in:", OUT_DIR)
