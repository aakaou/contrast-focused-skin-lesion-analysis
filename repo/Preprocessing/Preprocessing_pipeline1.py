"""
HAM10000 Preprocessing Pipeline
Uses images_path returned from load_dataset.py

from datasets.load_dataset import load_dataset
from preprocessing.preprocess_images import preprocess_images

# Step 1: Load dataset
images_path, metadata = load_dataset()

# Step 2: Preprocess images
processed_path = preprocess_images(images_path)

print("Processed images located at:", processed_path)
"""

from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from tqdm.auto import tqdm


def preprocess_images(images_path, img_size=256):
    """
    Preprocess HAM10000 images.

    Parameters
    ----------
    images_path : Path
        Path returned by load_dataset.py (merged images folder)

    img_size : int
        Target image size (default = 256)

    Returns
    -------
    processed_path : Path
        Folder containing processed images
    """

    # Convert to Path object
    images_path = Path(images_path)

    # Output directory
    processed_path = Path("/aakaou/pipeline1_processed_images")
    processed_path.mkdir(parents=True, exist_ok=True)

    print("📂 Input images folder:", images_path)
    print("📂 Output folder:", processed_path)

    # Collect images
    image_paths = sorted(images_path.glob("*.jpg"))

    print("Found images:", len(image_paths))
    print("Starting preprocessing...")

    for img_path in tqdm(image_paths):

        # Load image
        img = Image.open(img_path)

        # Fix orientation
        img = ImageOps.exif_transpose(img)

        # Convert to RGB
        img = img.convert("RGB")

        # Resize
        img = img.resize((img_size, img_size), Image.BILINEAR)

        # Convert to numpy
        arr = np.array(img).astype("float32") / 255.0

        # Compute mean & std
        mean = arr.mean(axis=(0, 1), keepdims=True)
        std = arr.std(axis=(0, 1), keepdims=True) + 1e-6

        # Standardization
        arr_norm = (arr - mean) / std

        # Rescale to [0,1]
        arr_norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min() + 1e-6)

        # Convert back to uint8
        arr_uint8 = (arr_norm * 255).clip(0, 255).astype("uint8")

        img_out = Image.fromarray(arr_uint8)

        # Save image
        img_out.save(processed_path / img_path.name)

    print("✅ Preprocessing complete!")
    print("📂 Processed images saved in:", processed_path)

    return processed_path
