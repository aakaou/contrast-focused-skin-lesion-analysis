import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import os

"""
from datasets.load_dataset import load_dataset
from preprocessing.preprocess_pipeline4 import preprocess_pipeline4

# Step 1: Load dataset
images_path, metadata = load_dataset()

# Step 2: Apply pipeline 4
processed_path = preprocess_pipeline4(images_path)

print("Pipeline 4 processed images:", processed_path)
"""

# -------------------------------------------------
# Logging setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------
# Hair removal + CLAHE + resize + normalize
# -------------------------------------------------
def dullrazor(img):
    """Remove hair artifacts using blackhat + inpainting."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    hair_ratio = np.sum(mask) / mask.size * 100
    if hair_ratio < 1:
        return img

    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def preprocess_single_image(image_path, output_dir):
    """Process a single image and save to output_dir."""
    img = cv2.imread(str(image_path))
    if img is None:
        logging.warning(f"Skipping {image_path.name}, cannot read image")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = dullrazor(img)
    img = cv2.resize(img, (256, 256))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Save processed image
    save_path = output_dir / image_path.name
    cv2.imwrite(str(save_path), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    logging.info(f"✅ {image_path.name} processed")


# -------------------------------------------------
# Pipeline 4 preprocessing
# -------------------------------------------------
def preprocess_pipeline4(images_path, output_dir="/aakaou/pipeline4_processed_images"):
    """
    Preprocess all images from images_path (Path or string) and save to output_dir.
    Connects to load_dataset.py output.
    """
    images_path = Path(images_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_path.glob("*.jpg"))
    logging.info(f"{len(image_files)} images → Pipeline 4 preprocessing (no augmentation)")

    # Parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda p: preprocess_single_image(p, output_dir), image_files))

    logging.info(f"🎉 Pipeline 4 preprocessing complete: {output_dir}")
    return output_dir


# -----------------------------
# Example usage with load_dataset
# -----------------------------
if __name__ == "__main__":
    from datasets.load_dataset import load_dataset

    # Load images_path from your dataset loader
    images_path, metadata = load_dataset()

    # Run pipeline 4 preprocessing
    processed_dir = preprocess_pipeline4(images_path)
    print("Processed images located at:", processed_dir)
