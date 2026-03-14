"""
Pipeline 2 - Advanced Preprocessing for HAM10000

Steps:
1. Resize images
2. Hair removal (Blackhat + inpainting)
3. White balance
4. CLAHE contrast enhancement
5. Normalize pixel values


from datasets.load_dataset import load_dataset
from preprocessing.preprocess_pipeline2 import preprocess_pipeline2

# Step 1: Load dataset
images_path, metadata = load_dataset()

# Step 2: Preprocess images (Pipeline 2)
processed_path = preprocess_pipeline2(images_path)

print("Pipeline 2 processed images:", processed_path)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


# --------------------------------------------------
# IMAGE PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess_final_image(image):
    """
    Apply full preprocessing to a single image
    """

    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize
    image = cv2.resize(image, (256, 256))

    # --------------------------------
    # Hair removal
    # --------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    if np.sum(mask) > 0:
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

    # --------------------------------
    # White balance
    # --------------------------------
    img = image.astype(np.float32)

    avg = np.mean(img, axis=(0, 1))
    gray_avg = np.mean(avg)

    scale = gray_avg / (avg + 1e-6)
    img *= scale

    img = np.clip(img, 0, 255).astype(np.uint8)

    # --------------------------------
    # CLAHE contrast enhancement
    # --------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    L = clahe.apply(L)

    lab = cv2.merge((L, A, B))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # --------------------------------
    # Normalization
    # --------------------------------
    img = img.astype(np.float32) / 255.0
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return img


# --------------------------------------------------
# PROCESS SINGLE IMAGE
# --------------------------------------------------
def process_single_image(img_path, output_dir):

    image = cv2.imread(str(img_path))

    if image is None:
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processed = preprocess_final_image(image)

    # Convert to uint8 for saving
    out = (processed * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_dir / img_path.name), out)


# --------------------------------------------------
# MAIN PREPROCESS FUNCTION
# --------------------------------------------------
def preprocess_pipeline2(images_path,
                         output_dir="/aakaou/pipeline2_processed_images"):
    """
    Preprocess all images from dataset loader
    """

    images_path = Path(images_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_path.glob("*.jpg"))

    print(f"Found {len(image_paths)} images")
    print("Starting Pipeline 2 preprocessing...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(lambda p: process_single_image(p, output_dir),
                             image_paths),
                total=len(image_paths)
            )
        )

    print("✅ Pipeline 2 preprocessing complete")

    return output_dir
