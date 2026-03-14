# sonar_effect_all_pipelines.py
# ===============================================================
# Apply sonar effect + fusion for multiple preprocessing pipelines
# ===============================================================

import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)


# --------------------------------------------------
# SONAR + FUSION FUNCTIONS
# --------------------------------------------------
def apply_sonar(image):
    """Convert RGB image to sonar-style colormap."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)


def fuse_lesion_background(original, sonar, mask):
    """Fuse lesion (mask=1) from original image onto sonar background."""
    fused = sonar.copy()
    mask_bin = (mask > 0).astype(np.uint8)
    if len(mask_bin.shape) == 2:
        mask_bin = np.stack([mask_bin]*3, axis=-1)
    fused = np.where(mask_bin == 1, original, fused)
    return fused


# --------------------------------------------------
# PROCESS SINGLE IMAGE
# --------------------------------------------------
def process_image_file(img_path, mask_path, output_folder):
    """Apply sonar + fusion for a single image."""
    filename = img_path.name

    # Load original image
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"Cannot read {filename}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load mask
    mask_file = mask_path / f"{img_path.stem}.png"
    if not mask_file.exists():
        logging.warning(f"No mask found for {filename}")
        return
    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)

    # Apply sonar
    sonar_img = apply_sonar(img)

    # Fuse lesion on sonar
    fused_img = fuse_lesion_background(img, sonar_img, mask)

    # Save results
    sonar_folder = output_folder / "sonar_only"
    fused_folder = output_folder / "fused"
    sonar_folder.mkdir(parents=True, exist_ok=True)
    fused_folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sonar_folder / f"{filename}"), cv2.cvtColor(sonar_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(fused_folder / f"{filename}"), cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))

    logging.info(f"✅ {filename} processed")


# --------------------------------------------------
# PROCESS ALL IMAGES IN PIPELINE
# --------------------------------------------------
def process_pipeline(preprocessed_folder, masks_folder, output_folder):
    """Process all images in a preprocessed pipeline folder."""
    preprocessed_folder = Path(preprocessed_folder)
    masks_folder = Path(masks_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    images = sorted(preprocessed_folder.glob("*.jpg"))
    logging.info(f"{len(images)} images found in {preprocessed_folder}")

    # Parallel processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        for img in images:
            executor.submit(process_image_file, img, masks_folder, output_folder)

    logging.info(f"🎉 Sonar + fusion complete for {preprocessed_folder}")


# --------------------------------------------------
# PROCESS MULTIPLE PIPELINES
# --------------------------------------------------
def process_all_pipelines(pipelines_info):
    """
    pipelines_info: list of dicts
    each dict = {
        "preprocessed": <folder with preprocessed images>,
        "masks": <folder with segmentation masks>,
        "output": <folder to save sonar + fused images>
    }
    """
    for pipeline in pipelines_info:
        logging.info(f"Processing pipeline: {pipeline['preprocessed']}")
        process_pipeline(pipeline['preprocessed'], pipeline['masks'], pipeline['output'])


# ===============================================================
# EXAMPLE USAGE
# ===============================================================
if __name__ == "__main__":
    pipelines_info = [
        {
            "preprocessed": r"/aakaou/pipeline1_processed_images",
            "masks": r"/aakaou/pipeline1_seg_masks",
            "output": r"/aakaou/pipeline1_sonar_output"
        },
        {
            "preprocessed": r"/aakaou/pipeline2_processed_images",
            "masks": r"/aakaou/pipeline2_seg_masks",
            "output": r"/aakaou/pipeline2_sonar_output"
        },
        {
            "preprocessed": r"/aakaou/pipeline3_processed_images",
            "masks": r"/aakaou/pipeline3_seg_masks",
            "output": r"/aakaou/pipeline3_sonar_output"
        },
        {
            "preprocessed": r"/aakaou/pipeline4_processed_images",
            "masks": r"/aakaou/pipeline4_seg_masks",
            "output": r"/aakaou/pipeline4_sonar_output"
        }
    ]

    process_all_pipelines(pipelines_info)
