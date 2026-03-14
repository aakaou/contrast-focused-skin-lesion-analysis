"""
Pipeline 3 - Advanced Dermoscopic Image Preprocessing

Steps:
1. Hair removal (blackhat + inpainting)
2. Bilateral filtering
3. Wavelet transform enhancement
4. Gabor filter bank
5. Unsharp masking
6. Normalization

from datasets.load_dataset import load_dataset
from preprocessing.preprocess_pipeline3 import preprocess_pipeline3

# Step 1: Load dataset
images_path, metadata = load_dataset()

# Step 2: Apply pipeline 3
processed_path = preprocess_pipeline3(images_path)

print("Pipeline 3 processed images:", processed_path)
"""

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import pywt

logging.basicConfig(level=logging.INFO)


# -------------------------------------------------
# Core preprocessing functions
# -------------------------------------------------
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA) if np.sum(mask) > 0 else img


def bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def wavelet_enhance(img, wavelet='db1', level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    coeffs = pywt.wavedec2(img_gray, wavelet, level=level)
    coeffs_H = list(coeffs)
    for i in range(1, len(coeffs_H)):
        coeffs_H[i] = tuple([c * 1.2 for c in coeffs_H[i]])
    img_enhanced = pywt.waverec2(coeffs_H, wavelet)
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)


def gabor_filter_bank(img, frequencies=[0.1, 0.2, 0.3], orientations=[0, 45, 90, 135]):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    accum = np.zeros_like(img_gray, dtype=np.float32)
    for theta in orientations:
        theta_rad = theta * np.pi / 180
        for freq in frequencies:
            kernel = cv2.getGaborKernel((21, 21), sigma=5.0, theta=theta_rad,
                                        lambd=1/freq, gamma=0.5, psi=0)
            accum += cv2.filter2D(img_gray, cv2.CV_32F, kernel)
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(accum, cv2.COLOR_GRAY2RGB)


def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = np.clip((amount+1)*img - amount*blurred, 0, 255).astype(np.uint8)
    return sharpened


def normalize_image(img):
    norm = img.astype(np.float32) / 255.0
    mean = np.mean(norm, axis=(0,1), keepdims=True)
    std = np.std(norm, axis=(0,1), keepdims=True) + 1e-6
    norm = (norm - mean) / std
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
    return norm


# -------------------------------------------------
# Full pipeline per image
# -------------------------------------------------
def preprocess_advanced(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (256, 256))
    img = remove_hair(img)
    img = bilateral_filter(img)
    img = wavelet_enhance(img)
    img = gabor_filter_bank(img)
    img = unsharp_mask(img)
    img = normalize_image(img)
    return img


def process_single_image(img_path, output_dir):
    image = cv2.imread(str(img_path))
    if image is None:
        logging.warning(f"Skipping {img_path.name}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_advanced(image)
    out = (processed * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / img_path.name), out)
    logging.info(f"✅ {img_path.name} processed")


# -------------------------------------------------
# Preprocess entire dataset
# -------------------------------------------------
def preprocess_pipeline3(images_path, output_dir="/aakaou/pipeline3_processed_images"):
    images_path = Path(images_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_path.glob("*.jpg"))
    logging.info(f"{len(image_paths)} images → Pipeline 3 advanced preprocessing")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda p: process_single_image(p, output_dir), image_paths))

    logging.info(f"🎉 Pipeline 3 complete: {output_dir}")
    return output_dir
