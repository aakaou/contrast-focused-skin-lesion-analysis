# ===============================================================
# ADVANCED SKIN LESION IMAGE PREPROCESSING PIPELINE
# ===============================================================
# This script performs advanced preprocessing on dermoscopic
# images from the HAM10000 dataset.
#
# Pipeline steps:
# 1. Hair removal using morphological blackhat + inpainting
# 2. Bilateral filtering (edge-preserving smoothing)
# 3. Wavelet transform enhancement
# 4. Gabor filter bank (texture enhancement)
# 5. Unsharp masking (image sharpening)
# 6. Normalization
#
# The processed images are saved to a new directory.
# ===============================================================


# ---------------------------------------------------------------
# Import required libraries
# ---------------------------------------------------------------

import os                          # For file and directory operations
import cv2                         # OpenCV library for image processing
import numpy as np                 # Numerical computations
from pathlib import Path           # Platform-independent file paths
from concurrent.futures import ThreadPoolExecutor  # Parallel processing
import logging                     # Logging messages during execution
import pywt                        # Wavelet transform library


# ---------------------------------------------------------------
# Define input and output directories
# ---------------------------------------------------------------

# Directory containing the original HAM10000 images
DATA_DIR = Path("/aakaou/HAM10000_images_all")

# Directory where processed images will be saved
OUTPUT_DIR = Path("/aakaou/HAM10000_processed_images")


# ---------------------------------------------------------------
# Configure logging to show information messages
# ---------------------------------------------------------------

logging.basicConfig(level=logging.INFO)


# ===============================================================
# 1. HAIR REMOVAL
# ===============================================================

def remove_hair(img):
    """
    Remove hair artifacts from dermoscopic images using
    morphological blackhat filtering and image inpainting.
    """

    # Convert RGB image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Create elliptical kernel for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

    # Apply blackhat operation to highlight dark hair structures
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold to create binary mask of hair pixels
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # If hair is detected, use inpainting to remove it
    hair_removed = cv2.inpaint(img, hair_mask, 5, cv2.INPAINT_TELEA) \
        if np.sum(hair_mask) > 0 else img

    return hair_removed


# ===============================================================
# 2. BILATERAL FILTER
# ===============================================================

def bilateral_filter(img, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to smooth the image while
    preserving edges.
    """

    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


# ===============================================================
# 3. WAVELET TRANSFORM ENHANCEMENT
# ===============================================================

def wavelet_enhance(img, wavelet='db1', level=1):
    """
    Enhance image details using discrete wavelet transform.
    """

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(img_gray, wavelet, level=level)

    # Convert coefficients to a mutable list
    coeffs_H = list(coeffs)

    # Amplify high-frequency components to enhance details
    for i in range(1, len(coeffs_H)):
        coeffs_H[i] = tuple([c * 1.2 for c in coeffs_H[i]])

    # Reconstruct enhanced image from modified coefficients
    img_enhanced = pywt.waverec2(coeffs_H, wavelet)

    # Clip values to valid pixel range
    img_enhanced = np.clip(img_enhanced, 0, 255).astype(np.uint8)

    # Convert grayscale result back to RGB
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)

    return img_rgb


# ===============================================================
# 4. GABOR FILTER BANK
# ===============================================================

def gabor_filter_bank(img, frequencies=[0.1, 0.2, 0.3], orientations=[0, 45, 90, 135]):
    """
    Apply a bank of Gabor filters to enhance texture patterns.
    """

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Create accumulator for filtered outputs
    accum = np.zeros_like(img_gray, dtype=np.float32)

    # Loop over different orientations
    for theta in orientations:

        # Convert orientation angle to radians
        theta_rad = theta * np.pi / 180

        # Loop over different spatial frequencies
        for freq in frequencies:

            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (21,21), sigma=5.0,
                theta=theta_rad,
                lambd=1/freq,
                gamma=0.5,
                psi=0
            )

            # Apply Gabor filter
            filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)

            # Accumulate filtered responses
            accum += filtered

    # Normalize accumulated response to pixel range
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert back to RGB
    return cv2.cvtColor(accum, cv2.COLOR_GRAY2RGB)


# ===============================================================
# 5. UNSHARP MASKING (IMAGE SHARPENING)
# ===============================================================

def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.0, threshold=0):
    """
    Sharpen image using unsharp masking technique.
    """

    # Blur image using Gaussian filter
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)

    # Create sharpened image by subtracting blur
    sharpened = float(amount + 1) * img - float(amount) * blurred

    # Clip values to valid range
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255*np.ones(sharpened.shape))

    # Convert to uint8
    sharpened = sharpened.round().astype(np.uint8)

    # Optionally avoid sharpening low-contrast areas
    if threshold > 0:
        low_contrast_mask = np.abs(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)

    return sharpened


# ===============================================================
# 6. IMAGE NORMALIZATION
# ===============================================================

def normalize_image(img):
    """
    Normalize image pixel values to the range [0,1]
    using mean and standard deviation normalization.
    """

    # Convert image to float
    norm = img.astype(np.float32) / 255.0

    # Compute mean across spatial dimensions
    mean = np.mean(norm, axis=(0,1), keepdims=True)

    # Compute standard deviation
    std = np.std(norm, axis=(0,1), keepdims=True) + 1e-6

    # Standardize image
    norm = (norm - mean) / std

    # Scale values to [0,1]
    norm_min, norm_max = norm.min(), norm.max()
    norm = np.clip((norm - norm_min) / (norm_max - norm_min + 1e-6), 0.0, 1.0)

    return norm


# ===============================================================
# FULL PREPROCESSING PIPELINE
# ===============================================================

def preprocess_advanced(img):

    # Convert grayscale images to RGB if needed
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize image to fixed size for deep learning
    img = cv2.resize(img, (256, 256))

    # Step 1: Remove hair artifacts
    img = remove_hair(img)

    # Step 2: Apply bilateral smoothing
    img = bilateral_filter(img)

    # Step 3: Enhance image using wavelet transform
    img = wavelet_enhance(img)

    # Step 4: Apply Gabor filters for texture enhancement
    img = gabor_filter_bank(img)

    # Step 5: Sharpen image
    img = unsharp_mask(img)

    # Step 6: Normalize pixel values
    img = normalize_image(img)

    return img


# ===============================================================
# PROCESS A SINGLE IMAGE
# ===============================================================

def process_single_image(filename, data_dir, output_dir):

    # Create full path of input image
    filepath = os.path.join(data_dir, filename)

    # Process only image files
    if filename.endswith(('.jpg', '.png', '.jpeg')):

        try:

            # Read image using OpenCV
            image = cv2.imread(filepath, cv2.IMREAD_COLOR)

            # Skip if image cannot be read
            if image is None:
                logging.warning(f"Skipping {filename}")
                return

            # Apply preprocessing pipeline
            processed = preprocess_advanced(image)

            # Convert normalized image back to uint8
            processed_uint8 = (processed * 255).astype(np.uint8)

            # Define output file path
            output_path = os.path.join(output_dir, filename)

            # Save processed image
            cv2.imwrite(
                output_path,
                cv2.cvtColor(processed_uint8, cv2.COLOR_RGB2BGR)
            )

            logging.info(f"✅ {filename} processed")

        except Exception as e:

            # Log errors if processing fails
            logging.error(f"❌ {filename}: {e}")


# ===============================================================
# PROCESS ALL IMAGES IN THE DATASET
# ===============================================================

def process_all_images():

    # Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ]

    logging.info(f"{len(files)} images → hair removal + advanced processing pipeline")

    # Use multithreading for faster processing
    with ThreadPoolExecutor(max_workers=8) as executor:

        # Submit tasks for each image
        futures = [
            executor.submit(
                process_single_image,
                f,
                str(DATA_DIR),
                str(OUTPUT_DIR)
            )
            for f in files
        ]

        # Wait for all threads to complete
        for future in futures:
            future.result()

    print(f"🎉 COMPLETE: {OUTPUT_DIR}")


# ===============================================================
# RUN THE PIPELINE
# ===============================================================

process_all_images()
