"""This Python script provides a complete pipeline for processing the HAM10000 skin lesion dataset. 
It begins by merging images from multiple folders into a single directory, then loads and cleans the associated metadata, 
filling missing ages with the median. The dataset is visualized through bar charts of class distribution, age histograms, 
and summary tables to understand its composition. Each image is preprocessed by resizing, converting to RGB, removing hair artifacts, 
applying white balance and CLAHE for contrast enhancement, and normalizing pixel values. The pipeline also generates sonar-style color maps for enhanced visualization. 
All processing steps are applied in parallel for efficiency, and the resulting outputs include merged raw images, fully preprocessed images, sonar-enhanced images, 
and visual summaries, making the dataset ready for analysis or deep learning applications."""

# ============================
# IMPORTS
# ============================
import os                     # For interacting with the operating system (paths, files)
import cv2                    # OpenCV for image processing
import shutil                 # For copying and moving files
import logging                # For logging info and errors
import numpy as np            # Numerical operations, arrays
import pandas as pd           # Data manipulation
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns         # Advanced plotting for statistical data
from pathlib import Path      # Modern path handling (cross-platform)
from concurrent.futures import ThreadPoolExecutor  # Parallel processing

# ============================
# PATHS
# ============================
ROOT_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/")  # Root dataset directory

# Raw image folders (part 1 and part 2)
PART1_DIR = ROOT_DIR / "HAM10000_images_part_1"
PART2_DIR = ROOT_DIR / "HAM10000_images_part_2"

# Output directories
ALL_IMAGES_DIR = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_images_all")
PROCESSED_DIR  = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_processed_images")
SONAR_DIR      = Path("/home/aboubakr/Descargas/article4/ham10000/pipeline2/HAM10000_sonar_images")

# Metadata CSV file
META_PATH = ROOT_DIR / "HAM10000_metadata.csv"

# Setup logging format
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================
# 1. MERGE IMAGE FOLDERS
# ============================
def merge_image_folders():
    # Create output folder if it does not exist
    ALL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Loop through both raw image directories
    for sub in [PART1_DIR, PART2_DIR]:
        for img_path in sub.glob("*.jpg"):  # Iterate over all JPG files
            shutil.copy2(img_path, ALL_IMAGES_DIR / img_path.name)  # Copy to merged folder

    logging.info(f"✅ Images merged into {ALL_IMAGES_DIR}")  # Log success

# ============================
# 2. LOAD & CLEAN METADATA
# ============================
def load_and_clean_metadata():
    meta = pd.read_csv(META_PATH)  # Load metadata CSV into DataFrame

    # Fill missing age values with the median age
    median_age = meta["age"].median()
    meta["age_filled"] = meta["age"].fillna(median_age)

    return meta  # Return cleaned DataFrame

# ============================
# 3. DATASET VISUALIZATION
# ============================
def plot_dataset_statistics(meta):
    sns.set(style="whitegrid", font_scale=1.1)  # Set Seaborn style

    # Count number of images per diagnosis class
    class_counts = meta["dx"].value_counts().sort_values(ascending=False)

    fig = plt.figure(figsize=(18, 12))  # Create figure

    # ---- Bar plot for class distribution ----
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)  # 2x3 grid, top-left, span 2 columns
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        palette="viridis",
        ax=ax1
    )
    ax1.set_title("HAM10000 Class Distribution", weight="bold")
    ax1.set_xlabel("Diagnosis (dx)")
    ax1.set_ylabel("Number of Images")
    ax1.tick_params(axis="x", rotation=45)  # Rotate x labels

    # ---- Histogram for age distribution ----
    ax2 = plt.subplot2grid((2, 3), (0, 2))  # Top-right cell
    sns.histplot(meta["age_filled"], bins=20, kde=True, ax=ax2)
    ax2.set_title("Age Distribution", weight="bold")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Count")

    # ---- Table for class counts and percentages ----
    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)  # Bottom row spans all 3 columns
    ax3.axis("off")  # Hide axes

    tbl = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values,
        "Percent (%)": (class_counts.values / len(meta) * 100).round(2)
    })

    table = ax3.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax3.set_title("Class Counts & Percentages", pad=10, weight="bold")

    plt.tight_layout()
    plt.show()  # Display plots

# ============================
# 4. IMAGE PREPROCESSING
# ============================
def preprocess_final_image(image):
    # Ensure image is RGB and resize to 256x256
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.uint8)

    # Hair removal using morphological blackhat
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    if np.sum(mask) > 0:
        image = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)

    # White balance (simple scaling)
    img = image.astype(np.float32)
    avg = np.mean(img, axis=(0, 1))
    gray_avg = np.mean(avg)
    scale = gray_avg / (avg + 1e-6)
    img *= scale
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Apply CLAHE to the L-channel in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    L = clahe.apply(L)
    lab = cv2.merge((L, A, B))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize image to [0,1]
    img = img.astype(np.float32) / 255.0
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return img

def process_single_image(filename):
    try:
        # Load image
        img = cv2.imread(str(ALL_IMAGES_DIR / filename))
        if img is None:
            return  # Skip if file cannot be read

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess image
        processed = preprocess_final_image(img)

        # Convert back to uint8 BGR for saving
        out = (processed * 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Save processed image
        cv2.imwrite(str(PROCESSED_DIR / filename), out)
        logging.info(f"✅ {filename}")

    except Exception as e:
        logging.error(f"❌ {filename}: {e}")

def preprocess_all_images():
    # Create folder if it doesn't exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # List all jpg files
    files = [f.name for f in ALL_IMAGES_DIR.iterdir() if f.suffix.lower() == ".jpg"]

    # Parallel processing with 8 threads
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(process_single_image, files)

# ============================
# 5. SONAR EFFECT
# ============================
def apply_sonar_effect(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    return cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_JET)  # Apply sonar colormap

def process_single_sonar(filename):
    try:
        # Load processed image
        img = cv2.imread(str(PROCESSED_DIR / filename))
        sonar = apply_sonar_effect(img)  # Apply sonar effect
        cv2.imwrite(str(SONAR_DIR / filename), sonar)  # Save result
        logging.info(f"🌊 {filename}")
    except Exception as e:
        logging.error(f"❌ {filename}: {e}")

def generate_sonar_images():
    # Create folder if it doesn't exist
    SONAR_DIR.mkdir(parents=True, exist_ok=True)

    # List all processed jpg files
    files = [f.name for f in PROCESSED_DIR.iterdir() if f.suffix.lower() == ".jpg"]

    # Parallel processing
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(process_single_sonar, files)

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    merge_image_folders()         # Step 1: Merge all images
    meta = load_and_clean_metadata()  # Step 2: Load metadata
    plot_dataset_statistics(meta)     # Step 3: Visualize dataset
    preprocess_all_images()           # Step 4: Preprocess all images
    generate_sonar_images()           # Step 5: Generate sonar effect images

    print("🎉 HAM10000 pipeline completed successfully!")  # Final message
