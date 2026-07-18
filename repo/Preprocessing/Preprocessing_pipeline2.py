"""
Pipeline 2 - Advanced Preprocessing for HAM10000 and ISIC 2024

Steps:
1. Resize images
2. Hair removal (Blackhat + inpainting)
3. White balance
4. CLAHE contrast enhancement
5. Normalize pixel values

Supports:
- HAM10000 merged folder or part folder(s)
- ISIC 2024 image folder

Example:
    processed_ham, processed_isic = preprocess_two_datasets(
        ham_images_path="/path/to/HAM10000_images_all",
        isic_images_path="/home/aboubakr/Descargas/article4/isic2016_2020/isic2024/train-image/image",
        ham_output_folder="/home/aboubakr/Descargas/article4/pipeline2_ham10000",
        isic_output_folder="/home/aboubakr/Descargas/article4/isic2016_2020/pipeline2_isic2024",
        img_size=256
    )
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# ==========================================================
# HAIR REMOVAL
# ==========================================================
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        10,
        255,
        cv2.THRESH_BINARY
    )

    if np.sum(mask) > 0:
        result = cv2.inpaint(
            img,
            mask,
            3,
            cv2.INPAINT_TELEA
        )
        return result

    return img


# ==========================================================
# WHITE BALANCE
# ==========================================================
def white_balance(img):
    img = img.astype(np.float32)

    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])

    avg = (avg_r + avg_g + avg_b) / 3.0

    img[:, :, 0] *= avg / (avg_r + 1e-6)
    img[:, :, 1] *= avg / (avg_g + 1e-6)
    img[:, :, 2] *= avg / (avg_b + 1e-6)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ==========================================================
# CLAHE
# ==========================================================
def apply_clahe(img):
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return enhanced


# ==========================================================
# NORMALIZATION
# ==========================================================
def normalize_image(img):
    img = img.astype(np.float32) / 255.0

    mean = img.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = img.std(
        axis=(0, 1),
        keepdims=True
    ) + 1e-6

    img = (img - mean) / std

    img = (
        (img - img.min()) /
        (img.max() - img.min() + 1e-6)
    )

    img = (img * 255).astype(np.uint8)

    return img


# ==========================================================
# FULL SINGLE-IMAGE PIPELINE
# ==========================================================
def preprocess_single_image_array(img, img_size=256):
    img = remove_hair(img)
    img = white_balance(img)
    img = apply_clahe(img)

    img = cv2.resize(
        img,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    img = normalize_image(img)
    return img


# ==========================================================
# PROCESS ONE IMAGE FILE
# ==========================================================
def process_one_image(img_path, output_folder, img_size=256):
    try:
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"❌ Could not read: {img_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processed = preprocess_single_image_array(
            img,
            img_size=img_size
        )

        Image.fromarray(processed).save(
            output_folder / img_path.name
        )

    except Exception as e:
        print(f"❌ Error {img_path.name}: {e}")


# ==========================================================
# COLLECT IMAGE FILES
# ==========================================================
def collect_image_paths(images_path):
    images_path = Path(images_path)

    image_paths = sorted(
        list(images_path.glob("*.jpg")) +
        list(images_path.glob("*.jpeg")) +
        list(images_path.glob("*.png"))
    )

    return image_paths


# ==========================================================
# PREPROCESS ONE DATASET
# ==========================================================
def preprocess_pipeline2(
    images_path,
    output_folder,
    img_size=256,
    max_workers=8
):
    images_path = Path(images_path)
    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = collect_image_paths(images_path)

    print(f"\n📂 Input folder: {images_path}")
    print(f"📂 Output folder: {output_folder}")
    print(f"🔍 Found: {len(image_paths)} images")
    print("🚀 Running Pipeline 2...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image(
                        img_path,
                        output_folder,
                        img_size
                    ),
                    image_paths
                ),
                total=len(image_paths)
            )
        )

    print("\n✅ Pipeline 2 completed.")
    return output_folder


# ==========================================================
# PREPROCESS HAM10000 FROM TWO PARTS
# ==========================================================
def preprocess_ham10000_from_parts(
    ham_part1,
    ham_part2,
    output_folder,
    img_size=256,
    max_workers=8
):
    ham_part1 = Path(ham_part1)
    ham_part2 = Path(ham_part2)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(ham_part1.glob("*.jpg")) +
        list(ham_part1.glob("*.jpeg")) +
        list(ham_part1.glob("*.png")) +
        list(ham_part2.glob("*.jpg")) +
        list(ham_part2.glob("*.jpeg")) +
        list(ham_part2.glob("*.png"))
    )

    print(f"\n📂 HAM10000 part 1: {ham_part1}")
    print(f"📂 HAM10000 part 2: {ham_part2}")
    print(f"📂 Output folder: {output_folder}")
    print(f"🔍 Found: {len(image_paths)} HAM10000 images")
    print("🚀 Running Pipeline 2 for HAM10000...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image(
                        img_path,
                        output_folder,
                        img_size
                    ),
                    image_paths
                ),
                total=len(image_paths)
            )
        )

    print("\n✅ HAM10000 Pipeline 2 completed.")
    return output_folder


# ==========================================================
# PREPROCESS BOTH DATASETS
# ==========================================================
def preprocess_two_datasets(
    ham_images_path=None,
    ham_part1=None,
    ham_part2=None,
    isic_images_path=None,
    ham_output_folder="pipeline2_ham10000",
    isic_output_folder="pipeline2_isic2024",
    img_size=256,
    max_workers=8
):
    processed_ham = None
    processed_isic = None

    # HAM10000: merged folder OR two parts
    if ham_images_path is not None:
        processed_ham = preprocess_pipeline2(
            images_path=ham_images_path,
            output_folder=ham_output_folder,
            img_size=img_size,
            max_workers=max_workers
        )
    elif ham_part1 is not None and ham_part2 is not None:
        processed_ham = preprocess_ham10000_from_parts(
            ham_part1=ham_part1,
            ham_part2=ham_part2,
            output_folder=ham_output_folder,
            img_size=img_size,
            max_workers=max_workers
        )

    # ISIC 2024
    if isic_images_path is not None:
        processed_isic = preprocess_pipeline2(
            images_path=isic_images_path,
            output_folder=isic_output_folder,
            img_size=img_size,
            max_workers=max_workers
        )

    return processed_ham, processed_isic


# ==========================================================
# SHOW IMAGES
# ==========================================================
def show_first_images(folder, n=5):
    folder = Path(folder)

    images = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

    plt.figure(figsize=(15, 5))

    for i, img_path in enumerate(images[:n]):
        img = Image.open(img_path)

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(img_path.name, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    # ------------------------------------------------------
    # ISIC 2024 path
    # ------------------------------------------------------
    ISIC2024_INPUT_PATH = "/path/to/isic2016_2020/isic2024/train-image/image"
    ISIC2024_OUTPUT_PATH = "/path/to/pipeline2_isic2024/processed_images"

    # ------------------------------------------------------
    # HAM10000 path
    # Use ONE of the following options:
    # Option A: merged folder
    # Option B: part_1 and part_2 folders
    # ------------------------------------------------------

    # Option A: merged HAM10000 folder
    HAM10000_INPUT_PATH = "/path/to/HAM10000_images_all"

    # Option B: separate HAM10000 parts
    HAM10000_PART1 = "/path/to/HAM10000_images_part_1"
    HAM10000_PART2 = "/path/to/HAM10000_images_part_2"

    HAM10000_OUTPUT_PATH = "/path/to/pipeline2_ham10000/processed_images"

    # ------------------------------------------------------
    # Choose one HAM10000 mode
    # ------------------------------------------------------
    USE_HAM_MERGED_FOLDER = True

    if USE_HAM_MERGED_FOLDER:
        processed_ham, processed_isic = preprocess_two_datasets(
            ham_images_path=HAM10000_INPUT_PATH,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_folder=HAM10000_OUTPUT_PATH,
            isic_output_folder=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )
    else:
        processed_ham, processed_isic = preprocess_two_datasets(
            ham_part1=HAM10000_PART1,
            ham_part2=HAM10000_PART2,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_folder=HAM10000_OUTPUT_PATH,
            isic_output_folder=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )

    print("\n✅ All preprocessing finished!")
    print("HAM10000 processed folder:", processed_ham)
    print("ISIC 2024 processed folder:", processed_isic)

    if processed_ham is not None:
        show_first_images(processed_ham, n=5)

    if processed_isic is not None:
        show_first_images(processed_isic, n=5)
