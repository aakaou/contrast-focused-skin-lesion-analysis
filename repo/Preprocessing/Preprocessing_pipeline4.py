import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


# ==========================================================
# 1. DULLRAZOR HAIR REMOVAL
# ==========================================================
def dullrazor_hair_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect dark hair using blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold hair regions
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Refine mask
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)

    # Inpaint hair regions
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result


# ==========================================================
# 2. INPAINTING (GENERAL CLEANUP)
# ==========================================================
def refine_inpainting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect small artifacts / low intensity noise
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 5)

    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return result


# ==========================================================
# 3. CLAHE ENHANCEMENT
# ==========================================================
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


# ==========================================================
# 4. INTENSITY NORMALIZATION
# ==========================================================
def normalize_intensity(img):
    img = img.astype(np.float32) / 255.0

    mean = np.mean(img, axis=(0, 1), keepdims=True)
    std = np.std(img, axis=(0, 1), keepdims=True) + 1e-6

    img = (img - mean) / std
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return (img * 255).astype(np.uint8)


# ==========================================================
# SINGLE IMAGE PIPELINE 4
# ==========================================================
def pipeline4_single_image(img, img_size=256):
    img = dullrazor_hair_removal(img)
    img = refine_inpainting(img)

    img = cv2.resize(
        img,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    img = apply_clahe(img)
    img = normalize_intensity(img)

    return img


# ==========================================================
# PROCESS ONE IMAGE
# ==========================================================
def process_one_image_pipeline4(img_path, output_path, img_size=256):
    try:
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"❌ Could not read: {img_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = pipeline4_single_image(
            img,
            img_size=img_size
        )

        Image.fromarray(img).save(output_path / img_path.name)

    except Exception as e:
        print(f"❌ Error {img_path.name}: {e}")


# ==========================================================
# COLLECT IMAGES
# ==========================================================
def collect_images(folder):
    folder = Path(folder)

    images = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

    return images


# ==========================================================
# PIPELINE 4 FOR ONE FOLDER
# ==========================================================
def pipeline4(
    input_path,
    output_path,
    img_size=256,
    max_workers=8
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)

    print(f"\n📂 Input folder: {input_path}")
    print(f"📂 Output folder: {output_path}")
    print("Found images:", len(images))
    print("Running Pipeline 4 – Optimized Contrast...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image_pipeline4(
                        img_path,
                        output_path,
                        img_size
                    ),
                    images
                ),
                total=len(images)
            )
        )

    print("\n✅ Pipeline 4 completed.")
    return output_path


# ==========================================================
# PIPELINE 4 FOR HAM10000 PARTS
# ==========================================================
def pipeline4_ham_parts(
    ham_part1,
    ham_part2,
    output_path,
    img_size=256,
    max_workers=8
):
    ham_part1 = Path(ham_part1)
    ham_part2 = Path(ham_part2)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(ham_part1.glob("*.jpg")) +
        list(ham_part1.glob("*.jpeg")) +
        list(ham_part1.glob("*.png")) +
        list(ham_part2.glob("*.jpg")) +
        list(ham_part2.glob("*.jpeg")) +
        list(ham_part2.glob("*.png"))
    )

    print(f"\n📂 HAM10000 part 1: {ham_part1}")
    print(f"📂 HAM10000 part 2: {ham_part2}")
    print(f"📂 Output folder: {output_path}")
    print("Found HAM10000 images:", len(images))
    print("Running Pipeline 4 for HAM10000...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image_pipeline4(
                        img_path,
                        output_path,
                        img_size
                    ),
                    images
                ),
                total=len(images)
            )
        )

    print("\n✅ HAM10000 Pipeline 4 completed.")
    return output_path


# ==========================================================
# PIPELINE 4 FOR BOTH DATASETS
# ==========================================================
def pipeline4_two_datasets(
    ham_images_path=None,
    ham_part1=None,
    ham_part2=None,
    isic_images_path=None,
    ham_output_path="pipeline4_ham10000",
    isic_output_path="pipeline4_isic2024",
    img_size=256,
    max_workers=8
):
    processed_ham = None
    processed_isic = None

    if ham_images_path is not None:
        processed_ham = pipeline4(
            input_path=ham_images_path,
            output_path=ham_output_path,
            img_size=img_size,
            max_workers=max_workers
        )
    elif ham_part1 is not None and ham_part2 is not None:
        processed_ham = pipeline4_ham_parts(
            ham_part1=ham_part1,
            ham_part2=ham_part2,
            output_path=ham_output_path,
            img_size=img_size,
            max_workers=max_workers
        )

    if isic_images_path is not None:
        processed_isic = pipeline4(
            input_path=isic_images_path,
            output_path=isic_output_path,
            img_size=img_size,
            max_workers=max_workers
        )

    return processed_ham, processed_isic


# ==========================================================
# VISUALIZATION
# ==========================================================
def show_samples(folder, n=5):
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

    # =========================
    # ISIC 2024
    # =========================
    ISIC2024_INPUT_PATH = "/path/to/isic2024/train-image/image"
    ISIC2024_OUTPUT_PATH = "/path/to/pipeline4_isic2024/processed_images"

    # =========================
    # HAM10000
    # Choose one mode:
    # 1) merged folder
    # 2) part_1 + part_2
    # =========================

    # Option 1: merged HAM10000 folder
    HAM10000_INPUT_PATH = "/path/to/HAM10000_images_all"

    # Option 2: separate parts
    HAM10000_PART1 = "/path/to/HAM10000_images_part_1"
    HAM10000_PART2 = "/path/to/HAM10000_images_part_2"

    HAM10000_OUTPUT_PATH = "/path/to/pipeline4_ham10000/processed_images"

    USE_HAM_MERGED_FOLDER = True

    if USE_HAM_MERGED_FOLDER:
        processed_ham, processed_isic = pipeline4_two_datasets(
            ham_images_path=HAM10000_INPUT_PATH,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_path=HAM10000_OUTPUT_PATH,
            isic_output_path=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )
    else:
        processed_ham, processed_isic = pipeline4_two_datasets(
            ham_part1=HAM10000_PART1,
            ham_part2=HAM10000_PART2,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_path=HAM10000_OUTPUT_PATH,
            isic_output_path=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )

    print("\n✅ All Pipeline 4 preprocessing finished!")
    print("HAM10000 processed folder:", processed_ham)
    print("ISIC 2024 processed folder:", processed_isic)

    if processed_ham is not None:
        show_samples(processed_ham, n=5)

    if processed_isic is not None:
        show_samples(processed_isic, n=5)
