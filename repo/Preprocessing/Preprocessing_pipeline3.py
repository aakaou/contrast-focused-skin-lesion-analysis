import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import pywt


# ==========================================================
# HAIR REMOVAL
# ==========================================================
def remove_hair(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    if np.sum(mask) > 0:
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        return result

    return img


# ==========================================================
# BILATERAL FILTER
# ==========================================================
def bilateral_filter(img):
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


# ==========================================================
# WAVELET ENHANCEMENT
# ==========================================================
def wavelet_enhancement(img):
    img = img.astype(np.float32) / 255.0
    channels = []

    for i in range(3):
        coeffs = pywt.dwt2(img[:, :, i], "haar")
        cA, (cH, cV, cD) = coeffs

        cA = cA * 1.2
        cH = cH * 1.3
        cV = cV * 1.3
        cD = cD * 1.3

        enhanced = pywt.idwt2((cA, (cH, cV, cD)), "haar")

        enhanced = cv2.resize(
            enhanced,
            (img.shape[1], img.shape[0])
        )

        channels.append(enhanced)

    out = np.stack(channels, axis=2)
    out = np.clip(out, 0, 1)

    return (out * 255).astype(np.uint8)


# ==========================================================
# GABOR FILTER BANK
# ==========================================================
def gabor_enhancement(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernels = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=5,
            theta=theta,
            lambd=10,
            gamma=0.5,
            psi=0
        )
        kernels.append(kernel)

    filtered = np.zeros_like(gray, dtype=np.float32)

    for k in kernels:
        filtered += cv2.filter2D(gray, cv2.CV_32F, k)

    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
    filtered = filtered.astype(np.uint8)

    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return filtered


# ==========================================================
# UNSHARP MASKING
# ==========================================================
def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ==========================================================
# NORMALIZATION
# ==========================================================
def normalize(img):
    img = img.astype(np.float32) / 255.0

    mean = img.mean(axis=(0, 1), keepdims=True)
    std = img.std(axis=(0, 1), keepdims=True) + 1e-6

    img = (img - mean) / std
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    return (img * 255).astype(np.uint8)


# ==========================================================
# SINGLE IMAGE PIPELINE 3
# ==========================================================
def pipeline3_single_image(img, img_size=256):
    img = cv2.resize(img, (img_size, img_size))

    img = remove_hair(img)
    img = bilateral_filter(img)
    img = wavelet_enhancement(img)

    gabor = gabor_enhancement(img)
    img = cv2.addWeighted(img, 0.7, gabor, 0.3, 0)

    img = unsharp_mask(img)
    img = normalize(img)

    return img


# ==========================================================
# PROCESS ONE IMAGE
# ==========================================================
def process_one_image_pipeline3(img_path, output_path, img_size=256):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not read: {img_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = pipeline3_single_image(img, img_size=img_size)

        Image.fromarray(img).save(output_path / img_path.name)

    except Exception as e:
        print(f"❌ Error {img_path.name}: {e}")


# ==========================================================
# COLLECT IMAGES FROM ONE FOLDER
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
# PIPELINE 3 FOR ONE DATASET FOLDER
# ==========================================================
def pipeline3(
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
    print("Running Pipeline 3 – Texture Enhancement...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image_pipeline3(
                        img_path,
                        output_path,
                        img_size
                    ),
                    images
                ),
                total=len(images)
            )
        )

    print("\n✅ Pipeline 3 completed.")
    return output_path


# ==========================================================
# PIPELINE 3 FOR HAM10000 PART_1 + PART_2
# ==========================================================
def pipeline3_ham_parts(
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
    print("Running Pipeline 3 for HAM10000...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(
                    lambda img_path: process_one_image_pipeline3(
                        img_path,
                        output_path,
                        img_size
                    ),
                    images
                ),
                total=len(images)
            )
        )

    print("\n✅ HAM10000 Pipeline 3 completed.")
    return output_path


# ==========================================================
# PIPELINE 3 FOR BOTH DATASETS
# ==========================================================
def pipeline3_two_datasets(
    ham_images_path=None,
    ham_part1=None,
    ham_part2=None,
    isic_images_path=None,
    ham_output_path="pipeline3_ham10000",
    isic_output_path="pipeline3_isic2024",
    img_size=256,
    max_workers=8
):
    processed_ham = None
    processed_isic = None

    if ham_images_path is not None:
        processed_ham = pipeline3(
            input_path=ham_images_path,
            output_path=ham_output_path,
            img_size=img_size,
            max_workers=max_workers
        )
    elif ham_part1 is not None and ham_part2 is not None:
        processed_ham = pipeline3_ham_parts(
            ham_part1=ham_part1,
            ham_part2=ham_part2,
            output_path=ham_output_path,
            img_size=img_size,
            max_workers=max_workers
        )

    if isic_images_path is not None:
        processed_isic = pipeline3(
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
    ISIC2024_OUTPUT_PATH = "/path/to/pipeline3_isic2024/processed_images"

    # =========================
    # HAM10000
    # Use one option:
    #   1) merged folder
    #   2) part_1 + part_2
    # =========================

    # Option 1: merged folder
    HAM10000_INPUT_PATH = "/path/to/HAM10000_images_all"

    # Option 2: separate folders
    HAM10000_PART1 = "/path/to/HAM10000_images_part_1"
    HAM10000_PART2 = "/path/to/HAM10000_images_part_2"

    HAM10000_OUTPUT_PATH = "/path/to/pipeline3_ham10000/processed_images"

    USE_HAM_MERGED_FOLDER = True

    if USE_HAM_MERGED_FOLDER:
        processed_ham, processed_isic = pipeline3_two_datasets(
            ham_images_path=HAM10000_INPUT_PATH,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_path=HAM10000_OUTPUT_PATH,
            isic_output_path=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )
    else:
        processed_ham, processed_isic = pipeline3_two_datasets(
            ham_part1=HAM10000_PART1,
            ham_part2=HAM10000_PART2,
            isic_images_path=ISIC2024_INPUT_PATH,
            ham_output_path=HAM10000_OUTPUT_PATH,
            isic_output_path=ISIC2024_OUTPUT_PATH,
            img_size=256,
            max_workers=8
        )

    print("\n✅ All Pipeline 3 preprocessing finished!")
    print("HAM10000 processed folder:", processed_ham)
    print("ISIC 2024 processed folder:", processed_isic)

    if processed_ham is not None:
        show_samples(processed_ham, n=5)

    if processed_isic is not None:
        show_samples(processed_isic, n=5)
