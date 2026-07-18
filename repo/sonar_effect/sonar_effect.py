"""
Sonar Effect Pipeline for:
- HAM10000
- ISIC 2024

Applies sonar transformation on outputs of:
- pipeline1
- pipeline2
- pipeline3
- pipeline4

Features:
✔ Lesion = strong sonar
✔ Background = soft sonar
✔ Better visual separation
✔ Improved pseudo-mask
✔ Supports single folder / multiple folders / multiple pipelines
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# =========================================================
# SONAR TRANSFORMATION
# =========================================================
def apply_sonar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    sonar = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    return sonar


# =========================================================
# IMPROVED PSEUDO MASK
# =========================================================
def generate_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Slight dilation
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# =========================================================
# NEW FUSION
# =========================================================
def fuse_lesion_background(original, sonar, mask):
    mask_bin = (mask > 0)
    mask_3ch = np.stack([mask_bin] * 3, axis=-1)

    sonar_strong = cv2.convertScaleAbs(sonar, alpha=1.5, beta=20)
    sonar_soft = (sonar * 0.4).astype(np.uint8)

    fused = np.where(mask_3ch, sonar_strong, sonar_soft)
    return fused.astype(np.uint8)


# =========================================================
# PROCESS SINGLE IMAGE
# =========================================================
def process_image(img_path, output_folder, save_mask=False):
    filename = img_path.name

    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"❌ Cannot read {filename}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sonar_img = apply_sonar(img_rgb)
    mask = generate_mask(img_rgb)
    fused_img = fuse_lesion_background(img_rgb, sonar_img, mask)

    sonar_folder = output_folder / "sonar_only"
    fused_folder = output_folder / "fused"
    mask_folder = output_folder / "pseudo_masks"

    sonar_folder.mkdir(parents=True, exist_ok=True)
    fused_folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(sonar_folder / filename),
        cv2.cvtColor(sonar_img, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        str(fused_folder / filename),
        cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
    )

    if save_mask:
        mask_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_folder / f"{img_path.stem}.png"), mask)

    logging.info(f"✅ Processed: {filename}")


# =========================================================
# COLLECT IMAGES FROM ONE FOLDER
# =========================================================
def collect_images(folder):
    folder = Path(folder)

    images = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

    return images


# =========================================================
# PROCESS ONE INPUT FOLDER
# =========================================================
def process_pipeline(input_folder, output_folder, workers=8, save_mask=False):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_folder)

    logging.info(f"📂 Input: {input_folder}")
    logging.info(f"📂 Output: {output_folder}")
    logging.info(f"📊 Found {len(images)} images")

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for img in images:
            futures.append(
                executor.submit(process_image, img, output_folder, save_mask)
            )

        for future in as_completed(futures):
            future.result()

    logging.info("🎉 Pipeline completed")


# =========================================================
# PROCESS HAM10000 TWO PARTS AS ONE INPUT
# =========================================================
def process_ham_parts(part1, part2, output_folder, workers=8, save_mask=False):
    part1 = Path(part1)
    part2 = Path(part2)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(part1.glob("*.jpg")) +
        list(part1.glob("*.jpeg")) +
        list(part1.glob("*.png")) +
        list(part2.glob("*.jpg")) +
        list(part2.glob("*.jpeg")) +
        list(part2.glob("*.png"))
    )

    logging.info(f"📂 HAM part1: {part1}")
    logging.info(f"📂 HAM part2: {part2}")
    logging.info(f"📂 Output: {output_folder}")
    logging.info(f"📊 Found {len(images)} HAM10000 images")

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for img in images:
            futures.append(
                executor.submit(process_image, img, output_folder, save_mask)
            )

        for future in as_completed(futures):
            future.result()

    logging.info("🎉 HAM10000 parts pipeline completed")


# =========================================================
# PROCESS MULTIPLE CONFIGS
# =========================================================
def process_multiple(configs, workers=8, save_mask=False):
    for cfg in configs:
        name = cfg.get("name", "unnamed")
        mode = cfg.get("mode", "single")

        logging.info(f"🚀 Processing config: {name}")

        if mode == "single":
            process_pipeline(
                input_folder=cfg["input"],
                output_folder=cfg["output"],
                workers=workers,
                save_mask=save_mask
            )

        elif mode == "ham_parts":
            process_ham_parts(
                part1=cfg["part1"],
                part2=cfg["part2"],
                output_folder=cfg["output"],
                workers=workers,
                save_mask=save_mask
            )

        else:
            logging.warning(f"⚠️ Unknown mode for config: {name}")


# =========================================================
# BUILD DEFAULT CONFIGS FOR 4 PIPELINES × 2 DATASETS
# =========================================================
def build_default_configs():
    configs = [
        # -------------------------------------------------
        # ISIC 2024 - pipeline1
        # -------------------------------------------------
        {
            "name": "isic2024_pipeline1",
            "mode": "single",
            "input": "/path/to/pipeline1/processed_images",
            "output": "/path/to/pipeline1/sonar_output"
        },

        # -------------------------------------------------
        # ISIC 2024 - pipeline2
        # -------------------------------------------------
        {
            "name": "isic2024_pipeline2",
            "mode": "single",
            "input": "/path/to/pipeline2/processed_images",
            "output": "/path/to/pipeline2/sonar_output"
        },

        # -------------------------------------------------
        # ISIC 2024 - pipeline3
        # -------------------------------------------------
        {
            "name": "isic2024_pipeline3",
            "mode": "single",
            "input": "/path/to/pipeline3/processed_images",
            "output": "/path/to/pipeline3/sonar_output"
        },

        # -------------------------------------------------
        # ISIC 2024 - pipeline4
        # -------------------------------------------------
        {
            "name": "isic2024_pipeline4",
            "mode": "single",
            "input": "/path/to/pipeline4/processed_images",
            "output": "/path/to/pipeline4/sonar_output"
        },

        # -------------------------------------------------
        # HAM10000 - pipeline1
        # -------------------------------------------------
        {
            "name": "ham10000_pipeline1",
            "mode": "single",
            "input": "/path/to/ham10000/pipeline1/processed_images",
            "output": "/path/to/ham10000/pipeline1/sonar_output"
        },

        # -------------------------------------------------
        # HAM10000 - pipeline2
        # -------------------------------------------------
        {
            "name": "ham10000_pipeline2",
            "mode": "single",
            "input": "/path/to/ham10000/pipeline2/processed_images",
            "output": "/path/to/ham10000/pipeline2/sonar_output"
        },

        # -------------------------------------------------
        # HAM10000 - pipeline3
        # -------------------------------------------------
        {
            "name": "ham10000_pipeline3",
            "mode": "single",
            "input": "/path/to/ham10000/pipeline3/processed_images",
            "output": "/path/to/ham10000/pipeline3/sonar_output"
        },

        # -------------------------------------------------
        # HAM10000 - pipeline4
        # -------------------------------------------------
        {
            "name": "ham10000_pipeline4",
            "mode": "single",
            "input": "/path/to/ham10000/pipeline4/processed_images",
            "output": "/path/to/ham10000/pipeline4/sonar_output"
        }
    ]

    return configs


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="Input folder")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument(
        "--run_default",
        action="store_true",
        help="Run default 4 pipelines for HAM10000 and ISIC2024"
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    # Manual single run
    if args.input and args.output:
        process_pipeline(
            input_folder=args.input,
            output_folder=args.output,
            workers=args.workers,
            save_mask=args.save_mask
        )

    # Default multi-run
    elif args.run_default:
        configs = build_default_configs()
        process_multiple(
            configs=configs,
            workers=args.workers,
            save_mask=args.save_mask
        )

    # Fallback example
    else:
        configs = [
            {
                "name": "isic2024_pipeline4_example",
                "mode": "single",
                "input": "/path/to/pipeline4/processed_images",
                "output": "/path/to/pipeline4/sonar_output"
            }
        ]

        process_multiple(
            configs=configs,
            workers=8,
            save_mask=True
        )
