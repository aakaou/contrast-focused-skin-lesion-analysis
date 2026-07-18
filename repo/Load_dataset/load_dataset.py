"""
Unified dataset loader for HAM10000 and ISIC 2024 (Kaggle)

Responsibilities:
1. Inspect Kaggle input directory
2. Detect dataset type automatically
3. Prepare images
4. Load metadata
"""

import os
import shutil
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Inspect Kaggle input directory
# --------------------------------------------------
def inspect_kaggle_input(input_path="/kaggle/input"):
    """
    Print all files available in Kaggle input directory
    """
    print("🔎 Inspecting Kaggle input directory...\n")

    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# --------------------------------------------------
# 2. Detect dataset type
# --------------------------------------------------
def detect_dataset_type(dataset_root):
    """
    Detect whether dataset is HAM10000 or ISIC2024
    """
    dataset_root = Path(dataset_root)

    ham_metadata = dataset_root / "HAM10000_metadata.csv"
    ham_part1 = dataset_root / "HAM10000_images_part_1"
    ham_part2 = dataset_root / "HAM10000_images_part_2"

    isic_metadata = dataset_root / "train-metadata.csv"
    isic_image_dir = dataset_root / "train-image"
    isic_hdf5 = dataset_root / "train-image.hdf5"

    if ham_metadata.exists() and (ham_part1.exists() or ham_part2.exists()):
        return "HAM10000"

    if isic_metadata.exists() and (isic_image_dir.exists() or isic_hdf5.exists()):
        return "ISIC2024"

    raise FileNotFoundError(
        f"Could not detect dataset type in: {dataset_root}\n"
        f"Expected HAM10000 or ISIC2024 file structure."
    )


# --------------------------------------------------
# 3A. Prepare HAM10000 images
# --------------------------------------------------
def prepare_ham10000_images(dataset_root, output_dir):
    """
    Combine HAM10000 images from part_1 and part_2 into one folder
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    part1 = dataset_root / "HAM10000_images_part_1"
    part2 = dataset_root / "HAM10000_images_part_2"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Combining HAM10000 images...")

    for folder in [part1, part2]:
        if folder.exists():
            for img in folder.glob("*.jpg"):
                shutil.copy2(img, output_dir / img.name)

    print(f"✅ HAM10000 images ready at: {output_dir}")
    return output_dir


# --------------------------------------------------
# 3B. Prepare ISIC2024 images
# --------------------------------------------------
def prepare_isic2024_images(dataset_root, output_dir=None, use_copy=False):
    """
    Prepare ISIC 2024 image source

    If train-image exists:
      - use_copy=False -> return folder path directly
      - use_copy=True  -> copy files to output_dir

    If train-image.hdf5 exists:
      - return hdf5 path
    """
    dataset_root = Path(dataset_root)
    train_image_dir = dataset_root / "train-image"
    train_image_hdf5 = dataset_root / "train-image.hdf5"

    if train_image_dir.exists():
        print(f"📂 Found ISIC2024 image folder: {train_image_dir}")

        if use_copy:
            if output_dir is None:
                raise ValueError("output_dir must be provided when use_copy=True")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print("📦 Copying ISIC2024 images...")
            for img in train_image_dir.iterdir():
                if img.is_file():
                    shutil.copy2(img, output_dir / img.name)

            print(f"✅ ISIC2024 images copied to: {output_dir}")
            return output_dir

        print("✅ Using ISIC2024 image folder directly.")
        return train_image_dir

    if train_image_hdf5.exists():
        print(f"✅ Found ISIC2024 HDF5 file: {train_image_hdf5}")
        return train_image_hdf5

    raise FileNotFoundError("No ISIC2024 image source found.")


# --------------------------------------------------
# 4. Load metadata
# --------------------------------------------------
def load_metadata(dataset_root, dataset_type):
    """
    Load metadata depending on dataset type
    """
    dataset_root = Path(dataset_root)

    if dataset_type == "HAM10000":
        metadata_path = dataset_root / "HAM10000_metadata.csv"
    elif dataset_type == "ISIC2024":
        metadata_path = dataset_root / "train-metadata.csv"
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    df = pd.read_csv(metadata_path)
    print(f"✅ Metadata loaded for {dataset_type}: {df.shape}")
    return df


# --------------------------------------------------
# 5. Main dataset loader
# --------------------------------------------------
def load_dataset(
    dataset_root,
    output_dir="/kaggle/working/dataset_images_all",
    use_copy=False,
    inspect_input=True
):
    """
    Full dataset loading pipeline for HAM10000 or ISIC2024

    Args:
        dataset_root: dataset root path
        output_dir: output folder for copied/combined images
        use_copy: only applies to ISIC2024 folder-based images
        inspect_input: whether to print /kaggle/input contents
    """

    if inspect_input:
        inspect_kaggle_input()

    dataset_type = detect_dataset_type(dataset_root)
    print(f"\n🧠 Detected dataset type: {dataset_type}")

    if dataset_type == "HAM10000":
        images_path = prepare_ham10000_images(dataset_root, output_dir)

    elif dataset_type == "ISIC2024":
        images_path = prepare_isic2024_images(
            dataset_root,
            output_dir=output_dir,
            use_copy=use_copy
        )

    metadata = load_metadata(dataset_root, dataset_type)

    return dataset_type, images_path, metadata


# --------------------------------------------------
# Test execution
# --------------------------------------------------
if __name__ == "__main__":

    # Example 1: HAM10000
    # dataset_root = "/kaggle/input/ham10000-dataset"

    # Example 2: ISIC 2024
    dataset_root = "/kaggle/input/isic-2024-challenge"

    dataset_type, images_path, metadata = load_dataset(
        dataset_root=dataset_root,
        output_dir="/kaggle/working/dataset_images_all",
        use_copy=False,
        inspect_input=True
    )

    print("\nDataset ready!")
    print("Dataset type:", dataset_type)
    print("Images path:", images_path)
    print(metadata.head())
