"""
Dataset loader for HAM10000 (Kaggle)

Responsibilities:
1. Inspect Kaggle input directory
2. Load metadata
3. Combine image folders into one directory
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
# 2. Combine image folders
# --------------------------------------------------
def combine_images(dataset_root, output_dir):
    """
    Combine images from part_1 and part_2 into one folder
    """

    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)

    part1 = dataset_root / "HAM10000_images_part_1"
    part2 = dataset_root / "HAM10000_images_part_2"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("📂 Combining HAM10000 images...")

    for folder in [part1, part2]:
        for img in folder.glob("*.jpg"):
            shutil.copy2(img, output_dir / img.name)

    print(f"✅ All images copied to: {output_dir}")

    return output_dir


# --------------------------------------------------
# 3. Load metadata
# --------------------------------------------------
def load_metadata(dataset_root):
    """
    Load HAM10000 metadata file
    """

    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "HAM10000_metadata.csv"

    df = pd.read_csv(metadata_path)

    print(f"✅ Metadata loaded: {df.shape}")

    return df


# --------------------------------------------------
# 4. Main dataset loader
# --------------------------------------------------
def load_dataset(
    dataset_root="/kaggle/input/ham10000-dataset",
    output_dir="/kaggle/working/HAM10000_images_all"
):
    """
    Full dataset loading pipeline
    """

    # Step 1: show Kaggle input files
    inspect_kaggle_input()

    # Step 2: combine images
    images_path = combine_images(dataset_root, output_dir)

    # Step 3: load metadata
    metadata = load_metadata(dataset_root)

    return images_path, metadata


# --------------------------------------------------
# Test execution
# --------------------------------------------------
if __name__ == "__main__":

    images_path, metadata = load_dataset()

    print("\nDataset ready!")
    print("Images folder:", images_path)
    print(metadata.head())
