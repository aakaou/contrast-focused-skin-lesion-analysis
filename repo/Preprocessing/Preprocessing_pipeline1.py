from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_and_save_images(images_path, output_folder, img_size=224):
    images_path = Path(images_path)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(images_path.glob("*.jpg")) +
        list(images_path.glob("*.png")) +
        list(images_path.glob("*.jpeg"))
    )

    print("📂 Input folder:", images_path)
    print("📂 Output folder:", output_folder)
    print("🔍 Found images:", len(image_paths))
    print("🚀 Preprocessing...\n")

    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path)

            # Fix orientation
            img = ImageOps.exif_transpose(img)

            # RGB
            img = img.convert("RGB")

            # Resize
            img = img.resize((img_size, img_size), Image.BILINEAR)

            # Normalize
            arr = np.array(img).astype("float32") / 255.0
            mean = arr.mean(axis=(0, 1), keepdims=True)
            std = arr.std(axis=(0, 1), keepdims=True) + 1e-6
            arr = (arr - mean) / std

            # Back to image range
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
            arr = (arr * 255).clip(0, 255).astype("uint8")

            out_img = Image.fromarray(arr)

            # Save with same filename
            out_img.save(output_folder / img_path.name)

        except Exception as e:
            print(f"❌ Error: {img_path} -> {e}")

    print("\n✅ Preprocessing done!")
    return output_folder


def show_first_images(folder_path, n=5):
    folder = Path(folder_path)
    image_files = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpeg"))
    )

    print(f"\n🖼️ Showing first {n} processed images:\n")

    plt.figure(figsize=(15, 5))

    for i, img_path in enumerate(image_files[:n]):
        img = Image.open(img_path)

        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Paths for the two datasets
# --------------------------------------------------

# ISIC 2024 image folder
isic2024_images_path = Path("/path/to/train-image/image")

# HAM10000 image folder
# Replace this with your real HAM10000 folder path
ham10000_images_path = Path("/path/to/HAM10000_images_all")

# Output folders
isic2024_output = Path("/path/to/isic2024_processed")
ham10000_output = Path("/path/to/ham10000_processed")


# --------------------------------------------------
# Run preprocessing for both datasets
# --------------------------------------------------

print("\n================ ISIC 2024 ================\n")
processed_isic2024 = preprocess_and_save_images(
    images_path=isic2024_images_path,
    output_folder=isic2024_output,
    img_size=224
)

print("\n================ HAM10000 ================\n")
processed_ham10000 = preprocess_and_save_images(
    images_path=ham10000_images_path,
    output_folder=ham10000_output,
    img_size=224
)


# --------------------------------------------------
# Show sample images
# --------------------------------------------------

show_first_images(processed_isic2024, n=5)
show_first_images(processed_ham10000, n=5)

print("\n✅ All preprocessing finished!")
print("ISIC 2024 processed folder:", processed_isic2024)
print("HAM10000 processed folder:", processed_ham10000)
