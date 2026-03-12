import os
import cv2
import logging
import numpy as np
import pandas as pd
import imgaug.augmenters as iaa
import tensorflow as tf

from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU

# Setup logging for info messages
logging.basicConfig(level=logging.INFO)


class KaggleImageProcessor:
    """
    Processor for skin lesion images with:
    - Metadata handling
    - Preprocessing (hair removal, inpainting, resize, CLAHE, normalization)
    - Segmentation using U-Net
    - Fusion of lesion & background (Sonar style)
    """

    def __init__(self, kaggle_json_path, destination_dir, competition_name,
                 data_dir, output_dir):
        # Paths and model placeholders
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model = None
        self.metadata = None

    # --------------------------------------------------
    # METADATA HANDLING
    # --------------------------------------------------
    def load_metadata(self, csv_path=None):
        """Load metadata CSV and impute missing values."""
        if csv_path is None:
            return
        meta = pd.read_csv(csv_path)
        self.metadata = self.impute_missing_values(meta)

    def impute_missing_values(self, meta):
        """Fill missing age values with median."""
        median_age = meta["age"].median()
        meta["age"] = meta["age"].fillna(median_age)
        logging.info("✅ Missing age values fixed")
        return meta

    # --------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------
    def dullrazor(self, img):
        """
        Remove hairs using DullRazor algorithm:
        1. Convert to grayscale
        2. Morphological blackhat to find hair
        3. Threshold to create mask
        4. Inpaint hair regions if significant
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # If hair occupies <1% of image, skip inpainting
        hair_ratio = np.sum(mask) / mask.size * 100
        if hair_ratio < 1:
            return img

        # Inpaint detected hair regions
        return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

    def preprocess_image(self, image, augment=False):
        """
        Full preprocessing pipeline:
        1. DullRazor hair removal + inpainting
        2. Resize to 256x256
        3. CLAHE enhancement for local contrast
        4. Optional data augmentation
        5. Intensity normalization [0,1]
        """
        # Convert BGR (OpenCV) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Remove hair
        image = self.dullrazor(image)

        # Resize to 256x256 for segmentation model
        image = cv2.resize(image, (256, 256))

        # CLAHE enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # Convert to LAB
        l, a, b = cv2.split(lab)                      # Split channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)                            # Apply CLAHE on L channel
        lab = cv2.merge((cl, a, b))                   # Merge channels
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convert back to RGB

        # Optional augmentation
        if augment:
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),                       # Horizontal flip 50%
                iaa.Multiply((0.9, 1.1)),              # Random brightness
                iaa.LinearContrast((0.9, 1.1))         # Random contrast
            ])
            image = seq(image=image)

        # Normalize intensity to [0,1]
        return image / 255.0

    def process_single_image(self, filename, augment):
        """Process a single image and save preprocessed version."""
        path = os.path.join(self.data_dir, filename)
        img = cv2.imread(path)
        if img is None:
            return

        processed = self.preprocess_image(img, augment)
        save_path = os.path.join(self.output_dir, filename)

        # Save processed image
        cv2.imwrite(
            save_path,
            cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

    def process_images_in_directory(self, augment=True):
        """Process all images in input directory in parallel."""
        os.makedirs(self.output_dir, exist_ok=True)
        files = [f for f in os.listdir(self.data_dir)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Use threads to speed up preprocessing
        with ThreadPoolExecutor() as executor:
            for f in files:
                executor.submit(self.process_single_image, f, augment)

    # --------------------------------------------------
    # SEGMENTATION MODEL (U-NET)
    # --------------------------------------------------
    def unet_model(self, input_size=(256, 256, 3)):
        """Define U-Net architecture for lesion segmentation."""
        inputs = Input(input_size)

        # Encoder
        c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling2D()(c1)

        c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling2D()(c2)

        c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
        p3 = MaxPooling2D()(c3)

        c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

        # Decoder
        u5 = UpSampling2D()(c4)
        u5 = concatenate([u5, c3])
        c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)

        u6 = UpSampling2D()(c5)
        u6 = concatenate([u6, c2])
        c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)

        u7 = UpSampling2D()(c6)
        u7 = concatenate([u7, c1])
        c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)

        # Output mask
        outputs = Conv2D(1, 1, activation='sigmoid')(c7)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(1e-4),
            loss=BinaryCrossentropy(),
            metrics=[MeanIoU(num_classes=2)]
        )
        return model

    # --------------------------------------------------
    # SONAR + FUSION (LESION = ORIGINAL, BACKGROUND = SONAR)
    # --------------------------------------------------
    def apply_sonar(self, image):
        """Apply sonar colormap on grayscale background."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)

    def fuse_lesion_background(self, original, sonar, mask):
        """Fuse lesion from original image onto sonar background."""
        fused = sonar.copy()
        fused[mask == 1] = original[mask == 1]
        return fused

    # --------------------------------------------------
    # SEGMENTATION PIPELINE
    # --------------------------------------------------
    def process_and_save_image(self, image_path, masks_folder, overlays_folder):
        """Process, segment, and save mask + overlay for one image."""
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        sonar = self.apply_sonar(original)

        # Resize & normalize for U-Net
        img = cv2.resize(sonar, (256, 256)) / 255.0
        img = np.expand_dims(img, axis=0)

        # Load U-Net model if not loaded
        if self.model is None:
            self.model = self.unet_model()
            self.model.load_weights("unet_model_st.h5")

        # Predict lesion mask
        pred = self.model.predict(img, verbose=0)[0, :, :, 0]
        mask = (pred > 0.5).astype(np.uint8)

        # Resize mask to original image size
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        # Fuse lesion on sonar background
        fused = self.fuse_lesion_background(original, sonar, mask)

        # Save mask and overlay
        name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(masks_folder, exist_ok=True)
        os.makedirs(overlays_folder, exist_ok=True)
        cv2.imwrite(f"{masks_folder}/{name}.png", mask * 255)
        cv2.imwrite(f"{overlays_folder}/{name}.png",
                    cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))

    def process_images_in_parallel(self, input_folder, masks_folder, overlays_folder):
        """Process all images in input folder in parallel with ThreadPoolExecutor."""
        images = [os.path.join(input_folder, f)
                  for f in os.listdir(input_folder)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        with ThreadPoolExecutor(max_workers=6) as executor:
            for img in images:
                executor.submit(
                    self.process_and_save_image,
                    img, masks_folder, overlays_folder
                )


# ==================================================
# MAIN EXECUTION
# ==================================================
if __name__ == "__main__":
    # Input raw images directory
    data_dir = r"/aakaou/ham10000/HAM10000_images_all"
    # Preprocessed images output directory
    output_dir = r"/aakaou/ham10000/pipeline3/processed-images1"

    # Segmentation outputs
    masks_folder = r"/aakaou/ham10000/pipeline3/seg_masks1"
    overlays_folder = r"/aakaou/ham10000/pipeline3/seg_overlays1"

    kaggle_json_path = None
    destination_dir = None
    competition_name = None

    # Initialize processor
    processor = KaggleImageProcessor(
        kaggle_json_path,
        destination_dir,
        competition_name,
        data_dir,
        output_dir
    )

    logging.info("🚀 Starting image preprocessing...")
    processor.process_images_in_directory(augment=True)

    logging.info("🧠 Starting segmentation and fusion (lesion=original, background=sonar)...")
    processor.process_images_in_parallel(
        input_folder=output_dir,
        masks_folder=masks_folder,
        overlays_folder=overlays_folder
    )

    logging.info("✅ Image processing & segmentation completed successfully.")
