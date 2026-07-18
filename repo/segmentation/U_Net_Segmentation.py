import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# ======================================================
# SETTINGS
# ======================================================

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 1
SEED = 42
VAL_SPLIT = 0.2

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# DICE METRIC
# ======================================================

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    inter = tf.keras.backend.sum(y_true_f * y_pred_f)

    return (2.0 * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) +
        tf.keras.backend.sum(y_pred_f) + smooth
    )

# ======================================================
# U-NET MODEL
# ======================================================

def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same")(c3)

    u4 = layers.UpSampling2D()(c3)
    u4 = layers.Concatenate()([u4, c2])
    c4 = layers.Conv2D(64, 3, activation="relu", padding="same")(u4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1])
    c5 = layers.Conv2D(32, 3, activation="relu", padding="same")(u5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", dice_coef]
    )

    return model

# ======================================================
# MASK GENERATION
# ======================================================

def generate_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# ======================================================
# DATA GENERATOR
# ======================================================

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=8):
        self.image_paths = image_paths
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]

        X = np.zeros(
            (len(batch_paths), IMG_SIZE, IMG_SIZE, 3),
            dtype=np.float32
        )
        Y = np.zeros(
            (len(batch_paths), IMG_SIZE, IMG_SIZE, 1),
            dtype=np.float32
        )

        for i, img_path in enumerate(batch_paths):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            mask = generate_mask(img)

            X[i] = img / 255.0
            Y[i] = mask[..., None] / 255.0

        return X, Y

# ======================================================
# FUSION FUNCTION
# ======================================================

def fuse(preprocessed_img, sonar_img, mask):
    mask = np.clip(mask, 0, 1)
    mask_3 = np.repeat(mask[:, :, None], 3, axis=2)

    fused = (
        mask_3 * preprocessed_img +
        (1 - mask_3) * sonar_img
    )

    return fused.astype(np.uint8)

# ======================================================
# IMAGE COLLECTION
# ======================================================

def collect_images(folder):
    folder = Path(folder)
    return sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png"))
    )

# ======================================================
# SINGLE EXPERIMENT RUN
# ======================================================

def run_unet_fusion_experiment(
    preprocessed_folder,
    sonar_folder,
    output_masks,
    output_final,
    model_dir,
    img_size=256,
    batch_size=8,
    epochs=1,
    val_split=0.2,
    seed=42
):
    preprocessed_folder = Path(preprocessed_folder)
    sonar_folder = Path(sonar_folder)
    output_masks = Path(output_masks)
    output_final = Path(output_final)
    model_dir = Path(model_dir)

    output_masks.mkdir(parents=True, exist_ok=True)
    output_final.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(preprocessed_folder)

    print("\n====================================================")
    print("Preprocessed folder:", preprocessed_folder)
    print("Sonar folder:", sonar_folder)
    print("Total images:", len(images))
    print("====================================================\n")

    if len(images) == 0:
        print("❌ No images found, skipping.")
        return

    train_paths, val_paths = train_test_split(
        images,
        test_size=val_split,
        random_state=seed
    )

    train_gen = DataGenerator(train_paths, batch_size=batch_size)
    val_gen = DataGenerator(val_paths, batch_size=batch_size)

    model = build_unet()

    best_model_path = model_dir / "best_unet.h5"
    final_model_path = model_dir / "final_unet.h5"

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(str(best_model_path), save_best_only=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(final_model_path))

    print("\nRunning inference...")

    for i, img_path in enumerate(images):
        pre_img_path = preprocessed_folder / img_path.name
        sonar_img_path = sonar_folder / img_path.name

        if not pre_img_path.exists():
            print(f"❌ Missing preprocessed image: {pre_img_path}")
            continue

        if not sonar_img_path.exists():
            print(f"❌ Missing sonar image: {sonar_img_path}")
            continue

        pre_img = cv2.imread(str(pre_img_path))
        sonar_img = cv2.imread(str(sonar_img_path))

        if pre_img is None or sonar_img is None:
            print(f"❌ Could not read image pair: {img_path.name}")
            continue

        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        sonar_img = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2RGB)

        h, w = pre_img.shape[:2]

        inp = cv2.resize(pre_img, (img_size, img_size)).astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)

        pred = model.predict(inp, verbose=0)[0, :, :, 0]

        mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0, 1)

        # Keep your original inversion logic
        mask = 1 - mask

        cv2.imwrite(
            str(output_masks / f"{img_path.stem}.png"),
            (mask * 255).astype(np.uint8)
        )

        fused = fuse(pre_img, sonar_img, mask)

        cv2.imwrite(
            str(output_final / img_path.name),
            cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"Processed {i+1}/{len(images)}")

    print("\nDONE ✔ ALL IMAGES PROCESSED")

# ======================================================
# DEFAULT CONFIGS
# ======================================================

def build_default_experiments():
    experiments = [
        # --------------------------------------------------
        # ISIC 2024
        # --------------------------------------------------
        {
            "name": "isic2024_pipeline1",
            "preprocessed_folder": "/path/to/pipeline1/processed_images",
            "sonar_folder": "/path/to/pipeline1/sonar_output/sonar_only",
            "output_masks": "/path/to/pipeline1/unet_masks",
            "output_final": "/path/to/pipeline1/unet_overlays_up",
            "model_dir": "/path/to/pipeline1/unet_models"
        },
        {
            "name": "isic2024_pipeline2",
            "preprocessed_folder": "/path/to/pipeline2/processed_images",
            "sonar_folder": "/path/to/pipeline2/sonar_output/sonar_only",
            "output_masks": "/path/to/pipeline2/unet_masks",
            "output_final": "/path/to/pipeline2/unet_overlays_up",
            "model_dir": "/path/to/pipeline2/unet_models"
        },
        {
            "name": "isic2024_pipeline3",
            "preprocessed_folder": "/path/to/pipeline3/processed_images",
            "sonar_folder": "/path/to/pipeline3/sonar_output/sonar_only",
            "output_masks": "/path/to/pipeline3/unet_masks",
            "output_final": "/path/to/pipeline3/unet_overlays_up",
            "model_dir": "/path/to/pipeline3/unet_models"
        },
        {
            "name": "isic2024_pipeline4",
            "preprocessed_folder": "/path/to/pipeline4/processed_images",
            "sonar_folder": "/path/to/pipeline4/sonar_output/sonar_only",
            "output_masks": "/path/to/pipeline4/unet_masks",
            "output_final": "/path/to/pipeline4/unet_overlays_up",
            "model_dir": "/path/to/pipeline4/unet_models"
        },

        # --------------------------------------------------
        # HAM10000
        # --------------------------------------------------
        {
            "name": "ham10000_pipeline1",
            "preprocessed_folder": "/path/to/ham10000/pipeline1/processed_images",
            "sonar_folder": "/path/to/ham10000/pipeline1/sonar_output/sonar_only",
            "output_masks": "/path/to/ham10000/pipeline1/unet_masks",
            "output_final": "/path/to/ham10000/pipeline1/unet_overlays_up",
            "model_dir": "/path/to/ham10000/pipeline1/unet_models"
        },
        {
            "name": "ham10000_pipeline2",
            "preprocessed_folder": "/path/to/ham10000/pipeline2/processed_images",
            "sonar_folder": "/path/to/ham10000/pipeline2/sonar_output/sonar_only",
            "output_masks": "/path/to/ham10000/pipeline2/unet_masks",
            "output_final": "/path/to/ham10000/pipeline2/unet_overlays_up",
            "model_dir": "/path/to/ham10000/pipeline2/unet_models"
        },
        {
            "name": "ham10000_pipeline3",
            "preprocessed_folder": "/path/to/ham10000/pipeline3/processed_images",
            "sonar_folder": "/path/to/ham10000/pipeline3/sonar_output/sonar_only",
            "output_masks": "/path/to/ham10000/pipeline3/unet_masks",
            "output_final": "/path/to/ham10000/pipeline3/unet_overlays_up",
            "model_dir": "/path/to/ham10000/pipeline3/unet_models"
        },
        {
            "name": "ham10000_pipeline4",
            "preprocessed_folder": "/path/to/ham10000/pipeline4/processed_images",
            "sonar_folder": "/path/to/ham10000/pipeline4/sonar_output/sonar_only",
            "output_masks": "/path/to/ham10000/pipeline4/unet_masks",
            "output_final": "/path/to/ham10000/pipeline4/unet_overlays_up",
            "model_dir": "/path/to/ham10000/pipeline4/unet_models"
        }
    ]

    return experiments

# ======================================================
# RUN MULTIPLE EXPERIMENTS
# ======================================================

def run_all_experiments(experiments):
    for exp in experiments:
        print("\n" + "=" * 70)
        print("RUNNING:", exp["name"])
        print("=" * 70)

        run_unet_fusion_experiment(
            preprocessed_folder=exp["preprocessed_folder"],
            sonar_folder=exp["sonar_folder"],
            output_masks=exp["output_masks"],
            output_final=exp["output_final"],
            model_dir=exp["model_dir"],
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            val_split=VAL_SPLIT,
            seed=SEED
        )

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    experiments = build_default_experiments()
    run_all_experiments(experiments)
