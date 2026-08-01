import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import re
import csv
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================================
# SETTINGS
# ==========================================================
IMG_SIZE = 224
BATCH_SIZE = 4
CAM_BATCH_SIZE = 1
EPOCHS = 1
LR = 1e-4
NUM_WORKERS = 4
CAM_NUM_WORKERS = 0
RANDOM_STATE = 42
N_SPLITS = 5
NUM_CLASSES = 7

RUN_TRAINING = False
SAVE_FOLD_CHECKPOINTS = True
USE_AUG_SMOOTH = False
USE_EIGEN_SMOOTH = False
CAM_THRESHOLD = 0.5
CAM_TARGET_MODE = "predicted"   # "predicted" or "true"

SAVE_HEATMAPS = True
SAVE_OVERLAYS = True
SAVE_THRESHOLDED_CAM = True
SAVE_MASK_OVERLAP = True
SAVE_DEBUG_PATHS = True
SAVE_SEG_OVERLAY_CAM = True

MAX_CAM_IMAGES_PER_FOLD = None
MAX_EVAL_IMAGES_PER_FOLD = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_PIN_MEMORY = DEVICE.type == "cuda"
NON_BLOCKING = DEVICE.type == "cuda"
AMP_ENABLED = DEVICE.type == "cuda"

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("Device:", DEVICE)
print("USE_PIN_MEMORY:", USE_PIN_MEMORY)
print("AMP_ENABLED:", AMP_ENABLED)

# ==========================================================
# PATHS
# ==========================================================
ROOT_DIR = Path("/path/to/ham10000")
IMG_DIR = Path("/path/to/ham10000/HAM10000_images_all")
SEG_OVERLAY_DIR = Path("/path/to/ham10000/pipeline4/seg_overlays1")
MASK_DIR = Path("/path/to/ham10000/pipeline4/seg_masks1")
CSV_FILE = Path("/path/to/ham10000/HAM10000_metadata.csv")
BASE_MODEL_DIR = Path("/path/to/ham10000/efficientnet_b7_gradcam_results")
OUTPUT_DIR = Path("/path/to/ham10000/efficientnet_b7_gradcam_5fold_cv_memory_safe_7class")

BASE_MODEL_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

label_col = "label_idx"
img_col = "image_id"
group_col = "lesion_id"
dx_col = "dx"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

# ==========================================================
# REPRODUCIBILITY
# ==========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# ==========================================================
# HELPERS
# ==========================================================
def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def find_existing_image(base_dir, image_id, exts=(".jpg", ".jpeg", ".png")):
    for ext in exts:
        p = base_dir / f"{image_id}{ext}"
        if p.exists():
            return str(p)
    return None


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_file_index(root_dir, exts=(".png", ".jpg", ".jpeg")):
    all_files = []
    for ext in exts:
        all_files.extend(root_dir.rglob(f"*{ext}"))

    index = {}
    for p in all_files:
        stem = p.stem
        norm_stem = normalize_name(stem)

        keys = {
            norm_stem,
            normalize_name(stem.replace("_mask", "")),
            normalize_name(stem.replace("mask_", "")),
            normalize_name(stem.replace("-mask", "")),
            normalize_name(stem.replace("mask-", "")),
            normalize_name(stem.replace("_seg", "")),
            normalize_name(stem.replace("seg_", "")),
            normalize_name(stem.replace("_segmentation", "")),
            normalize_name(stem.replace("segmentation_", "")),
        }

        m = re.search(r"(isic\d+)", norm_stem)
        if m:
            keys.add(m.group(1))

        for k in keys:
            if k and k not in index:
                index[k] = str(p)

    return index, all_files


def find_indexed_path(image_id, file_index):
    norm_id = normalize_name(image_id)
    candidates = [
        norm_id,
        normalize_name(f"{image_id}_mask"),
        normalize_name(f"mask_{image_id}"),
        normalize_name(f"{image_id}-mask"),
        normalize_name(f"mask-{image_id}"),
        normalize_name(f"{image_id}_seg"),
        normalize_name(f"seg_{image_id}"),
        normalize_name(f"{image_id}_segmentation"),
        normalize_name(f"segmentation_{image_id}")
    ]

    for c in candidates:
        if c in file_index:
            return file_index[c]

    for key, value in file_index.items():
        if norm_id in key or key in norm_id:
            return value

    return None


def load_rgb_for_cam(image_path, size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_np = np.clip(img_np, 0, 1)
    return np.ascontiguousarray(img_np)


def load_binary_mask(mask_path, size=(224, 224)):
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size)
    mask = np.array(mask)
    return (mask > 127).astype(np.uint8)


def dice_score(mask1, mask2, eps=1e-8):
    inter = np.sum(mask1 * mask2)
    return (2.0 * inter) / (np.sum(mask1) + np.sum(mask2) + eps)


def iou_score(mask1, mask2, eps=1e-8):
    inter = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)
    return inter / (union + eps)


def percent_cam_inside_lesion(cam_mask, lesion_mask, eps=1e-8):
    cam_pixels = np.sum(cam_mask)
    inside = np.sum(cam_mask * lesion_mask)
    return inside / (cam_pixels + eps) if cam_pixels > 0 else np.nan


def save_confusion_matrix(cm, out_path, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_multiclass_roc_curves(y_true, y_probs, out_path, title="Multiclass ROC Curves"):
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = []

    for i, cls in enumerate(CLASS_NAMES):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        auc_val = auc(fpr, tpr)
        aucs.append(auc_val)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return float(np.mean(aucs)) if len(aucs) else np.nan


def append_rows_csv(csv_path, rows, fieldnames):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

# ==========================================================
# DATASET
# ==========================================================
class SkinDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil_img = Image.open(row["path"]).convert("RGB")
        image = self.transform(pil_img)
        label = torch.tensor(int(row[label_col]), dtype=torch.long)
        image_id = row[img_col]
        return image, label, image_id

# ==========================================================
# TRANSFORMS
# ==========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================================
# MODEL
# ==========================================================
def build_model():
    try:
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        print("Loaded EfficientNet-B7 pretrained ImageNet weights.")
    except Exception as e:
        warnings.warn(f"Could not load pretrained weights, using random init. Error: {e}")
        model = models.efficientnet_b7(weights=None)

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(DEVICE)

# ==========================================================
# LOAD DATA
# ==========================================================
print("\nIndexing masks...")
mask_index, all_mask_files = build_file_index(MASK_DIR)
print("Total discovered mask files:", len(all_mask_files))
print("Indexed mask keys:", len(mask_index))

print("\nLoading metadata...")
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=[dx_col, group_col])
df = df[df[dx_col].isin(CLASS_NAMES)].copy()
df[label_col] = df[dx_col].map(class_to_idx).astype(int)

df["path"] = df[img_col].astype(str).apply(lambda x: find_existing_image(IMG_DIR, x))
df["original_path"] = df[img_col].astype(str).apply(lambda x: find_existing_image(IMG_DIR, x))
df["seg_overlay_path"] = df[img_col].astype(str).apply(lambda x: find_existing_image(SEG_OVERLAY_DIR, x))
df["mask_path"] = df[img_col].astype(str).apply(lambda x: find_indexed_path(x, mask_index))

df = df[df["path"].notna()].reset_index(drop=True)

print("\nDataset loaded")
print("Total samples:", len(df))
print("Rows with masks:", int(df["mask_path"].notna().sum()))
print("Rows without masks:", int(df["mask_path"].isna().sum()))
print("Unique lesions:", df[group_col].nunique())
print("Class distribution:")
print(df[dx_col].value_counts().sort_index())

print("\nSample mask matches:")
sample_check = df[[img_col, "mask_path"]].head(20)
print(sample_check.to_string(index=False))

# ==========================================================
# CROSS-VALIDATION
# ==========================================================
gkf = GroupKFold(n_splits=N_SPLITS)

all_fold_metrics = []
all_prediction_files = []
all_gradcam_metric_files = []
all_debug_files = []

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, y=df[label_col], groups=df[group_col]),
    start=1
):
    print(f"\n{'='*40}")
    print(f"FOLD {fold}/{N_SPLITS}")
    print(f"{'='*40}")

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True, parents=True)

    heatmap_dir = fold_dir / "03_gradcam_heatmaps"
    overlay_dir = fold_dir / "04_gradcam_overlay"
    seg_overlay_cam_dir = fold_dir / "05_gradcam_on_seg_overlay"
    binary_cam_dir = fold_dir / "06_thresholded_cam"
    overlap_dir = fold_dir / "07_gradcam_vs_segmentation_overlap"
    plots_dir = fold_dir / "plots"
    metrics_dir = fold_dir / "metrics"
    debug_dir = fold_dir / "debug"

    for d in [heatmap_dir, overlay_dir, seg_overlay_cam_dir, binary_cam_dir, overlap_dir, plots_dir, metrics_dir, debug_dir]:
        d.mkdir(exist_ok=True, parents=True)

    fold_predictions_csv = metrics_dir / f"fold_{fold}_predictions.csv"
    fold_gradcam_csv = metrics_dir / f"fold_{fold}_gradcam_metrics.csv"
    fold_debug_csv = debug_dir / f"fold_{fold}_paths_debug.csv"

    for p in [fold_predictions_csv, fold_gradcam_csv, fold_debug_csv]:
        if p.exists():
            p.unlink()

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    if MAX_EVAL_IMAGES_PER_FOLD is not None:
        val_df = val_df.head(MAX_EVAL_IMAGES_PER_FOLD).reset_index(drop=True)

    print("Train size:", len(train_df))
    print("Val size  :", len(val_df))
    print("Train lesions:", train_df[group_col].nunique())
    print("Val lesions  :", val_df[group_col].nunique())
    print("Val masks found:", int(val_df["mask_path"].notna().sum()))

    train_dataset = SkinDataset(train_df, train_transform)
    val_dataset = SkinDataset(val_df, test_transform)

    class_counts = train_df[label_col].value_counts().sort_index()
    weights = train_df[label_col].map(lambda x: 1.0 / class_counts[x]).values
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    ce_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=USE_PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_PIN_MEMORY
    )

    cam_loader = DataLoader(
        val_dataset,
        batch_size=CAM_BATCH_SIZE,
        shuffle=False,
        num_workers=CAM_NUM_WORKERS,
        pin_memory=USE_PIN_MEMORY
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=ce_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    fold_ckpt = BASE_MODEL_DIR / f"best_efficientnet_b7_fold_{fold}.pth"

    if RUN_TRAINING:
        best_bal_acc = -np.inf
        print("\nTraining model...")

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels, _ in train_loader:
                images = images.to(DEVICE, non_blocking=NON_BLOCKING)
                labels = labels.to(DEVICE, non_blocking=NON_BLOCKING)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                del images, labels, outputs, loss

            avg_loss = running_loss / max(1, len(train_loader))

            model.eval()
            y_true_epoch = []
            y_pred_epoch = []

            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(DEVICE, non_blocking=NON_BLOCKING)

                    with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                        outputs = model(images)

                    preds = torch.argmax(outputs, dim=1)
                    y_true_epoch.extend(labels.numpy().tolist())
                    y_pred_epoch.extend(preds.cpu().numpy().tolist())
                    del images, outputs, preds

            bal_acc_epoch = balanced_accuracy_score(y_true_epoch, y_pred_epoch)
            scheduler.step(bal_acc_epoch)

            print(f"Fold {fold} | Epoch {epoch+1:02d}/{EPOCHS} | Loss={avg_loss:.4f} | BalAcc={bal_acc_epoch:.4f}")

            if SAVE_FOLD_CHECKPOINTS and bal_acc_epoch > best_bal_acc:
                best_bal_acc = bal_acc_epoch
                torch.save(model.state_dict(), fold_ckpt)
                print(f"Saved best fold checkpoint: {fold_ckpt}")

            cleanup_cuda()

    if fold_ckpt.exists():
        print(f"Loading fold checkpoint: {fold_ckpt}")
        checkpoint = torch.load(fold_ckpt, map_location=DEVICE)
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                raise ValueError(f"Unsupported checkpoint format for fold {fold}.")
    else:
        print(f"WARNING: No fold checkpoint found for fold {fold}. Using current model weights.")

    model.eval()

    # ======================================================
    # EVALUATION
    # ======================================================
    print("Running validation evaluation...")

    y_true = []
    y_pred = []
    y_probs_list = []
    image_ids = []

    with torch.no_grad():
        for images, labels, ids in val_loader:
            images = images.to(DEVICE, non_blocking=NON_BLOCKING)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_probs_list.append(probs.cpu().numpy())
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            image_ids.extend(ids)

            del images, outputs, probs, preds

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.concatenate(y_probs_list, axis=0) if len(y_probs_list) > 0 else np.empty((0, NUM_CLASSES))

    fold_pred_df = pd.DataFrame({
        "fold": fold,
        "image_id": image_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_class_name": [idx_to_class[int(x)] for x in y_true],
        "predicted_class_name": [idx_to_class[int(x)] for x in y_pred],
    })
    for i, cls in enumerate(CLASS_NAMES):
        fold_pred_df[f"prob_{cls}"] = y_probs[:, i]
    fold_pred_df.to_csv(fold_predictions_csv, index=False)
    all_prediction_files.append(fold_predictions_csv)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(metrics_dir / f"fold_{fold}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(cm, plots_dir / f"fold_{fold}_confusion_matrix.png", title=f"Fold {fold} Confusion Matrix")
    mean_ovr_auc_plot = save_multiclass_roc_curves(y_true, y_probs, plots_dir / f"fold_{fold}_roc_auc.png", title=f"Fold {fold} ROC Curve")

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recv_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        roc_auc_ovr_macro = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_ovr_macro = np.nan

    try:
        roc_auc_ovo_macro = roc_auc_score(y_true, y_probs, multi_class="ovo", average="macro")
    except Exception:
        roc_auc_ovo_macro = np.nan

    fold_metric_row = {
        "fold": fold,
        "n_val": len(val_df),
        "n_val_lesions": val_df[group_col].nunique(),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": recv_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr_macro": roc_auc_ovr_macro,
        "roc_auc_ovo_macro": roc_auc_ovo_macro,
        "mean_ovr_auc_plot": mean_ovr_auc_plot,
    }
    all_fold_metrics.append(fold_metric_row)

    print(
        f"Fold {fold} | Acc={acc:.4f} | BalAcc={bal_acc:.4f} | "
        f"MacroF1={f1_macro:.4f} | WeightedF1={f1_weighted:.4f} | "
        f"OVR-AUC={roc_auc_ovr_macro:.4f} | OVO-AUC={roc_auc_ovo_macro:.4f}"
    )

    # ======================================================
    # LOOKUPS
    # ======================================================
    pred_lookup_cols = ["true_label", "predicted_label", "true_class_name", "predicted_class_name"] + [f"prob_{cls}" for cls in CLASS_NAMES]
    pred_lookup = fold_pred_df.set_index("image_id")[pred_lookup_cols].to_dict("index")
    meta_lookup = val_df.set_index("image_id")[["path", "original_path", "seg_overlay_path", "mask_path"]].to_dict("index")

    del fold_pred_df
    cleanup_cuda()

    # ======================================================
    # GRAD-CAM
    # ======================================================
    print("Generating Grad-CAM for validation images...")

    target_layers = [model.features[-1]]
    processed = 0
    masks_found_count = 0
    metrics_computed_count = 0

    gradcam_fieldnames = [
        "fold", "image_id", "true_label", "predicted_label",
        "true_class_name", "predicted_class_name",
        "target_class_for_cam", "target_class_name_for_cam",
        *[f"prob_{cls}" for cls in CLASS_NAMES],
        "cam_threshold", "iou", "dice",
        "percent_cam_inside_lesion", "pixelwise_correlation"
    ]

    debug_fieldnames = [
        "fold", "image_id", "image_path", "original_path",
        "seg_overlay_path", "mask_path", "mask_found"
    ]

    for p in model.parameters():
        p.requires_grad_(True)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        model.eval()

        for images, labels, ids in cam_loader:
            if MAX_CAM_IMAGES_PER_FOLD is not None and processed >= MAX_CAM_IMAGES_PER_FOLD:
                break

            image_id = ids[0]
            images = images.to(DEVICE, non_blocking=NON_BLOCKING)

            info = pred_lookup[image_id]
            true_label = int(info["true_label"])
            predicted_label = int(info["predicted_label"])
            true_class_name = info["true_class_name"]
            predicted_class_name = info["predicted_class_name"]
            prob_dict = {f"prob_{cls}": float(info[f"prob_{cls}"]) for cls in CLASS_NAMES}

            target_class = predicted_label if CAM_TARGET_MODE == "predicted" else true_label
            target_class_name = idx_to_class[target_class]

            grayscale_cam = cam(
                input_tensor=images,
                targets=[ClassifierOutputTarget(target_class)],
                aug_smooth=USE_AUG_SMOOTH,
                eigen_smooth=USE_EIGEN_SMOOTH
            )[0]

            row_paths = meta_lookup.get(image_id, {})
            image_path = row_paths.get("path", None)
            original_path = row_paths.get("original_path", None)
            seg_overlay_path = row_paths.get("seg_overlay_path", None)
            mask_path = row_paths.get("mask_path", None)

            iou_val = np.nan
            dice_val = np.nan
            inside_ratio = np.nan
            pixel_corr = np.nan

            lesion_mask = None
            if mask_path is not None and os.path.exists(mask_path):
                masks_found_count += 1
                lesion_mask = load_binary_mask(mask_path, size=(IMG_SIZE, IMG_SIZE))

            if SAVE_HEATMAPS:
                heatmap_uint8 = np.uint8(255 * grayscale_cam)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                cv2.imwrite(str(heatmap_dir / f"{image_id}_heatmap.jpg"), heatmap_color)

            base_image_path = original_path if original_path is not None and os.path.exists(original_path) else image_path
            if SAVE_OVERLAYS and base_image_path is not None and os.path.exists(base_image_path):
                base_np = load_rgb_for_cam(base_image_path, size=(IMG_SIZE, IMG_SIZE))
                gradcam_overlay = show_cam_on_image(base_np, grayscale_cam, use_rgb=True)
                cv2.imwrite(str(overlay_dir / f"{image_id}_gradcam_overlay.jpg"), cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
                del base_np, gradcam_overlay

            if SAVE_SEG_OVERLAY_CAM and seg_overlay_path is not None and os.path.exists(seg_overlay_path):
                seg_np = load_rgb_for_cam(seg_overlay_path, size=(IMG_SIZE, IMG_SIZE))
                seg_cam_overlay = show_cam_on_image(seg_np, grayscale_cam, use_rgb=True)
                cv2.imwrite(str(seg_overlay_cam_dir / f"{image_id}_gradcam_on_seg_overlay.jpg"), cv2.cvtColor(seg_cam_overlay, cv2.COLOR_RGB2BGR))
                del seg_np, seg_cam_overlay

            cam_binary = (grayscale_cam >= CAM_THRESHOLD).astype(np.uint8)

            if SAVE_THRESHOLDED_CAM:
                cv2.imwrite(str(binary_cam_dir / f"{image_id}_thresholded_cam.png"), cam_binary * 255)

            if lesion_mask is not None and lesion_mask.sum() > 0:
                iou_val = iou_score(cam_binary, lesion_mask)
                dice_val = dice_score(cam_binary, lesion_mask)
                inside_ratio = percent_cam_inside_lesion(cam_binary, lesion_mask)

                if np.std(grayscale_cam.flatten()) > 0 and np.std(lesion_mask.flatten()) > 0:
                    pixel_corr = np.corrcoef(grayscale_cam.flatten(), lesion_mask.flatten())[0, 1]

                metrics_computed_count += 1

                if SAVE_MASK_OVERLAP:
                    overlap_vis = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    overlap_vis[..., 1] = lesion_mask * 255
                    overlap_vis[..., 2] = cam_binary * 255
                    overlap_vis[(lesion_mask == 1) & (cam_binary == 1)] = [255, 255, 0]
                    cv2.imwrite(str(overlap_dir / f"{image_id}_gradcam_vs_segmentation_overlap.png"), cv2.cvtColor(overlap_vis, cv2.COLOR_RGB2BGR))
                    del overlap_vis

            gradcam_row = {
                "fold": fold,
                "image_id": image_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_class_name": true_class_name,
                "predicted_class_name": predicted_class_name,
                "target_class_for_cam": target_class,
                "target_class_name_for_cam": target_class_name,
                **prob_dict,
                "cam_threshold": CAM_THRESHOLD,
                "iou": iou_val,
                "dice": dice_val,
                "percent_cam_inside_lesion": inside_ratio,
                "pixelwise_correlation": pixel_corr
            }
            append_rows_csv(fold_gradcam_csv, [gradcam_row], gradcam_fieldnames)

            if SAVE_DEBUG_PATHS:
                append_rows_csv(fold_debug_csv, [{
                    "fold": fold,
                    "image_id": image_id,
                    "image_path": image_path,
                    "original_path": original_path,
                    "seg_overlay_path": seg_overlay_path,
                    "mask_path": mask_path,
                    "mask_found": int(mask_path is not None and os.path.exists(mask_path))
                }], debug_fieldnames)

            processed += 1
            if processed % 50 == 0:
                print(f"Fold {fold}: processed Grad-CAM for {processed}/{len(val_dataset)} images")

            del images, labels, ids, grayscale_cam, cam_binary, lesion_mask
            cleanup_cuda()

    all_gradcam_metric_files.append(fold_gradcam_csv)
    if SAVE_DEBUG_PATHS:
        all_debug_files.append(fold_debug_csv)

    print(f"Fold {fold}: Grad-CAM done for {processed} images")
    print(f"Fold {fold}: masks found = {masks_found_count}")
    print(f"Fold {fold}: metrics computed = {metrics_computed_count}")

    del model, train_loader, val_loader, cam_loader, train_dataset, val_dataset
    del train_df, val_df, pred_lookup, meta_lookup, weights, sampler
    cleanup_cuda()

# ==========================================================
# AGGREGATION ACROSS FOLDS
# ==========================================================
print("\nAggregating results across folds...")

all_fold_metrics_df = pd.DataFrame(all_fold_metrics)
all_fold_metrics_df.to_csv(OUTPUT_DIR / "cv_fold_classification_metrics.csv", index=False)

pred_frames = [pd.read_csv(p) for p in all_prediction_files]
all_predictions_df = pd.concat(pred_frames, ignore_index=True)
all_predictions_df.to_csv(OUTPUT_DIR / "cv_all_predictions.csv", index=False)
del pred_frames, all_predictions_df

cam_frames = [pd.read_csv(p) for p in all_gradcam_metric_files if Path(p).exists()]
if len(cam_frames) > 0:
    all_gradcam_df = pd.concat(cam_frames, ignore_index=True)
else:
    all_gradcam_df = pd.DataFrame(columns=[
        "fold", "image_id", "true_label", "predicted_label",
        "true_class_name", "predicted_class_name",
        "target_class_for_cam", "target_class_name_for_cam",
        *[f"prob_{cls}" for cls in CLASS_NAMES],
        "cam_threshold", "iou", "dice",
        "percent_cam_inside_lesion", "pixelwise_correlation"
    ])
all_gradcam_df.to_csv(OUTPUT_DIR / "cv_all_gradcam_metrics.csv", index=False)
del cam_frames

if len(all_debug_files) > 0:
    dbg_frames = [pd.read_csv(p) for p in all_debug_files if Path(p).exists()]
    if len(dbg_frames) > 0:
        all_debug_df = pd.concat(dbg_frames, ignore_index=True)
        all_debug_df.to_csv(OUTPUT_DIR / "cv_all_paths_debug.csv", index=False)
        del all_debug_df, dbg_frames

classification_summary = all_fold_metrics_df.drop(columns=["fold"]).agg(["mean", "std"]).transpose().reset_index()
classification_summary.columns = ["metric", "mean", "std"]
classification_summary.to_csv(OUTPUT_DIR / "cv_classification_summary_mean_std.csv", index=False)

gradcam_metric_cols = ["iou", "dice", "percent_cam_inside_lesion", "pixelwise_correlation"]
gradcam_summary_rows = []

for col in gradcam_metric_cols:
    valid = pd.to_numeric(all_gradcam_df[col], errors="coerce").dropna()
    gradcam_summary_rows.append({
        "metric": col,
        "count": int(valid.shape[0]),
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "std": float(valid.std()) if len(valid) else np.nan,
        "median": float(valid.median()) if len(valid) else np.nan,
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan
    })

gradcam_summary_df = pd.DataFrame(gradcam_summary_rows)
gradcam_summary_df.to_csv(OUTPUT_DIR / "cv_gradcam_summary_mean_std.csv", index=False)

gradcam_by_fold = (
    all_gradcam_df.groupby("fold")[gradcam_metric_cols]
    .agg(["count", "mean", "std", "median"])
)
gradcam_by_fold.columns = ["_".join(col) for col in gradcam_by_fold.columns]
gradcam_by_fold = gradcam_by_fold.reset_index()
gradcam_by_fold.to_csv(OUTPUT_DIR / "cv_gradcam_metrics_by_fold.csv", index=False)

print("\n==============================")
print("5-Fold CV finished successfully")
print("==============================")
print("Saved:")
print("-", OUTPUT_DIR / "cv_fold_classification_metrics.csv")
print("-", OUTPUT_DIR / "cv_all_predictions.csv")
print("-", OUTPUT_DIR / "cv_all_gradcam_metrics.csv")
print("-", OUTPUT_DIR / "cv_classification_summary_mean_std.csv")
print("-", OUTPUT_DIR / "cv_gradcam_summary_mean_std.csv")
print("-", OUTPUT_DIR / "cv_gradcam_metrics_by_fold.csv")
if len(all_debug_files) > 0:
    print("-", OUTPUT_DIR / "cv_all_paths_debug.csv")
print("==============================")


