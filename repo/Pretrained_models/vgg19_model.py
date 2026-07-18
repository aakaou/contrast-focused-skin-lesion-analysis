# ==========================================================
# FULL VGG19 CLINICAL PIPELINE
# FOR TWO DATASET TYPES:
#   1) ISIC 2024-style metadata
#   2) HAM10000-style metadata
# ==========================================================

import os
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from torchvision import (
    models,
    transforms
)

from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# SETTINGS
# ==========================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# ==========================================================
# DATASET CONFIGS
# ==========================================================

DATASET_CONFIGS = [
    {
        "name": "ISIC2024_pipeline_each_time",
        "dataset_type": "isic2024",
        "img_dir": "/path/to/pipeline/unet_overlays_up",
        "csv_file": "/path/to/isic2024/train_metadata_clean_full.csv",
        "img_col": "isic_id",
        "label_col": "target",
        "patient_col": "patient_id",
        "file_ext": ".jpg",
        "output_dir": "/path/to/vgg19_results"
    },
    {
        "name": "HAM10000_pipeline_each_time",
        "dataset_type": "ham10000",
        "img_dir": "/path/to/ham10000/pipeline4/unet_overlays_up",
        "csv_file": "/path/to/ham10000/HAM10000_metadata_clean.csv",
        "img_col": "image_id",
        "label_col": "dx",
        "patient_col": "lesion_id",
        "file_ext": ".jpg",
        "output_dir": "/path/to/ham10000/vgg19_results"
    }
]

# ==========================================================
# TRANSFORMS
# ==========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# DATASET CLASS
# ==========================================================

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform, label_col, img_id_col):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col
        self.img_id_col = img_id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        image_id = str(row[self.img_id_col])

        return image, label, image_id

# ==========================================================
# FOCAL LOSS
# ==========================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none"
        )
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ==========================================================
# VGG19 MODEL
# ==========================================================

def build_model():
    model = models.vgg19(
        weights=models.VGG19_Weights.IMAGENET1K_V1
    )

    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        1
    )

    return model.to(DEVICE)

# ==========================================================
# HELPERS
# ==========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def safe_group_column(df, preferred_group_col, fallback_img_col):
    if preferred_group_col in df.columns:
        return preferred_group_col
    return fallback_img_col

def build_paths(df, img_dir, img_col, file_ext):
    df = df.copy()
    df["path"] = df[img_col].astype(str).apply(
        lambda x: str(Path(img_dir) / f"{x}{file_ext}")
    )
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    return df

def create_binary_target_if_needed(df, label_col, dataset_type):
    df = df.copy()

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")

    if not pd.api.types.is_numeric_dtype(df[label_col]):
        if dataset_type == "ham10000":
            mapping = {
                "mel": 1,
                "bcc": 1,
                "akiec": 1,
                "nv": 0,
                "bkl": 0,
                "df": 0,
                "vasc": 0
            }
            df[label_col] = df[label_col].map(mapping)

    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    return df

def make_weighted_sampler(train_df, label_col):
    class_counts = train_df[label_col].value_counts().to_dict()
    weights = train_df[label_col].map(
        lambda x: 1.0 / class_counts[x]
    ).values
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_probs = []
    image_ids = []

    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(DEVICE)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)

            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            y_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
            image_ids.extend(list(ids))

    return np.array(y_true), np.array(y_probs), image_ids

def optimize_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)

    if len(thresholds) == 0:
        return 0.5, precision, recall, thresholds

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    return best_threshold, precision, recall, thresholds

def save_confusion_matrix(cm, out_path, title):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_roc_curve(y_true, y_probs, out_path, title):
    fpr_m, tpr_m, _ = roc_curve(y_true, y_probs)
    auc_m = auc(fpr_m, tpr_m)

    fpr_b, tpr_b, _ = roc_curve(1 - y_true, 1 - y_probs)
    auc_b = auc(fpr_b, tpr_b)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_b,
        tpr_b,
        linewidth=2,
        label=f"Benign (AUC={auc_b:.4f})"
    )
    plt.plot(
        fpr_m,
        tpr_m,
        linewidth=2,
        label=f"Malignant (AUC={auc_m:.4f})"
    )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return auc_m, auc_b

def save_pr_curve(recall, precision, pr_auc, out_path, title):
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"PR-AUC={pr_auc:.4f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# MAIN RUN FUNCTION
# ==========================================================

def run_pipeline(cfg):
    print("\n" + "=" * 70)
    print(f"Running: {cfg['name']}")
    print("=" * 70)

    img_dir = Path(cfg["img_dir"])
    csv_file = Path(cfg["csv_file"])
    img_col = cfg["img_col"]
    label_col = cfg["label_col"]
    patient_col = cfg["patient_col"]
    file_ext = cfg["file_ext"]
    output_dir = Path(cfg["output_dir"])
    dataset_type = cfg["dataset_type"]

    ensure_dir(output_dir)

    # ------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------
    print("\nLoading metadata...")
    df = pd.read_csv(csv_file)

    df = create_binary_target_if_needed(df, label_col, dataset_type)
    df = build_paths(df, img_dir, img_col, file_ext)

    print("\nDataset loaded")
    print("Total samples:", len(df))

    if len(df) == 0:
        print("No valid image paths found. Skipping.")
        return None

    print("\nClass distribution:")
    print(df[label_col].value_counts())

    # ------------------------------------------------------
    # GROUP SPLIT
    # ------------------------------------------------------
    print("\nPerforming patient/lesion-level split...")

    group_col = safe_group_column(df, patient_col, img_col)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        gss.split(df, groups=df[group_col])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print("\nTrain size:", len(train_df))
    print("Test size :", len(test_df))

    print("\nTrain distribution:")
    print(train_df[label_col].value_counts())

    print("\nTest distribution:")
    print(test_df[label_col].value_counts())

    # ------------------------------------------------------
    # DATASETS
    # ------------------------------------------------------
    train_dataset = SkinDataset(
        train_df,
        train_transform,
        label_col=label_col,
        img_id_col=img_col
    )

    test_dataset = SkinDataset(
        test_df,
        test_transform,
        label_col=label_col,
        img_id_col=img_col
    )

    print("\nDatasets created successfully")

    # ------------------------------------------------------
    # WEIGHTED RANDOM SAMPLER
    # ------------------------------------------------------
    print("\nCreating weighted sampler...")
    sampler = make_weighted_sampler(train_df, label_col)

    # ------------------------------------------------------
    # DATALOADERS
    # ------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print("\nDataLoaders ready")

    # ------------------------------------------------------
    # MODEL + LOSS + OPTIMIZER + SCHEDULER
    # ------------------------------------------------------
    print("\nLoading VGG19...")
    model = build_model()
    print("Model loaded ✔")

    criterion = FocalLoss(alpha=0.75, gamma=2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # ------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------
    best_pr_auc = -1.0
    best_state = None
    history = []

    print("\nStarting Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)

        # ==================================================
        # VALIDATION
        # ==================================================
        val_true, val_probs, _ = evaluate_model(model, test_loader)
        pr_auc = average_precision_score(val_true, val_probs)

        scheduler.step(pr_auc)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_pr_auc": float(pr_auc)
        })

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Loss={avg_loss:.4f} | "
            f"PR-AUC={pr_auc:.4f}"
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = copy.deepcopy(model.state_dict())
            print(f"New Best Model Saved (PR-AUC={pr_auc:.4f})")

    print("\nTraining Finished")
    print(f"Best Validation PR-AUC: {best_pr_auc:.4f}")

    best_model_path = output_dir / "best_vgg19.pth"
    torch.save(best_state, best_model_path)

    pd.DataFrame(history).to_csv(
        output_dir / "training_history_vgg19.csv",
        index=False
    )

    # ------------------------------------------------------
    # LOAD BEST MODEL
    # ------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    print("\nBest VGG19 loaded.")

    # ------------------------------------------------------
    # EVALUATION + REPORTS + PLOTS
    # ------------------------------------------------------
    print("\nGenerating predictions...")

    y_true, y_probs, image_ids = evaluate_model(model, test_loader)

    # ------------------------------------------------------
    # OPTIMAL THRESHOLD
    # ------------------------------------------------------
    best_threshold, precision, recall, thresholds = optimize_threshold(y_true, y_probs)
    print("\nBest Threshold:", best_threshold)

    y_pred = (y_probs >= best_threshold).astype(int)

    # ------------------------------------------------------
    # SAVE PREDICTIONS CSV
    # ------------------------------------------------------
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability": y_probs
    })

    pred_df.to_csv(output_dir / "vgg19_predictions.csv", index=False)
    print("Saved: vgg19_predictions.csv")

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report_vgg19.csv")

    print("\nClassification Report")
    print(report_df)

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(
        cm,
        output_dir / "confusion_matrix_vgg19.png",
        f"{cfg['name']} VGG19 Confusion Matrix"
    )

    # ------------------------------------------------------
    # ROC CURVE
    # ------------------------------------------------------
    auc_m, auc_b = save_roc_curve(
        y_true,
        y_probs,
        output_dir / "roc_auc_vgg19.png",
        f"{cfg['name']} VGG19 ROC Curve"
    )

    # ------------------------------------------------------
    # PR CURVE
    # ------------------------------------------------------
    pr_auc = average_precision_score(y_true, y_probs)
    save_pr_curve(
        recall,
        precision,
        pr_auc,
        output_dir / "precision_recall_vgg19.png",
        f"{cfg['name']} VGG19 Precision-Recall Curve"
    )

    # ------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    summary = {
        "dataset_name": cfg["name"],
        "n_samples_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best_threshold": float(best_threshold),
        "balanced_accuracy": float(bal_acc),
        "roc_auc_malignant": float(auc_m),
        "roc_auc_benign": float(auc_b),
        "pr_auc": float(pr_auc),
        "best_validation_pr_auc": float(best_pr_auc)
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "final_metrics_summary_vgg19.csv",
        index=False
    )
    save_json(summary, output_dir / "final_metrics_summary_vgg19.json")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Benign ROC-AUC    : {auc_b:.4f}")
    print(f"Malignant ROC-AUC : {auc_m:.4f}")
    print(f"PR-AUC            : {pr_auc:.4f}")
    print(f"Best Threshold    : {best_threshold:.4f}")
    print("=" * 60)

    print("\nSaved files:")
    print("best_vgg19.pth")
    print("vgg19_predictions.csv")
    print("classification_report_vgg19.csv")
    print("confusion_matrix_vgg19.png")
    print("roc_auc_vgg19.png")
    print("precision_recall_vgg19.png")

    print("\nPipeline completed successfully.")

    return summary

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    all_results = []

    for cfg in DATASET_CONFIGS:
        result = run_pipeline(cfg)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 0:
        combined_out = Path("./combined_vgg19_results")
        ensure_dir(combined_out)

        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(
            combined_out / "all_datasets_summary_vgg19.csv",
            index=False
        )
        save_json(
            all_results,
            combined_out / "all_datasets_summary_vgg19.json"
        )

        print("\nCombined results saved ✔")
        print(combined_df)
