# ==========================================================
# FULL RESNET34 CLINICAL PIPELINE
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
    balanced_accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# =========================
# REPRODUCIBILITY
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# =========================
# DATASET CONFIGS
# =========================
DATASET_CONFIGS = [
    {
        "name": "ISIC2024_pipeline_each_time",
        "dataset_type": "isic2024",
        "img_dir": "/path/to/pipeline/unet_overlays_up",
        "csv_file": "/path/to/train_metadata_clean_full.csv",
        "img_col": "isic_id",
        "label_col": "target",
        "patient_col": "patient_id",
        "file_ext": ".jpg",
        "output_dir": "/path/to/resnet34_results"
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
        "output_dir": "/path/to/ham10000/resnet34_results"
    }
]

# =========================
# TRANSFORMS
# =========================
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

# =========================
# DATASET
# =========================
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

        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        image_id = str(row[self.img_id_col])

        return img, label, image_id

# =========================
# HELPERS
# =========================
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
    weights = train_df[label_col].map(lambda x: 1.0 / class_counts[x]).values
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler

def get_pos_weight(train_df, label_col):
    positives = float(train_df[label_col].sum())
    negatives = float(len(train_df) - positives)

    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32).to(DEVICE)

    return torch.tensor([negatives / positives], dtype=torch.float32).to(DEVICE)

def evaluate_model(model, loader):
    model.eval()
    y_true, y_probs, image_ids = [], [], []

    with torch.no_grad():
        for imgs, labels, ids in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs).squeeze()
            probs = torch.sigmoid(outputs)

            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            y_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
            image_ids.extend(list(ids))

    return np.array(y_true), np.array(y_probs), image_ids

def optimize_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    if len(thresholds) == 0:
        return 0.5, precision, recall, thresholds

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    return best_threshold, precision, recall, thresholds

def save_confusion_matrix(cm, out_path, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        cmap="Blues"
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_roc_curve(y_true, y_probs, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc

def save_pr_curve(y_true, y_probs, out_path, title):
    pr_auc = average_precision_score(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pr_auc

# =========================
# MODEL (RESNET34)
# =========================
def build_model():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(DEVICE)
    return model

# =========================
# RUN PIPELINE
# =========================
def run_pipeline(cfg):
    print("\n" + "=" * 70)
    print(f"Running: {cfg['name']}")
    print("=" * 70)

    img_dir = Path(cfg["img_dir"])
    csv_file = Path(cfg["csv_file"])
    label_col = cfg["label_col"]
    img_col = cfg["img_col"]
    patient_col = cfg["patient_col"]
    file_ext = cfg["file_ext"]
    dataset_type = cfg["dataset_type"]
    output_dir = Path(cfg["output_dir"])

    ensure_dir(output_dir)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(csv_file)
    df = create_binary_target_if_needed(df, label_col, dataset_type)
    df = build_paths(df, img_dir, img_col, file_ext)

    print("Total samples:", len(df))
    print(df[label_col].value_counts())

    if len(df) == 0:
        print("No valid images found. Skipping.")
        return None

    # =========================
    # PATIENT / LESION SPLIT
    # =========================
    group_col = safe_group_column(df, patient_col, img_col)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print("Train:", len(train_df), "Test:", len(test_df))

    # =========================
    # BALANCED SAMPLING
    # =========================
    sampler = make_weighted_sampler(train_df, label_col)

    # =========================
    # DATASET + DATALOADER
    # =========================
    train_ds = SkinDataset(
        train_df,
        train_transform,
        label_col=label_col,
        img_id_col=img_col
    )
    test_ds = SkinDataset(
        test_df,
        test_transform,
        label_col=label_col,
        img_id_col=img_col
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # =========================
    # MODEL + LOSS + OPTIMIZER
    # =========================
    model = build_model()

    pos_weight = get_pos_weight(train_df, label_col)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # =========================
    # TRAINING
    # =========================
    best_loss = np.inf
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss
        })

        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            print("New best model saved ✔")

    print("Training completed ✔")

    best_model_path = output_dir / "best_resnet34.pth"
    torch.save(best_state, best_model_path)

    pd.DataFrame(history).to_csv(
        output_dir / "training_history_resnet34.csv",
        index=False
    )

    # =========================
    # LOAD BEST MODEL
    # =========================
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # =========================
    # EVALUATION
    # =========================
    y_true, y_probs, image_ids = evaluate_model(model, test_loader)

    # =========================
    # THRESHOLD OPTIMIZATION
    # =========================
    best_threshold, precision, recall, thresholds = optimize_threshold(y_true, y_probs)
    print("Best threshold:", best_threshold)

    y_pred = (y_probs >= best_threshold).astype(int)

    # =========================
    # SAVE PREDICTIONS CSV
    # =========================
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "probability": y_probs
    })
    pred_df.to_csv(output_dir / "resnet34_predictions.csv", index=False)
    print("Saved predictions ✔")

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_dir / "classification_report_resnet34.csv")
    print(report_df)

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(
        cm,
        output_dir / "confusion_matrix_resnet34.png",
        f"{cfg['name']} Confusion Matrix"
    )

    # =========================
    # ROC CURVE
    # =========================
    roc_auc = save_roc_curve(
        y_true,
        y_probs,
        output_dir / "roc_auc_curve_resnet34.png",
        f"{cfg['name']} ROC Curve"
    )

    # =========================
    # PR CURVE
    # =========================
    pr_auc = save_pr_curve(
        y_true,
        y_probs,
        output_dir / "pr_curve_resnet34.png",
        f"{cfg['name']} Precision-Recall Curve"
    )

    # =========================
    # FINAL METRICS
    # =========================
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    summary = {
        "dataset_name": cfg["name"],
        "n_samples_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best_train_loss": float(best_loss),
        "best_threshold": float(best_threshold),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "final_metrics_summary_resnet34.csv",
        index=False
    )
    save_json(summary, output_dir / "final_metrics_summary_resnet34.json")

    print("=" * 50)
    print("Balanced Accuracy:", bal_acc)
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("=" * 50)
    print("ALL OUTPUTS SAVED ✔")

    return summary

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    all_results = []

    for cfg in DATASET_CONFIGS:
        result = run_pipeline(cfg)
        if result is not None:
            all_results.append(result)

    if len(all_results) > 0:
        combined_out = Path("./combined_resnet34_results")
        ensure_dir(combined_out)

        combined_df = pd.DataFrame(all_results)
        combined_df.to_csv(
            combined_out / "all_datasets_summary_resnet34.csv",
            index=False
        )
        save_json(
            all_results,
            combined_out / "all_datasets_summary_resnet34.json"
        )

        print("\nCombined results saved ✔")
        print(combined_df)
