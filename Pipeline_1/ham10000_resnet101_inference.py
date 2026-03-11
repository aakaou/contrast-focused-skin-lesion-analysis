# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import os                      # operating system utilities
import cv2                     # image processing
import numpy as np             # numerical arrays
import pandas as pd            # dataframe handling
from pathlib import Path       # path manipulation
from tqdm import tqdm          # progress bar

import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


# ==========================================================
# 2. PATHS AND PARAMETERS
# ==========================================================

# Folder containing segmented images
PROC_DIR = Path("/kaggle/working/HAM10000_segmented_p1")

# CSV file to store predictions
OUTPUT_CSV = "/kaggle/working/ham10000_resnet101_segmented_predictions_p1.csv"

# Image size required by ResNet
IMG_SIZE = (224, 224)

# Number of images processed at once
BATCH_SIZE = 32


# ==========================================================
# 3. LOAD PRETRAINED RESNET101
# ==========================================================

print("Loading ResNet101 pretrained on ImageNet...")

# Load ResNet101 backbone without classification head
base_model = ResNet101(
    weights='imagenet',          # pretrained weights
    include_top=False,           # remove ImageNet classifier
    input_shape=(224,224,3)
)

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)  # spatial pooling
x = Dense(512, activation='relu')(x)             # dense layer
x = Dense(256, activation='relu')(x)             # dense layer
predictions = Dense(7, activation='softmax')(x)  # 7 skin lesion classes

# Build final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze pretrained layers
model.trainable = False

print("ResNet101 model ready")


# ==========================================================
# 4. CLASS NAMES
# ==========================================================

class_names = ['nv','mel','bkl','bcc','akiec','vasc','df']


# ==========================================================
# 5. LOAD SEGMENTED IMAGE FILES
# ==========================================================

# Collect jpg and png files
seg_files = list(PROC_DIR.glob("*.jpg")) + list(PROC_DIR.glob("*.png"))

# Sort images by numeric ID
def extract_number(filename):
    import re
    m = re.search(r'ISIC_(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

seg_files.sort(key=lambda x: extract_number(x.name))

print("Total segmented images:", len(seg_files))


# ==========================================================
# 6. PREDICTION LOOP
# ==========================================================

print("Running predictions...")

results = []

# Iterate through batches
for i in tqdm(range(0, len(seg_files), BATCH_SIZE)):

    batch_paths = seg_files[i:i+BATCH_SIZE]

    batch_images = []
    batch_filenames = []

    for img_path in batch_paths:

        # Read image
        image = cv2.imread(str(img_path))

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, IMG_SIZE)

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply ResNet preprocessing
        image = tf.keras.applications.resnet.preprocess_input(
            image.astype(np.float32)
        )

        batch_images.append(image)
        batch_filenames.append(img_path.name)

    if len(batch_images) == 0:
        continue

    # Convert to numpy array
    X_batch = np.array(batch_images)

    # Model prediction
    probs = model.predict(X_batch, verbose=0)

    # Predicted class index
    pred_classes = np.argmax(probs, axis=1)

    # Maximum probability
    pred_probs = np.max(probs, axis=1)

    # Save results
    for j,(fname,prob,pred_class) in enumerate(zip(batch_filenames,probs,pred_classes)):

        results.append({

            "filename": fname,

            "pred_class_id": int(pred_class),

            "pred_class_name": class_names[pred_class],

            "pred_confidence": float(pred_probs[j]),

            "prob_nv": float(prob[0]),
            "prob_mel": float(prob[1]),
            "prob_bkl": float(prob[2]),
            "prob_bcc": float(prob[3]),
            "prob_akiec": float(prob[4]),
            "prob_vasc": float(prob[5]),
            "prob_df": float(prob[6])

        })


# ==========================================================
# 7. SAVE PREDICTIONS
# ==========================================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_CSV,index=False)

print("Predictions saved to:", OUTPUT_CSV)
print("Images processed:", len(results_df))

# ==========================================================
# 8. IMPORT EVALUATION LIBRARIES
# ==========================================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================================
# 9. LOAD PREDICTIONS
# ==========================================================

pred_df = pd.read_csv(OUTPUT_CSV)


# ==========================================================
# 10. LOAD HAM10000 METADATA
# ==========================================================

metadata = pd.read_csv("/kaggle/input/ham10000-dataset/HAM10000_metadata.csv")

# Create filename column
metadata["filename"] = metadata["image_id"] + ".jpg"


# ==========================================================
# 11. MERGE PREDICTIONS WITH TRUE LABELS
# ==========================================================

df = pred_df.merge(metadata[['filename','dx']],on='filename',how='inner')

print("Merged dataset size:", df.shape)


# ==========================================================
# 12. CLASSIFICATION REPORT
# ==========================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(
        df['dx'],
        df['pred_class_name'],
        labels=class_names,
        digits=4
))


# ==========================================================
# 13. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(df['dx'],df['pred_class_name'],labels=class_names)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()

# ==========================================================
# 14. ROC CURVE ANALYSIS
# ==========================================================

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc


# Convert labels to one-hot
lb = LabelBinarizer()
y_true_bin = lb.fit_transform(df['dx'])


plt.figure(figsize=(12,9))

colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))

auc_scores = {}


for i,cls in enumerate(class_names):

    y_true_cls = y_true_bin[:,i]

    y_prob_cls = df[f'prob_{cls}'].values

    # Skip if not enough samples
    if y_true_cls.sum() < 2:
        print("Skipping class:", cls)
        continue

    # Compute ROC
    fpr,tpr,_ = roc_curve(y_true_cls,y_prob_cls)

    roc_auc = auc(fpr,tpr)

    auc_scores[cls] = roc_auc

    plt.plot(
        fpr,
        tpr,
        lw=3,
        color=colors[i],
        label=f"{cls} (AUC={roc_auc:.3f})"
    )


# Random baseline
plt.plot([0,1],[0,1],'k--',label="Random (AUC=0.5)")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves - ResNet101")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig("/kaggle/working/resnet101_roc_curve.png",dpi=300)

plt.show()


# ==========================================================
# 15. MACRO AUC
# ==========================================================

macro_auc = np.mean(list(auc_scores.values()))

print("\nMACRO AUC:", round(macro_auc,4))

print("\nAUC per class")

for k,v in auc_scores.items():
    print(k,":",round(v,3))
