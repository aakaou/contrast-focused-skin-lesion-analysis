# Multi-Pipeline Preprocessing for Robust Skin Lesion Segmentation and Classification
Multi-Pipeline Preprocessing Framework for Robust Skin Lesion Segmentation and Classification using HAM10000
and ISIC 2024 dermoscopic datasets. The project evaluates four image preprocessing pipelines
combined with U-Net segmentation and a broad set of pretrained deep learning models to
study their impact on lesion segmentation quality, multi-class classification performance, and
diagnostic accuracy. The ISIC 2024 data is distributed through the official Kaggle competition
page for the challenge [https://www.kaggle.com/competitions/isic-2024-challenge/data]

This repository contains the implementation of the framework proposed in the paper
“Multi-Pipeline Preprocessing for Robust Skin Lesion Segmentation and Classification.” The
framework is designed to measure how preprocessing affects segmentation quality and
classification performance across different datasets and model families

---

## Overview
Skin lesion analysis benefits from early and accurate automated assessment, especially when
segmentation and classification are studied jointly in a consistent experimental pipeline. Prior
research also highlights the importance of combining lesion localization and classification in
dermoscopic image analysis workflows.

This project proposes a multi-stage deep learning framework that integrates:
1. Image preprocessing
2. Lesion segmentation
3. Post-processing refinement
4. Downstream classification
5. Explainability analysis

The main objective is to analyse how different preprocessing strategies influence segmentation accuracy, classification performance, and model interpretability.

## Datasets
This project uses two dermoscopic datasets:
* HAM10000: a dermoscopic image dataset  containing 10,015 images across seven diagnostic categories.
* ISIC 2024:  the official challenge dataset provided through the Kaggle competition ISIC 2024 - Skin Cancer Detection with 3D-TBP.

## Dataset sources
* HAM 10000(Kaggle): [https://kaggle/input/ham10000-dataset]

* ISIC 2024(Kaggle competition): [https://www.kaggle.com/competitions/isic-2024-challenge/data]

## HAM10000 classes
HAM 10000 contains seven diagnostic categories:
* Melanocytic nevi (NV)
* Melanoma (MEL)
* Benign keratosis (BKL)
* Basal cell carcinoma (BCC)
* Actinic keratoses (AKIEC)
* Vascular lesions (VASC)
* Dermatofibroma (DF)
  
![image alt](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/e29ecee96d265824ed131484e653f30b97879259/schema%20of%20datasets.png)

Use consistent naming for iamge and mask pairs and keep split definitions in CSV files. When training across both datasets, adding a source column such as HAM10000 or ISIC2024 helps with controlled analysis and stratified evaluation.

## Installation

Clone the repository:

```bash
git clone https://github.com/aakaou/contrast-focused-skin-lesion-analysis.git
cd contrast-focused-skin-lesion-analysis
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pi install --upgrade pip
pip install -r requirements.txt
```
## Framework Architecture

The proposed framework follows a structured pipeline composed of preprocessing,
segmentation, post-processing, classification, and explainability analysis. This type of
sequential pipeline is consistent with published dermoscopic workflows that combine lesion
localization with downstream prediction.

## Main stages:
* Input dermoscopic image
* Four preprocessing pipelines
* Sonar-inspired enhancement
* UNet-based lesion segmentation
* Morphological post-processing
* Segmentation mask and overlay generation
* Multi-class and binary classification
* Grad-CAM explainability analysis

![image alt](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/5a5a99295f75bbac403fe9eec97ff4b5f9c0c910/architecture_up.png)

## Pipeline design
The framework studies how contrast-focused preprocessing changes the visibility of lesion
structures before segmentation and classification. Explainability is used to verify whether
classification models attend to lesion regions rather than irrelevant background patterns,
which is aligned with Grad-CAM-based interpretation practices in skin lesion analysis.

## Preprocessing Pipelines

Four preprocessing pipelines are evaluated on the dermoscopic images before segmentation and classification.

### Pipeline 1 – Baseline
- Image resizing
- Intensity normalization

### Pipeline 2 – Artifact Removal and Contrast Enhancement
- Hair removal (black-hat transformation)
- Inpainting
- White balance correction
- CLAHE contrast enhancement
- Normalization

### Pipeline 3 – Texture Enhancement
- Resize 
- Hair removal
- Bilateral filtering
- Wavelet-based enhancement
- Gabor filter bank
- Unsharp masking
- Normalization

### Pipeline 4 – Optimized Contrast Pipeline
- DullRazor hair removal
- Inpainting
- Resize (256x256)
- CLAHE enhancement
- Intensity normalization

Pipeline 4 produced  the strongest overall classification behavior in the current experimental setting described in the repository.

## Segmentation Model

Lesion segmentation is performed using a **U-Net convolutional neural network**, a widely adopted architecture in medical image analysis due to its ability to capture contextual information while preserving fine spatial details.

### Segmentation Example

Lesion segmentation is performed using a U-Net architecture. U-Net remains a widely used
segmentation model in medical imaging because its encoder-decoder structure and skip
connections preserve both semantic and spatial information needed for fine boundary
delineation.

The segmentation network follows an **encoder–decoder structure**:

- **Encoder:** Extracts hierarchical features from dermoscopic images using successive convolution and pooling layers.
- **Bottleneck:** captures compact high-level lesion representations.
- **Decoder:** Reconstructs the segmentation mask by progressively upsampling feature maps.
- **Skip Connections:** Direct links between encoder and decoder layers preserve high-resolution spatial information and improve lesion boundary detection.

### Additional Enhancements

The segmentation stage also includes:

- **Sonar-inspired Background Transformation:** Enhances the contrast between lesion regions and surrounding skin, helping the network better identify lesion boundaries.
- **Morphological Post-processing:** Applies operations such as opening, closing, and small-region removal to refine predicted masks and reduce segmentation noise.

Segmentation quality is evaluated with:
* Dice coefficient
* Intersection over Union (IoU)
* Jaccard index
* Sensitivity
* Accuracy

These metrics provide a complementary view of overlap quality and pixel-wise lesion recovery performance.

### Classification Models
To evaluate the effect of preprocessing on diagnosis, the framework compares a large set of
pretrained deep learning architectures. Published work in skin lesion analysis commonly
benchmarks multiple CNN backbones because feature extraction behavior can vary
substantially across model families.

The repository description refers to evaluation across pretrained models from the following families:

- VGG (16, 19)
- ResNet (18, 34, 50, 101, 152)
- DenseNet (121, 169, 201)
- InceptionV3
- InceptionResNetV2
- Xception
- MobileNet (V1, V2)
- MobileNetV3 Small
- MobileNetV3 Large
- EfficientNet (B0, B1, B2, B3, B4, B5, B6, B7)

## Tasks
The framework supports:
*  Multi-class classification for lesion category prediction.
*  Binary classification for benign versus malignant prediction when binary labels are prepared.

Using segmented lesions or lesion overlays as classifier inputs helps align diagnosis with the localized lesion region rather than unrelated background content. 

## Explainability Analysis

Grad-CAM is used to generate heatmaps from the last convolutional layer of each evaluated
classification model. Grad-CAM has been used in skin lesion analysis studies to visualize the
image regions that contribute most to a model prediction.

The explainability analysis can include:

*  Grad-CAM heatmap generation
*  Overlay on the original dermoscopic image
*  Comparison with lesion segmentation masks
*  Overlap-based metrics such as IoU or Dice between thresholded Grad-CAM and lesion masks
*  Percnetation of activation inside the lesion region

this step helps assess whether the classifier attends to clinically meaningful lesion structures.

## Evaluatioin Metrics
### Segmentation
* Dice coefficient
* IoU / Jaccard index
* Sensitivity
* Accuracy

### Classification
* Precision
* Recall
* F1-score
* Accuracy
* AUC

### Explainability
* Heatmap-mask overlap
* Activation percentage inside lesion area
* Correlation or region-consistency measures when implemented

These metrics make it possible to analyze the pipeline from three complementary perspectives: lesion localization, diagnosis, and interpretability.

### How to Run Experiments

The exact commands should match the real script names in the repository. The examples below provide a clear template for documenting training workflows.
1. Run
```bash
# run file of segmentation 
python u_net_segmentation.py
# run file of classification
python classification.py
# generate grad cam explanations
python generate_gradcam.py
```
### Reproducibility Notes
For reproducible experiments, document the following in the final repository version:
* Python version
* CUDA version
* Deep learning framwork version
* Random seed
* Dataset split files
* Trained checkpoint names
* output directories for masks, predictions, and Grad-CAM results

Keeping dataset-specific test results separate for HAM10000 and ISIC2024 is recommended so
performance changes can be attributed to preprocessing, architecture, or domain shi rather
than to mixed evaluation settings.

