# contrast-focused-skin-lesion-analysis
Contrast-focused deep learning framework for automated skin lesion analysis using the HAM10000 dataset. The project evaluates four image preprocessing pipelines combined with U-Net segmentation and 25 pretrained deep learning models to study their impact on multi-class skin lesion classification performance and diagnostic accuracy.


This repository contains the implementation of the framework proposed in the paper:

**“Contrast-Focused Preprocessing for Skin Lesion Segmentation and Classification.”**

The framework evaluates the impact of **four preprocessing pipelines** on segmentation quality and multi-class classification performance using the **HAM10000 dataset**.

---

## Overview

Skin cancer is one of the most common cancers worldwide. Early and accurate detection is essential for improving patient outcomes.

This project proposes a **multi-stage deep learning pipeline** that integrates:

1. Image preprocessing  
2. Lesion segmentation  
3. Post-processing refinement  
4. Multi-model classification  

The goal is to analyze how different preprocessing strategies influence segmentation accuracy and classification performance.

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
pip install -r requirements.txt
```

---
## Framework Architecture

The proposed framework follows a structured pipeline:

![image alt](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/36eaaed9d7f8b0f4c6ba6069a0e5272ff7f632d3/architecture_up.png)

## Preprocessing Pipelines

Four preprocessing pipelines are evaluated. The figure below shows the preprocessing steps for each pipeline applied to the same sample image:

![Preprocessing Pipelines Comparison](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/f22579e3b6adebcea28066182f0aaadd977861db/preprocessing_steps_pipelines.png)

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
- Resize (256*256)
- CLAHE enhancement
- Intensity normalization

Pipeline 4 produced the **best overall classification performance**.

## Segmentation Model

Lesion segmentation is performed using a **U-Net convolutional neural network**, a widely adopted architecture in medical image analysis due to its ability to capture contextual information while preserving fine spatial details.

### Segmentation Example

The figure below illustrates the segmentation process, including the transformed input image, predicted lesion mask, and the final overlay of the segmented lesion.

![Segmentation Example](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/7e46f5ea5e86c54fde2cc0bde2302a90dd286bd1/seg_up_.png)

### Architecture

The segmentation network follows an **encoder–decoder structure**:

- **Encoder:** Extracts hierarchical features from dermoscopic images using successive convolution and pooling layers.
- **Decoder:** Reconstructs the segmentation mask by progressively upsampling feature maps.
- **Skip Connections:** Direct links between encoder and decoder layers preserve high-resolution spatial information and improve lesion boundary detection.

### Additional Enhancements

To improve segmentation performance on dermoscopic images, the framework includes several enhancements:

- **Sonar-inspired Background Transformation:** Enhances the contrast between lesion regions and surrounding skin, helping the network better identify lesion boundaries.
- **Morphological Post-processing:** Applies operations such as opening, closing, and small-region removal to refine predicted masks and reduce segmentation noise.

### Evaluation Metrics

Segmentation performance is evaluated using the following metrics:

- **Dice Coefficient:** Measures the overlap between predicted masks and ground truth masks.
- **Intersection over Union (IoU):** Ratio between the intersection and the union of predicted and actual lesion regions.
- **Jaccard Index:** A similarity metric closely related to IoU for segmentation quality assessment.
- **Sensitivity:** Measures the proportion of actual lesion pixels correctly identified by the model.
- **Pixel Accuracy:** The proportion of correctly classified pixels over the total number of pixels.


These metrics provide a comprehensive evaluation of the segmentation model's ability to accurately delineate lesion boundaries before the classification stage.

## Classification Models

To evaluate the impact of preprocessing strategies on diagnostic performance, the framework tests **25 pretrained deep learning models** across the four preprocessing pipelines. These models include classical convolutional neural networks, lightweight architectures, and modern transformer-based networks.

### Evaluated Models

The following pretrained architectures were used for multi-class skin lesion classification:

- VGG16
- VGG19
- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152
- DenseNet121
- DenseNet161
- DenseNet169
- DenseNet201
- InceptionV3
- InceptionResNetV2
- Xception
- MobileNetV1
- MobileNetV2
- MobileNetV3 Small
- MobileNetV3 Large
- EfficientNetB0
- EfficientNetB1
- EfficientNetB2
- EfficientNetB3
- EfficientNetB4
- EfficientNetB5
- EfficientNetB6
- EfficientNetB7

These models were fine-tuned on dermoscopic images from the **HAM10000 dataset** to perform **seven-class skin lesion classification**.

---

## Classification Metrics

The performance of each classification model is evaluated using several standard metrics commonly used in medical image analysis.

### Precision

Precision measures the proportion of correctly predicted positive samples among all predicted positive samples.

$$
Precision = \frac{TP}{TP + FP}
$$

Where:

- **TP** = True Positives  
- **FP** = False Positives  

A high precision indicates that the model produces **few false positive predictions**, meaning that predicted lesion classes are more likely to be correct.

---

### Recall (Sensitivity)

Recall measures the proportion of actual positive samples that are correctly detected by the model.

$$
Recall = \frac{TP}{TP + FN}
$$

Where:

- **TP** = True Positives  
- **FN** = False Negatives  

High recall is particularly important in medical diagnosis because it ensures that **most real lesion cases are detected by the system**.

---

### F1-score

The F1-score is the **harmonic mean of precision and recall**, providing a balanced evaluation when both false positives and false negatives must be considered.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

This metric is especially useful when the dataset contains **class imbalance**, which is common in medical imaging datasets such as HAM10000.

---

### Accuracy

Accuracy measures the proportion of correctly classified samples among all predictions.

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Where:

- **TP** = True Positives  
- **TN** = True Negatives  
- **FP** = False Positives  
- **FN** = False Negatives  

Accuracy provides a general indication of model performance, although it may not fully reflect classification quality in imbalanced datasets. Therefore, additional metrics such as **precision, recall, and F1-score** are also considered.

---

## Experimental Comparison

The figure below compares the performance of the four preprocessing pipelines across all classification models using the four evaluation metrics.

![Pipeline Comparison](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/01f5f75c7d2feb67b632df6b07cca5e8f8897f4b/comparison_pipelines.png)

The results show that **Pipeline 4 consistently achieves higher precision, recall, F1-score, and accuracy across most models**, with **EfficientNetB7 demonstrating the competitive overall performance**.
