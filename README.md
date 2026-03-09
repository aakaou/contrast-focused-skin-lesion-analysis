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

---

## Framework Architecture

The proposed framework follows a structured pipeline:

![image alt](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/36eaaed9d7f8b0f4c6ba6069a0e5272ff7f632d3/architecture_up.png)

## Preprocessing Pipelines

Four preprocessing pipelines are evaluated. The figure below shows the preprocessing steps for each pipeline applied to the same sample image:

![Preprocessing Pipelines Comparison](docs/figures/preprocessing_comparison.png)

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
