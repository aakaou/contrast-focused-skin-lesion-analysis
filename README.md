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
