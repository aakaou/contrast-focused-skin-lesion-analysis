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

![image alt](https://github.com/aakaou/contrast-focused-skin-lesion-analysis/blob/e070518686b903b02247c0976aa7abd77222091b/architecture_up_with_gradcam.png)

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

### Citation
This repository contributes our research:
```bash
## References

[1] A. Aakaou, K. Thurnhofer-Hemsi, and E. Domínguez, "Contrast-Focused Preprocessing for Skin Lesion Segmentation and Classification," in *Proc. International Work-Conference on the Interplay Between Natural and Artificial Computation (IWINAC)*, pp. 63--72, Springer, 2026.

[2] J. Biro *et al*., "Analysis of the ISIC image datasets: Usage, benchmarks and recommendations," *Medical Image Analysis*, vol. 75, p. 102304, Jan. 2022.

[3] P. Tschandl *et al*., "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *Scientific Data*, vol. 5, p. 180161, Aug. 2018.

[4] N. Codella *et al*., "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the ISIC," *arXiv:1902.03368*, 2019.

[5] N. Codella *et al*., "Skin lesion analysis toward melanoma detection 2017: An ISIC challenge," *arXiv:1710.05006*, 2017.

[6] N. Codella *et al*., "Skin lesion analysis toward melanoma detection 2016: An ISIC challenge," *arXiv:1601.02663*, 2016.

[7] A. Esteva *et al*., "Dermatologist-level classification of skin cancer with deep neural networks," *Nature*, vol. 542, no. 7639, pp. 115--118, Feb. 2017.

[8] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *MICCAI*, Munich, Germany, 2015, pp. 234--241.

[9] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *ICML*, Long Beach, CA, USA, 2019, pp. 6105--6114.

[10] G. Huang *et al*., "Densely connected convolutional networks," in *CVPR*, Honolulu, HI, USA, 2017, pp. 4700--4708.

[11] K. He *et al*., "Deep residual learning for image recognition," in *CVPR*, Las Vegas, NV, USA, 2016, pp. 770--778.

[12] S. Bibi *et al*., "MSRNet: Multiclass skin lesion recognition using additional residual block based fine-tuned deep models," *Diagnostics*, vol. 13, no. 19, p. 3063, 2023.

[13] S. Jain *et al*., "Deep learning-based transfer learning for classification of skin cancer," *Sensors*, vol. 21, no. 23, p. 8142, Dec. 2021.

[14] M. Fraiwan and E. Faouri, "On the automatic detection and classification of skin cancer using deep transfer learning," *Sensors*, vol. 22, no. 13, p. 4963, 2022.

[15] J. Rashid *et al*., "Skin cancer disease detection using transfer learning technique," *Applied Sciences*, vol. 12, no. 11, p. 5714, 2022.

[16] G. Alwakid *et al*., "Melanoma detection using deep learning-based classifications," *Healthcare*, vol. 10, no. 12, p. 2481, 2022.

[17] I. Abunadi and E. M. Senan, "Deep learning and machine learning techniques of diagnosis dermoscopy images for early detection of skin diseases," *Electronics*, vol. 10, no. 24, p. 3158, 2021.

[18] A. A. Adegun, S. Viriri, and M. H. Yousaf, "A probabilistic-based deep learning model for skin lesion segmentation," *Applied Sciences*, vol. 11, no. 7, p. 3025, Mar. 2021.

[19] A. Mahbod *et al*., "The effect of region of interest crops and preprocessing on skin lesion classification performance," *Symmetry*, vol. 11, no. 8, p. 1042, Aug. 2019.

[20] C. H. Lee, C. J. Li, and C. Y. Chen, "DullRazor: A software approach to hair removal from images," *Computers in Biology and Medicine*, vol. 27, no. 6, pp. 533--543, Nov. 1997.

[21] K. Zuiderveld, "Contrast limited adaptive histogram equalization," in *Graphics Gems IV*, San Diego, CA, USA: Academic, 1994, pp. 474--485.

[22] S. M. Pizer *et al*., "Adaptive histogram equalization and its variations," *Computer Vision, Graphics, Image Processing*, vol. 39, no. 3, pp. 355--368, Sep. 1987.

[23] A. Telea, "An image inpainting technique based on the fast marching method," *Journal of Graphics Tools*, vol. 9, no. 1, pp. 23--34, 2004.

[24] M. E. Celebi *et al*., "Lesion border detection in dermoscopy images," *Computerized Medical Imaging and Graphics*, vol. 33, no. 2, pp. 148--153, Mar. 2009.

[25] M. Combalia *et al*., "Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images," *JAMA Dermatology*, vol. 157, no. 7, pp. 800--807, Jul. 2021.

[26] V. Rotemberg *et al*., "A patient-centric dataset of images and metadata for identifying melanomas using clinical context," *Scientific Data*, vol. 8, no. 1, p. 34, Jan. 2021.

[27] F. Alenezi, A. Armghan, and K. Polat, "A novel multi-task learning network based on melanoma segmentation and classification," *Diagnostics*, vol. 13, no. 2, p. 262, 2023.

[28] M. A. Khan *et al*., "Skin lesion segmentation and multiclass classification using deep learning features," *Mathematics*, vol. 11, no. 1, p. 247, 2023.

[29] M. Z. Ur Rehman *et al*., "Classification of skin cancer lesions using explainable deep learning," *Sensors*, vol. 22, no. 18, p. 6915, 2022.

[30] M. H. Strzelecki *et al*., "Skin lesion detection algorithms in whole body images," *Sensors*, vol. 21, no. 19, p. 6639, 2021.

[31] P. Jayaraman *et al*., "Wavelet-based classification of enhanced melanoma skin lesions through deep neural architectures," *Information*, vol. 13, no. 12, p. 583, 2022.

[32] P. Velez *et al*., "Does a previous segmentation improve the automatic detection of basal cell carcinoma using deep neural networks?," *Applied Sciences*, vol. 12, no. 4, p. 2092, 2022.

[33] M. U. Ali *et al*., "Enhancing skin lesion detection: A multistage multiclass convolutional neural network-based framework," *Bioengineering*, vol. 10, no. 12, p. 1430, 2023.

[34] T. G. Debelee *et al*., "Skin lesion classification and detection using machine learning techniques: A systematic review," *Diagnostics*, vol. 13, no. 19, p. 3147, 2023.

[35] K. Behara, E. Bhero, and J. T. Agee, "An improved skin lesion classification using a hybrid approach with active contour snake model and lightweight attention-guided capsule networks," *Diagnostics*, vol. 14, no. 6, p. 636, 2024.

[36] M. Obayya *et al*., "Internet of things-assisted smart skin cancer detection using metaheuristics with deep learning model," *Cancers*, vol. 15, no. 20, p. 5016, 2023.

[37] X. Yang *et al*., "A novel multi-task deep learning model for skin lesion segmentation and classification," *arXiv:1703.01025*, 2017.

[38] N. R. Kurtansky *et al*., "The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for skin cancer detection," *Scientific Data*, vol. 11, no. 1, p. 884, 2024.

[39] Y. Li and L. Shen, "Skin lesion analysis towards melanoma detection using deep learning network," *Sensors*, vol. 18, no. 2, p. 556, 2018.

[40] H. El-Khatib, D. Popescu, and L. Ichim, "Deep learning-based methods for automatic diagnosis of skin lesions," *Sensors*, vol. 20, no. 6, p. 1753, 2020.

[41] American Cancer Society, "Cancer Facts & Figures 2025," Atlanta: American Cancer Society, 2025.

[42] M. A. Albahar, "Skin lesion classification using convolutional neural network with novel regularizer," *IEEE Access*, vol. 7, pp. 38306--38313, 2019.

[43] M. A. Khan, M. Y. Javed, M. Sharif, T. Saba, and A. Rehman, "Multi-model deep neural network based features extraction and optimal selection approach for skin lesion classification," in *Proc. 2019 International Conference on Computer and Information Sciences (ICCIS)*, 2019, pp. 1--7.

[44] A. Mahbod, G. Schaefer, I. Ellinger, R. Ecker, A. Pitiot, and C. Wang, "Fusing fine-tuned deep features for skin lesion classification," *Computerized Medical Imaging and Graphics*, vol. 71, pp. 19--29, 2019.

[45] K. P. Arjun, K. Sampath Kumar, R. K. Dhanaraj, V. Ravi, and T. Ganesh Kumar, "Optimizing time prediction and error classification in early melanoma detection using a hybrid RCNN-LSTM model," *Microscopy Research and Technique*, vol. 87, no. 8, pp. 1789--1809, 2024.

[46] M. Shakya, R. Patel, and S. Joshi, "A comprehensive analysis of deep learning and transfer learning techniques for skin cancer classification," *Scientific Reports*, vol. 15, no. 1, Art. no. 4633, 2025.

[47] K. M. Hosny, M. A. Kassem, and M. M. Fouad, "Classification of skin lesions using transfer learning and augmentation with AlexNet," *PLOS ONE*, vol. 14, no. 5, Art. no. e0217293, 2019.

[48] M. Z. Ur Rehman, F. Ahmed, S. A. Alsuhibany, S. S. Jamal, M. Z. Ali, and J. Ahmad, "Classification of skin cancer lesions using explainable deep learning," *Sensors*, vol. 22, no. 18, Art. no. 6915, 2022.

[49] S. Bibi, M. A. Khan, J. H. Shah, R. Damaševičius, A. Alasiry, M. Marzougui, M. Alhaisoni, and A. Masood, "MSRNet: Multiclass skin lesion recognition using additional residual block based fine-tuned deep models information fusion and best feature selection," *Diagnostics*, vol. 13, no. 19, Art. no. 3063, 2023.
```

