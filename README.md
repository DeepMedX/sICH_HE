# An End-to-End Deep Learning Pipeline for Hematoma Expansion Prediction in Spontaneous Intracerebral Hemorrhage Based on Non-Contrast Computed Tomography

---

## Authors
Qiang Yu, Xin Fan, Jinwei Li, Qianyu Hao, Youquan Ning, Shichao Long, Wenhao Jiang, Fajin Lv, Xianlei Yan, Quan Liu, Xiaoquan Xu, Zongqian Wu, Juan Peng, and Min Wu

---

## Project Overview
This repository provides a complete end-to-end deep learning pipeline for predicting hematoma expansion in spontaneous intracerebral hemorrhage (sICH) patients based on non-contrast CT (NCCT) scans. The pipeline includes three stages: automated hematoma segmentation, synthetic minority data augmentation, and automated hematoma classification.

---

## Overall Study Design
The modular pipeline comprises three sequential stages:

**Stage 1: Automated Hematoma Segmentation**  
Four state-of-the-art 3D segmentation networks (**U-Mamba**, **nnU-Net**, **nnFormer**, and **UNETR++**) were benchmarked on a preliminary dataset of **1,000 NCCT scans** (baseline and follow-up) from **500 sICH patients** randomly selected from the full training cohort of **1,103 patients**.  
The best-performing model (**U-Mamba**) was then trained on all **2,206 scans** from the complete training cohort to generate **high-quality hematoma masks**.

**Stage 2: Synthetic Minority Data Augmentation**  
Synthetic minority data augmentation employed **Diffusion-UKAN** to generate high-fidelity synthetic HE images, yielding two augmented training sets:  
- **UKAN-Balanced:** HE : NHE = 1 : 1  
- **UKAN-Semibalanced:** HE : NHE = 1 : 2

**Stage 3: Automated Hematoma Classification**  
Automated classification used a **Vision Transformer (ViT)** trained on three consecutive slices centered on the maximum hematoma area (**Max−1, Max, Max+1**).  
Patient-level predictions were obtained by averaging slice-level probabilities, with **Grad-CAM** providing visual interpretation of discriminative regions.

---

## Environment Setup
```bash
conda create -n hematoma python=3.9
conda activate hematoma
pip install -r requirements.txt
```

---

## Data Preparation

**Automated Hematoma Segmentation (U-Mamba)**

```
Models/Automated Hematoma Segmentation/U-Mamba/data/
├── nnUNet_raw/
│   └── Task04_Hemorrhage/
│       ├── imagesTr/
│       ├── imagesTs/
│       └── labelsTr/
├── nnUNet_preprocessed/
└── nnUNet_results/
```

**Synthetic Minority Data Augmentation (Diffusion-UKAN)**

```
Models/Synthetic Minority Data Augmentation/Diffusion_UKAN/data/
└── HE/
    └── images_64/
```

**Automated Hematoma Classification (ViT)**

```
Models/Automated Hematoma Classification/ViT/data/
├── train/
│   ├── HE/
│   └── NHE/
├── val/
│   ├── HE/
│   └── NHE/
└── test/
    ├── HE/
    └── NHE/
```

---

## Training

**Segmentation Training**

```bash
cd Models/Automated Hematoma Segmentation/U-Mamba/umamba/nnunetv2/run
python run_training.py 004 3d_fullres 0 -tr nnUNetTrainerUMambaEnc
```

**Synthetic Data Generation**

```bash
cd Models/Synthetic Minority Data Augmentation/Diffusion_UKAN/training_scripts
bash HE.sh
```

**Classification Training**

```bash
cd Models/Automated Hematoma Classification/ViT
python train.py
```

---

## Inference

**Segmentation Inference**

```bash
cd Models/Automated Hematoma Segmentation/U-Mamba/umamba/nnunetv2/inference
python predict_from_raw_data.py
```

**Classification Inference**

```bash
cd Models/Automated Hematoma Classification/ViT
python inference_HE.py   # for HE prediction
python inference_NHE.py  # for NHE prediction
```

---

## Directory Structure

```
Models/                           # U-Mamba, Diffusion-UKAN, and ViT models
Preprocessing_resample.ipynb      # Resample NCCT images to uniform voxel spacing
crop2DROI.ipynb                   # Extract 2D rectangular ROIs from axial slices
Hematoma Volume Calculation.ipynb # Segmentation-based hematoma volume calculation
requirements.txt                  # Python dependencies
```

