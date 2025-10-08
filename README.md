# An End-to-End Deep Learning Pipeline for Hematoma Expansion Prediction in Spontaneous Intracerebral Hemorrhage Based on Non-Contrast Computed Tomography

---

## Authors
Qiang Yu, Xin Fan, Jinwei Li, Qianyu Hao, Youquan Ning, Shichao Long, Wenhao Jiang, Fajin Lv, Xianlei Yan, Quan Liu, Xiaoquan Xu, Zongqian Wu, Juan Peng, and Min Wu

---

## Overall Study Design

The modular pipeline comprises three sequential stages: **automated hematoma segmentation**, **synthetic minority data augmentation**, and **automated hematoma classification**.

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

## Directory Structure

```

Models/                           # Contains the Full-Scale Segmentation U-Mamba model,
# Synthetic Minority Data Augmentation Diffusion-UKAN model,
# and Automated Hematoma Classification Vision Transformer model

Preprocessing_resample.ipynb      # Resamples all NCCT images to uniform voxel spacing (0.5 × 0.5 × 5 mm³)
# using linear interpolation for standardized spatial resolution

crop2DROI.ipynb                   # Extracts 2D rectangular ROIs from three consecutive axial NCCT slices for each hematoma

Hematoma Volume Calculation.ipynb # Implements segmentation-based hematoma volume calculation

Requirements.txt                  # List of required Python packages

```



