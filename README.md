# SeleneX Assignment 5 – Ovarian Ultrasound + Biomarker Diagnostic Pipeline

## Overview
This project builds a **mini-prototype diagnostic pipeline** that combines:
- **Ultrasound image classification** (via CNNs + Grad-CAM explainability)
- **Biomarker-based prediction** (CA-125, BRCA, age features + SHAP explainability)
- **Fusion model** (jointly leveraging both modalities)

Dataset: MMOTU ovarian ultrasound dataset (8 classes, collapsed into benign/malignant).


## Setup
```bash
# Create environment
conda create -n lunartech python=3.10 -y
conda activate lunartech

# Install requirements
pip install -r requirements.txt

---

## Prepare Data
```bash
python make_test_split.py
python build_manifest_mmOTU.py


## Run Notebooks
01_data_preparation.ipynb → builds parquet splits
02_train_and_explain.ipynb → trains models + generates explainability plots


## Outputs
data/manifest.csv
Grad-CAM heatmaps → assets/gradcam/
SHAP plots → assets/shap/
ROC/CM plots + metrics.json → assets/metrics/
