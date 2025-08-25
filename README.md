# SeleneX Assignment 5 – Ovarian Ultrasound + Biomarker Diagnostic Pipeline

This project implements a mini-prototype diagnostic pipeline that integrates ultrasound imaging with biomarker features to classify ovarian tumors as benign or malignant.

## Key components:
<b>Ultrasound Image Classifier</b> – CNN backbone + Grad-CAM explainability <br>
<b>Biomarker Tabular Model</b> – Age, CA-125, BRCA status + SHAP explainability <br>
<b>Fusion Model</b> – Combines both modalities for improved performance <br>
<b>Streamlit App</b> – Lightweight demo interface for real-time inference <br>

### Dataset: MMOTU ovarian ultrasound dataset (benign vs malignant), with synthetic biomarker features.

## Repository Structure
```
SeleneX_Assignment5_HiteshPant/
│
├── notebooks/
│ ├── 01_data_preparation.ipynb            # Data loading, preprocessing, synthetic biomarker generation
│ └── 02_train_and_explain.ipynb           # Training models + explainability (Grad-CAM, SHAP)
│
├── app/
│ └── streamlit_app.py                     # Streamlit demo interface
│
├── image_data/                            # Sample images for quick testing
│
├── Assets_Outputs/
│ └── assets/                              # Grad-CAM, SHAP, ROC/CM plots, metrics.json
│
├── docs/
│ ├── model_card.pdf
│ ├── risk_bias_log.pdf
│ ├── summary_annex.pdf
│ ├── product_requirements_annex.pdf
│ └── data_plan_annex.pdf
│
├── build_manifest_mmOTU.py
├── make_test_split.py
├── requirements.txt
├── metrics.json
└── README.md

```

## Setup Instructions
### 1. Create environment
```
conda create -n lunartech python=3.10 -y
conda activate lunartech
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

## Dataset Preparation
1. Download OTU_2D images (~1,469)
2. Place images into: data/OTU_2D/

## Build manifest & splits (adjust paths as needed):
```
python build_manifest_mmOTU.py
python make_test_split.py 
```

## Run Notebooks
01_data_preparation.ipynb → builds parquet splits
02_train_and_explain.ipynb → trains models + generates explainability plots


## Outputs
data/manifest.csv
Grad-CAM heatmaps → assets/gradcam/
SHAP plots → assets/shap/
ROC/CM plots + metrics.json → assets/metrics/
