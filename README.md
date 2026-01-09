<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18194376">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.18194376.svg" alt="DOI"/>
  </a>
  <img src="https://img.shields.io/badge/Accuracy-97.37%25-success?style=for-the-badge&logo=python&logoColor=white" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Precision-100%25-brightgreen?style=for-the-badge" alt="Precision"/>
  <img src="https://img.shields.io/badge/Recall-92.86%25-blue?style=for-the-badge" alt="Recall"/>
  <img src="https://img.shields.io/badge/ROC--AUC-99.24%25-blueviolet?style=for-the-badge" alt="ROC-AUC"/>
</p>

<h1 align="center">VitaShield AI</h1>
<p align="center">
  <strong>Achieving Near-Perfect Precision in Breast Cancer Diagnosis with Tuned LightGBM and SHAP Interpretability</strong><br>
  DOI: <a href="https://doi.org/10.5281/zenodo.18194376">10.5281/zenodo.18194376</a>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#key-features">Features</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

## About

**VitaShield AI** is a comprehensive machine learning project for high-precision breast cancer diagnosis using the Wisconsin Breast Cancer Diagnostic (WBCD) dataset. It achieves 97.37% accuracy and 100% precision with a tuned LightGBM model, SMOTE for imbalance handling, and SHAP for explainability.

Author:  
Agha Wafa Abbas  
Lecturer, School of Computing, University of Portsmouth, United Kingdom  
Lecturer, School of Computing, Arden University, Coventry, United Kingdom  
Lecturer, School of Computing, Pearson, London, United Kingdom  
Lecturer, School of Computing, IVY College of Management Sciences, Lahore, Pakistan  
Emails: agha.wafa@port.ac.uk , awabbas@arden.ac.uk, wafa.abbas.lhr@rootsivy.edu.pk  

This project is licensed under the MIT License.

The repository includes the full Jupyter notebook (breastcancer.ipynb), dataset (Cancer_Data.csv), pre-trained models, figures, and a research paper PDF.

## Key Features

- Exploratory Data Analysis (EDA) with visualizations
- SMOTE oversampling for class balance
- Baseline model comparison across 5 algorithms
- GridSearchCV hyperparameter tuning
- LightGBM classifier with superior performance
- SHAP interpretability (beeswarm and feature importance)
- Confusion matrix and performance metrics
- Reproducible Jupyter notebook
- Research paper draft

## Results

| Metric          | Value     | Description                                      |
|-----------------|-----------|--------------------------------------------------|
| Accuracy        | 97.37%    | Overall correct predictions                      |
| Precision       | 100.00%   | No false positives for malignant class           |
| Recall          | 92.86%    | 39/42 malignant cases identified                 |
| F1-Score        | 96.30%    | Harmonic mean of precision and recall            |
| ROC-AUC         | 99.24%    | Excellent class separation                       |

Top 10 Important Features (by SHAP):  
area_worst (2.789), perimeter_worst (2.778), concave points_worst (1.782), texture_worst (1.722), concave points_mean (1.526), concavity_worst (1.339), area_se (1.224), texture_mean (1.053), smoothness_mean (0.765), symmetry_worst (0.676)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/VitaShield-AI.git
cd VitaShield-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Usage
jupyter notebook breastcancer.ipynb

# Quick model inference example
import joblib
import numpy as np

model = joblib.load('VitaShield_AI_LightGBM_model.pkl')
scaler = joblib.load('VitaShield_AI_scaler.pkl')

# Sample input (30 features)
new_data = np.array([[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
print("Prediction:", "Malignant" if prediction[0] == 1 else "Benign")

# Project Structure 
VitaShield-AI/
├── Cancer_Data.csv
├── VitaShield_AI_LightGBM_model.pkl
├── VitaShield_AI_scaler.pkl
├── breastcancer.ipynb
├── VitaShield AI Achieving Near-Perfect Precision in Breast Cancer Diagnosis with Tuned LightGBM and SHAP Interpretability.pdf
├── requirements.txt
├── README.md
└── LICENSE

@article{abbas2026vitashield,
  title = {VitaShield AI: Achieving Near-Perfect Precision in Breast Cancer Diagnosis with Tuned LightGBM and SHAP Interpretability},
  author = {Abbas, Agha Wafa},
  year = {2026},
  doi = {10.5281/zenodo.18194376},
  url = {https://doi.org/10.5281/zenodo.18194376}
}


