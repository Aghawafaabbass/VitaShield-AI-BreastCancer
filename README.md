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
  <strong>Near-Perfect Precision in Breast Cancer Diagnosis using Tuned LightGBM & SHAP Interpretability</strong><br>
  DOI: <a href="https://doi.org/10.5281/zenodo.18194376">10.5281/zenodo.18194376</a>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#citation">Citation</a> •
  <a href="#license">License</a>
</p>

---

## About

**VitaShield AI** is a fully reproducible machine learning pipeline for **interpretable and accurate breast tumor classification** (benign vs malignant) using the **Wisconsin Breast Cancer Diagnostic (WBCD) dataset**.

**Highlights:**
- ✅ **Accuracy:** 97.37%  
- ✅ **Precision:** 100% (zero false positives for malignant cases)  
- ✅ **Recall:** 92.86%  
- ✅ **ROC-AUC:** 99.24%  
- ✅ **Model Interpretability:** SHAP (SHapley Additive exPlanations)  

**Methodology:**
- SMOTE for class imbalance correction  
- GridSearchCV for hyperparameter optimization  
- Publication-quality visualizations: confusion matrix, SHAP beeswarm, feature importance  

Perfect for **researchers, students, and AI practitioners in healthcare** looking to explore **explainable AI** in medical diagnostics.

**Zenodo DOI:** [10.5281/zenodo.18194376](https://doi.org/10.5281/zenodo.18194376)

---

## Results

| Metric          | Value     | Description                                      |
|-----------------|-----------|--------------------------------------------------|
| Accuracy        | 97.37%    | Overall correct predictions                      |
| Precision       | 100.00%   | Zero false positives for malignant class         |
| Recall          | 92.86%    | 39/42 malignant cases correctly identified       |
| F1-Score        | 96.30%    | Harmonic mean of precision and recall            |
| ROC-AUC         | 99.24%    | Excellent class separation                       |

**Top 3 Important Features (by mean |SHAP| value):**  
1. `area_worst`  
2. `perimeter_worst`  
3. `concave points_worst`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Aghawafaabbass/VitaShield-AI.git
cd VitaShield-AI

# (Recommended) Create & activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
Usage
1. Run the complete analysis notebook
bash
Copy code
jupyter notebook notebooks/VitaShield_AI_Complete.ipynb
2. Quick inference with pre-trained model
python
Copy code
import joblib
import numpy as np

# Load saved model & scaler
model = joblib.load('models/VitaShield_AI_LightGBM_model.pkl')
scaler = joblib.load('models/VitaShield_AI_scaler.pkl')

# Example new patient data (30 features in correct order)
new_patient = np.array([[17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760, 0.30010, 0.14710,
                         0.24190, 0.07871, 1.0950, 0.9053, 8.5890, 153.40, 0.006399, 0.04904,
                         0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.60, 2019.0,
                         0.16220, 0.66560, 0.71190, 0.26540, 0.46010, 0.11890]])

# Scale & predict
scaled = scaler.transform(new_patient)
prediction = model.predict(scaled)[0]
probability = model.predict_proba(scaled)[0][1]

print("Diagnosis:", "Malignant" if prediction == 1 else "Benign")
print(f"Malignant Probability: {probability:.2%}")
Project Structure
kotlin
Copy code
VitaShield-AI/
├── data/
│   └── Cancer_Data.csv
├── models/
│   ├── VitaShield_AI_LightGBM_model.pkl
│   └── VitaShield_AI_scaler.pkl
├── notebooks/
│   └── VitaShield_AI_Complete.ipynb
├── figures/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── shap_beeswarm.png
│   └── top_shap_features.png
├── docs/
│   └── VitaShield_AI_Research_Paper.pdf
├── requirements.txt
├── README.md
└── LICENSE
Citation
If you use this work, please cite:

bibtex
Copy code
@misc{abbas2026vitashield,
  author       = {Agha Wafa Abbas},
  title        = {VitaShield AI: Achieving Near-Perfect Precision in Breast Cancer Diagnosis with Tuned LightGBM and SHAP Interpretability},
  year         = {2026},
  doi          = {10.5281/zenodo.18194376},
  publisher    = {Zenodo},
  howpublished = {\url{https://doi.org/10.5281/zenodo.18194376}},
  note         = {GitHub repository also available at \url{https://github.com/Aghawafaabbass/VitaShield-AI}}
}
License
Distributed under the MIT License. See LICENSE for more information.
