# Heart Disease Classification Using ML Classification Model

## Problem Statement

This project aims to predict the presence of heart disease in a patient based on clinical and demographic attributes. The dataset used is the Heart Disease Dataset sourced from Kaggle, consisting of 1,024 records and 13 features. Early and accurate prediction of heart disease can assist medical professionals in identifying high-risk patients and taking timely preventive measures.


## Dataset Description

| Property | Value |
|---|---|
| **Name** | Heart Disease Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/sintariosatya/heart-disease-dataset) |
| **Records** | 1,024 |
| **Features** | 13 (after dropping `num` (used for multiclass classification)) |
| **Target** | `target_binary` — 0 = No Disease, 1 = Disease |
| **Class Distribution** | 554 No Disease / 470 Disease |
| **Missing Values** | None |
| **Problem Type** | Binary Classification |

### Feature Description

| Feature | Description |
|---|---|
| `age` | Age of the patient in years |
| `sex` | Gender of the patient (1 = Male, 0 = Female) |
| `cp` | Chest pain type (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic) |
| `trestbps` | Resting blood pressure in mm Hg |
| `chol` | Serum cholesterol level in mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restecg` | Resting ECG results (0 = Normal, 1 = ST-T abnormality, 2 = Left ventricular hypertrophy) |
| `thalach` | Maximum heart rate achieved during exercise |
| `exang` | Exercise induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (1 = Upsloping, 2 = Flat, 3 = Downsloping) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversible defect) |

## GitHub Repository Link

🔗 [https://github.com/prashant67singh/ml-classification-models](https://github.com/prashant67singh/ml-classification-models)

## Live Streamlit App

🚀 [https://ml-classification-models-5xhcnfbrkavbh3wbk3w8ca.streamlit.app/](https://ml-classification-models-5xhcnfbrkavbh3wbk3w8ca.streamlit.app/)

**How to use:**
1. Select a model from the dropdown
2. Upload `test_data.csv` from the sidebar
3. View evaluation metrics, confusion matrix and classification report
4. Switch to "All Models Comparison" tab to compare all 5 models


## Models Used — Comparison Table

All 6 models were trained on the same dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8585** | 0.9251 | **0.8571** | 0.8298 | **0.8432** | **0.7147** |
| Decision Tree | 0.7610 | 0.7369 | 0.7711 | 0.6809 | 0.7232 | 0.5174 |
| KNN | 0.8439 | 0.9057 | 0.8523 | 0.7979 | 0.8242 | 0.6853 |
| Naive Bayes | 0.8439 | 0.9131 | 0.8298 | 0.8298 | 0.8298 | 0.6856 |
| Random Forest (Ensemble) | 0.8537 | **0.9333** | 0.8404 | **0.8404** | 0.8404 | 0.7053 |


## Model Observations
 
| ML Model | Observation |
|---|---|
| **Logistic Regression** | Best accuracy (0.8585) and highest precision (0.8571). Strong linear baseline for this dataset. |
| **Decision Tree** | Weakest performer — lowest AUC (0.7369) and recall (0.6809). Likely overfit the training data. |
| **KNN** | Solid accuracy (0.8439) and AUC (0.9057) with scaled features. Slightly lower recall than others. |
| **Naive Bayes** | Balanced Precision = Recall = F1 = 0.8298. Second highest AUC (0.9131). Fast and effective. |
| **Random Forest** | Highest AUC (0.9333) and balanced metrics. Best overall model for this dataset. |
| **Overall Winner** | **Random Forest** — highest AUC (0.9333) and balanced Precision/Recall/F1 (0.8404). |
 

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/prashant67singh/ml-classification-models
cd ml-classification-models

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```