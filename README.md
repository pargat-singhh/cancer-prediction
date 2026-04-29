# Cancer Type Prediction Using Machine Learning and Web Application

MCA Capstone Project — Lovely Professional University, 2026

This project predicts the type of cancer from gene expression data using machine learning and provides a web application for easy access.

**Live App:** https://cancer-type-predictor01.streamlit.app/

## Dataset

The dataset is the TCGA Pan-Cancer RNA-Seq dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+RNA-Seq). It contains 801 patient samples with 20,531 gene expression values across 5 cancer types:

| Code | Cancer Type | Samples |
|------|------------|---------|
| BRCA | Breast Cancer | 300 |
| KIRC | Kidney Cancer | 146 |
| LUAD | Lung Cancer | 141 |
| PRAD | Prostate Cancer | 136 |
| COAD | Colon Cancer | 78 |

## How It Works

The gene expression data (20,531 features) is first normalized using StandardScaler, then reduced to 100 dimensions using PCA. Three models were trained and compared — Logistic Regression, Random Forest, and SVM — using 5-fold cross-validation. The best model was saved and deployed as a Streamlit web app.

## Results

| Model | CV Accuracy | Test Accuracy |
|-------|------------|---------------|
| Logistic Regression | 99.84% | 98.76% |
| SVM (RBF) | 99.38% | 98.14% |
| Random Forest | 98.13% | 97.52% |

Logistic Regression was selected as the best model. Only 2 out of 161 test samples were misclassified.

## Files in This Repository

| File | Description |
|------|-------------|
| `Cancer_Type_Prediction.ipynb` | Complete ML pipeline — data loading, EDA, PCA, model training, evaluation |
| `app.py` | Streamlit web application code |
| `requirements.txt` | Python libraries needed to run the project |
| `models/best_model.pkl` | Trained Logistic Regression model |
| `models/feature_names.csv` | List of 20,531 gene names used by the model |
| `models/cancer_classes.csv` | The 5 cancer type labels |

## Tools Used

Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Streamlit, joblib

## Note

This project is for educational purposes only and is not validated for clinical use.
