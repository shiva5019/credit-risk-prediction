# Credit Risk Default Prediction 🏦

## Overview
This project builds an end-to-end machine learning pipeline to predict whether a borrower will default on a loan. Using a Lending Club style dataset of 54,231 loans, we tackle a real-world binary classification problem with severe class imbalance (87% non-default vs 13% default). The project demonstrates the complete data science workflow — from raw data exploration to a production-ready model with threshold tuning.

---

## Problem Statement
Financial institutions lose billions every year due to loan defaults. Early identification of high-risk borrowers allows lenders to make better lending decisions, reduce losses, and offer appropriate interest rates. This project predicts loan default risk using borrower financial profiles, loan details, and repayment behavior data.

---

## Dataset
- Source: Lending Club Loan Dataset
- Total Rows: 54,231
- Raw Columns: 43
- Columns after Preprocessing: 21
- Target Variable: Binary (0 = Non-Default, 1 = Default)
- Default Rate: 12.8% (severe class imbalance)
- Key Features: loan_amount, int_rate, grade, term, annual_inc, dti, employment_length, purpose, home_ownership, verification_status

---

## Project Structure

```
credit-risk-prediction/
│
├── data/
│   ├── raw/
│   │   └── loan_payments.csv
│   └── processed/
│       └── loan_payments_preprocessed.csv
│
├── models/
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_building.ipynb
│   └── 04_final_evaluation.ipynb
│
└── README.md
```

---

## Notebooks Walkthrough

### 01_eda.ipynb — Exploratory Data Analysis
- Loaded raw dataset and inspected shape, dtypes, null values
- Created binary target column from loan_status
- Analyzed class imbalance — 87.2% Non-Default vs 12.8% Default
- Plotted numerical features (loan_amount, int_rate, annual_inc, dti) vs target using boxplots
- Plotted categorical features (grade, purpose, home_ownership, term) default rates using barplots
- Generated correlation heatmap to identify multicollinearity
- Key finding — int_rate and grade are strongest predictors, G grade loans default at 32% vs A grade at 5%

### 02_preprocessing.ipynb — Data Preprocessing
- Dropped high null columns (>50% missing) — mths_since_last_record, mths_since_last_major_derog, next_payment_date, mths_since_last_delinq
- Dropped identifier columns — id, member_id
- Dropped zero variance column — policy_code
- Dropped date columns — issue_date, earliest_credit_line, last_payment_date, last_credit_pull_date
- Dropped redundant correlated columns — funded_amount, funded_amount_inv, out_prncp_inv, total_payment_inv
- Dropped post-default leakage columns — recoveries, collection_recovery_fee, total_rec_late_fee
- Imputed nulls using lambda + pd.isna() to handle Arrow dtype issues
- Encoded term and employment_length ordinally
- Encoded grade ordinally (A=1 to G=7)
- Label encoded home_ownership, verification_status, purpose
- Applied log1p transformation to highly skewed columns
- Saved final preprocessed CSV with 21 features

### 03_model_building.ipynb — Model Building
- Split data into Train (70%) / Validation (15%) / Test (15%) with stratification
- Applied 4 sampling strategies — SMOTE, SMOTETomek, Scaled+SMOTE, Scaled+SMOTETomek
- Trained 4 models — Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluated using precision, recall, F1 score, ROC-AUC on validation set
- Selected XGBoost with Scaled+SMOTE as best model (ROC-AUC 0.9820)
- Saved best model and scaler as pkl files

### 04_final_evaluation.ipynb — Final Evaluation
- Loaded saved XGBoost model and scaler
- Found optimal threshold using precision-recall curve (0.3048 vs default 0.5)
- Threshold tuning improved recall from 0.85 to 0.87 on validation set
- Evaluated final model on unseen test set
- Plotted confusion matrix, ROC curve, feature importance
- Business conclusion and model interpretation

---

## Models Tried

| Model | Sampling | Recall (Default) | Precision (Default) | F1 (Default) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.81 | 0.47 | 0.59 | 0.9100 |
| Logistic Regression | SMOTETomek | 0.65 | 0.25 | 0.36 | 0.7445 |
| Logistic Regression | Scaled + SMOTE | 0.85 | 0.67 | 0.75 | 0.9525 |
| Logistic Regression | Scaled + SMOTETomek | 0.85 | 0.67 | 0.75 | 0.9525 |
| Random Forest | SMOTE | 0.82 | 1.00 | 0.90 | 0.9766 |
| Random Forest | SMOTETomek | 0.82 | 1.00 | 0.90 | 0.9768 |
| Random Forest | Scaled + SMOTE | 0.83 | 1.00 | 0.90 | 0.9795 |
| Random Forest | Scaled + SMOTETomek | 0.83 | 1.00 | 0.90 | 0.9806 |
| LightGBM | Scaled + SMOTE | 0.84 | 1.00 | 0.91 | 0.9813 |
| LightGBM | Scaled + SMOTETomek | 0.84 | 1.00 | 0.91 | 0.9812 |
| XGBoost | SMOTE | 0.85 | 0.99 | 0.91 | 0.9797 |
| XGBoost | SMOTETomek | 0.85 | 0.98 | 0.91 | 0.9827 |
| XGBoost | Scaled + SMOTETomek | 0.85 | 0.99 | 0.91 | 0.9826 |
| XGBoost | Scaled + SMOTE | 0.85 | 0.99 | 0.91 | 0.9820 |

---

## Best Model — XGBoost with Scaled + SMOTE

### Validation Set Performance
| Metric | Default Threshold (0.5) | Optimal Threshold (0.3048) |
|---|---|---|
| Recall (Default) | 0.85 | 0.87 |
| Precision (Default) | 0.99 | 0.97 |
| F1 Score (Default) | 0.91 | 0.92 |
| ROC-AUC | 0.9820 | 0.9820 |

### Test Set Performance (Final)
| Metric | Value |
|---|---|
| Accuracy | 98% |
| Recall (Default) | 0.85 |
| Precision (Default) | 0.95 |
| F1 Score (Default) | 0.90 |
| ROC-AUC | 0.9799 |
| Optimal Threshold | 0.3048 |

---

## Confusion Matrix — Test Set
| | Predicted Non-Default | Predicted Default |
|---|---|---|
| Actual Non-Default | 7048 (True Negative) | 46 (False Positive) |
| Actual Default | 154 (False Negative) | 887 (True Positive) |

Model catches 887 out of 1041 actual defaulters — missing only 154 high risk borrowers.

---

## Key Findings
- Grade is the strongest categorical predictor — G grade loans default at 32% vs A grade at 5%
- int_rate has 0.39 correlation with default — higher interest rates signal higher risk
- 60 month term loans default at 17.5% vs 36 month loans at 10.5%
- Small business and educational loans have highest default rates among loan purposes
- total_rec_prncp and loan_amount are top features in XGBoost feature importance
- Threshold tuning from 0.5 to 0.3048 improved recall without significantly hurting precision
- Scaled + SMOTE consistently outperformed other sampling strategies for tree based models
- SMOTE applied only on training set to avoid data leakage

---

## Class Imbalance Handling
| Strategy | Description |
|---|---|
| SMOTE | Synthetic Minority Oversampling — generates synthetic default samples |
| SMOTETomek | SMOTE + Tomek links removal — removes borderline ambiguous samples |
| Scaled + SMOTE | StandardScaler applied before SMOTE — best results for all models |
| Scaled + SMOTETomek | StandardScaler applied before SMOTETomek |

---

## Tech Stack
- Python 3.13
- pandas — data manipulation
- numpy — numerical operations
- matplotlib, seaborn — data visualization
- scikit-learn — preprocessing, model evaluation, train test split
- xgboost — gradient boosting model
- lightgbm — light gradient boosting model
- imbalanced-learn — SMOTE, SMOTETomek
- joblib — model serialization

---

## How to Run
1. Clone this repository
2. Place loan_payments.csv inside data/raw/ folder
3. Install all dependencies using pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn joblib
4. Run notebooks strictly in order — 01 → 02 → 03 → 04
5. Final model and scaler will be saved in models/ folder after running Notebook 3

---

## Business Impact
In real banking scenarios, missing a defaulter (False Negative) is far more costly than rejecting a good customer (False Positive). This model achieves:
- 85% recall on default class — catches 85 out of every 100 actual defaulters
- 95% precision — when model flags a default, it is correct 95% of the time
- Only 154 defaulters missed out of 8135 test samples
- ROC-AUC of 0.9799 — significantly better than random (0.5) and strong enough for production screening

---

## Author
J. Shiva
GitHub: shiva5019
