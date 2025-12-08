# Insurance Risk Analysis & Predictive Modeling

A comprehensive data pipeline and exploratory analysis project for a large-scale South African vehicle insurance dataset (18 Months). The goal is to understand risk drivers, detect anomalies, and lay the foundation for predictive modeling (e.g., claim probability, premium pricing, fraud detection).

## Project Structure

## Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── insurance_eda.ipynb
│   ├── predictive_models.ipynb      # claim probability + severity models
│   └── ab_hypothesis_testing.ipynb   # statistical risk-difference tests
├── scripts/
│   ├── constants.py
│   └── utils/
│       └── parser.py
├── src/
│   ├── data/
│   │   └── manager.py
│   ├── preparer.py
│   └── modeling.py                     # ClaimClassifier & ClaimSeverityRegressor (RF, XGBoost, Linear)
├── tests/
├── requirements.txt
└── README.md
```

## Key Features

| Component                         | Description                                                                                                                         |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| DataManager (src/data/manager.py) | One-liner dm.load_csv(load_clean=True) → fully cleaned DataFrame with outlier flags, correct dtypes, no duplicates                  |
| Smart imputation & cleaning       | Critical columns → drop, categorical → mode, numeric specs → median, logical defaults                                               |
| Outlier flagging                  | Boolean columns Outlier_TotalPremium, Outlier_TotalClaims, Outlier_CustomValueEstimate                                              |
| DataPreparer (src/preparer.py)    | Ready-made preprocessors for linear regression (log-transform target), tree-based models and binary classification                  |
| Modeling (src/modeling.py)        | • ClaimClassifier – Random Forest (balanced classes) • ClaimSeverityRegressor – XGBoost / RF / Linear • Built-in SHAP summary plots |
| Predictive Models Notebook        | Trains & evaluates claim-probability (ROC-AUC ≈ 0.68) and claim-severity models (XGBoost best RMSE ≈ 34k)                           |
| A/B Hypothesis Testing Notebook   | Chi-square & t-tests on risk differences across provinces, postal codes and gender (all null hypotheses failed to be rejected)      |
| Interpretability                  | SHAP value computation + summary plots for tree-based models                                                                        |

## Quick Start

### Clone the repository

git clone https://github.com/Elshaday97/insurance-risk-analysis-week3.git
cd insurance-risk-analysis-week3

### Set up environment

python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

### Explore the data

jupyter lab notebooks/insurance_eda.ipynb

### Run statistical hypothesis tests

jupyter lab notebooks/ab_hypothesis_testing.ipynb

### Run predictive modeling

jupyter lab notebooks/predictive_models.ipynb
