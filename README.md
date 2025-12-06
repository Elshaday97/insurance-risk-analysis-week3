# Insurance Risk Analysis & Predictive Modeling

A comprehensive data pipeline and exploratory analysis project for a large-scale South African vehicle insurance dataset (18 Months). The goal is to understand risk drivers, detect anomalies, and lay the foundation for predictive modeling (e.g., claim probability, premium pricing, fraud detection).

## Project Structure

## Repository Structure

```txt
├── data/
│   ├── processed
│   ├── raw
├── notebooks/
│   ├── insurance_eda.ipynb
├── scripts/
│   ├── __init__.py
│   ├── constants.py
│   ├──utils/
│   │   └── parser.py
├── src/
│   ├── data/
│   │   └── __init__.py
│   │   └── manager.py
├── tests/
└── README.md
```

## Key Features

- **Robust DataManager** (`src/data/manager.py`)

  - Loads raw pipe-separated file or cleaned CSV
  - Comprehensive cleaning pipeline (missing values, dtype conversion, outlier flagging, duplicate removal, non-negative enforcement)
  - One-line clean data loading: `dm.load_data()`
  - Automatically saves cleaned dataset

- **Smart Missing Value Handling**

  - Critical columns → drop rows
  - Categorical → mode imputation
  - Numeric specs → median
  - Logical defaults (0, False, "Unknown")

- **Outlier Detection & Flagging**

  - Adds boolean flags: `Outlier_TotalPremium`, `Outlier_TotalClaims`, etc.

- **EDA Notebook**
  - Full statistical summary
  - Distributions, correlations, time-series trends
  - Ready for modeling iteration

## Quick Start

```bash
# Clone the repo
git clone git@github.com:Elshaday97/insurance-risk-analysis-week3.git
cd insurance-risk-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the EDA notebook
jupyter notebook notebooks/insurance_eda.ipynb
```
