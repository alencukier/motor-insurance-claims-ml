# Motor Insurance Claims Modeling: GLM vs. Tree-Based ML

## Executive Summary
This project builds and evaluates predictive models for claim frequency and severity in motor insurance using the French Motor Third-Party Liability dataset (`freMTPL2`).

The core objective is to compare traditional actuarial pricing methodologies (Generalized Linear Models with Poisson and Tweedie distributions) against modern Gradient Boosted Decision Trees (XGBoost and LightGBM), evaluating trade-offs in predictive performance, exposure management, and model interpretability.

## Key Technical Objectives
* **Actuarial Baseline:** Fit a Poisson GLM incorporating log-exposure ($\log(\text{Exposure})$) as a canonical offset.
* **Machine Learning Benchmark:** Train XGBoost and LightGBM using specialized Poisson loss functions to respect exposure constraints.
* **Model Explainability:** Apply SHAP (SHapley Additive exPlanations) to evaluate global feature importance and partial dependence.
* **MLOps & Tracking:** Log parameters, metrics, and models using MLflow within Databricks.

## Repository Structure
```text
motor-insurance-claims-ml/
├── data/                  # Local data directory (ignored by git)
│   ├── raw/               # Original freMTPL2 dataset
│   └── processed/         # Joined and transformed features
├── notebooks/             # Sequential analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_glm_baseline.ipynb
│   └── 03_ml_models.ipynb
├── src/                   # Reusable Python modules
│   ├── __init__.py
│   └── utils.py
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
