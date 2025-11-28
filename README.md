# 🏥 Clinical Readmission Prediction Pipeline (XGBoost)

This project focuses on building an end-to-end Machine Learning pipeline in Azure ML to predict 30-day hospital readmission risk (binary classification) using clinical data.

## 🎯 Goal
To develop a robust model that can accurately predict 30-day hospital readmission, specifically improving the identification of the **minority class** (readmitted patients), as indicated by a poor baseline **Balanced Accuracy (BACC)**.

## 🛠️ Pipeline Architecture
The solution is orchestrated via an Azure ML Pipeline (`4_pipeline_build.ipynb`) consisting of chained components:

1.  **ICD Enrichment (`enrich.py`):** Takes raw data, performs feature engineering, and creates the binary target variable (`readmitted_30d_binary`) and a `higha1c` risk indicator.
2.  **Data Preparation (`prep.py`):** Takes enriched data and performs a stratified `train_test_split` to create `train.csv` and `test.csv`.
3.  **Training (`train.py`):** Trains an XGBoost model, logging metrics (`auc_custom`, `balanced_accuracy_custom`) via MLflow.

## 📊 Model Performance Comparison (Checkpoint: Nov 27, 2025)

The table below summarizes the performance of models trained using the current data and training scripts.

| Run Type | Primary Metric (AUC) | **Balanced Accuracy (BACC)** | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Pipeline (XGBoost Default)** | 0.68931 | **0.50688** | Untuned XGBoost defaults. Very low BACC confirms class imbalance issue. |
| **XGBoost Sweep Best (Manual Tuning)** | 0.69256 | TBD | Tuned XGBoost on AUC. Marginal improvement over baseline. |
| **AutoML Best (Voting Ensemble)** | **0.70736** | 0.50336 | Achieved the highest AUC, but still shows failure on minority class (BACC near 0.50). |

***Key Finding: The Balanced Accuracy near 0.50 is unacceptable for a risk model and requires immediate attention to address the class imbalance.***

## ⚙️ Hyperparameter Tuning Details (Best Manual Sweep)

| Parameter Sampling | Primary Metric | Best Trial AUC | Git Commit |
| :--- | :--- | :--- | :--- |
| **RANDOM** | `auc_custom` (Maximize) | 0.69256 | `dc0d9d1a98c9675677ccfe04fdd534345fa64d32` |

| Parameter | Space Defined |
| :--- | :--- |
| `max_depth` | Choice of [3, 5, 7, 9] |
| `learning_rate` | Uniform range [0.01, 0.3] |
| `n_estimators` | Choice of [100, 200, 400] |
| `subsample` | Uniform range [0.7, 1] |
| `colsample_bytree` | Uniform range [0.7, 1] |

## ⏭️ Next Steps (Focus: Imbalance Mitigation & Deployment)

1.  **Implement Class Weighting:** Modify `train.py` to use `scale_pos_weight` in XGBoost to combat the class imbalance.
2.  **Targeted Sweep:** Run a new sweep to optimize for **Balanced Accuracy** using the new class weighting.
3.  **Deployment:** Deploy the best-performing, high-BACC model to an Azure ML Online Endpoint.
4.  **Responsible AI:** Generate model explanations (SHAP/LIME) for transparency.