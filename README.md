# 🏥 Clinical Readmission Prediction Pipeline (XGBoost)

This project focuses on building an end-to-end Machine Learning pipeline in Azure ML to predict 30-day hospital readmission risk (binary classification) using clinical data.

## 🎯 Goal
To develop a robust model that can accurately predict 30-day hospital readmission, specifically improving the identification of the **minority class** (readmitted patients), as assessed by the **Balanced Accuracy (BACC)** metric.

## 🛠️ Pipeline Architecture
The solution is orchestrated via an Azure ML Pipeline (`4_pipeline_build.ipynb`) consisting of chained components:

1.  **ICD Enrichment (`enrich.py`):** Feature engineering and creation of the binary target variable (`readmitted_30d_binary`).
2.  **Data Preparation (`prep.py`):** Stratified `train_test_split`.
3.  **Training (`train.py`):** Trains the XGBoost model using **Class Weighting** and logs `balanced_accuracy_custom` via MLflow.

---

## 📊 Final Production Model Results

The following metrics represent the performance of the **final, single model** trained on Version 3 data using the optimized hyperparameters found during the refinement sweep.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Balanced Accuracy (BACC)** | **0.6325** | **Primary Success Metric.** The model is $\approx 13\%$ better than random guessing at identifying high-risk patients. |
| **AUC** | **0.6824** | Indicates good overall ranking ability. |
| **Accuracy** | **0.6610** | Lower than the baseline (0.88) because the model is no longer ignoring the minority class to inflate the score. |

**Job Details:**
* **Run ID:** `polite_shark_hhgcfklck5`
* **Status:** Completed (Nov 28, 2025)
* **Training Duration:** 3m 30s

---

## ⚙️ Production Hyperparameters

These are the exact settings used to train the final model (`clinical-readmission-xgb:2`):

| Parameter | Value | Role in Optimization |
| :--- | :--- | :--- |
| **`scale_pos_weight`** | **8.593** | **Critical:** Penalty applied to misclassifying positive cases (Readmission). |
| `max_depth` | **9** | Allowed the model to learn complex, non-linear patterns. |
| `learning_rate` | **0.04104** | Slow, careful learning rate prevented overfitting. |
| `n_estimators` | **200** | Sufficient iterations for the gradient boosting to converge. |
| `subsample` | 0.80967 | Fraction of samples used per tree (reduces variance). |
| `colsample_bytree` | 0.9402 | Fraction of features used per tree. |

---

## 📊 Tuning History: Model Performance Comparison

The table below summarizes the performance improvements achieved throughout the project after diagnosing and correcting the severe class imbalance problem.

| Run Type | Primary Metric (AUC) | **Balanced Accuracy (BACC)** | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Pipeline (Unweighted)** | 0.68931 | **0.50688** | Initial benchmark. Model struggled severely with minority class ($\approx$ random guessing). |
| **XGBoost Sweep Best (AUC Goal)** | 0.69256 | TBD | Original sweep goal was incorrect for imbalanced data. |
| **AutoML Best (Voting Ensemble)** | 0.70736 | 0.50336 | Best overall AUC, but BACC confirmed its failure on minority class. |
| **XGBoost Sweep Best (BACC Goal)** | **0.68220** | **0.63507** | **Best Sweep Result.** Achieved a $\mathbf{+12.8\%}$ improvement in BACC through targeted tuning. |

---

## ⏭️ Project Roadmap & Next Steps

1.  **Final Model Training:** (Completed) Single model trained on Version 3 data.
2.  **Responsible AI (RAI) Analysis:** (In Progress) Generating **SHAP** feature importance and Error Analysis using the `azureml-rai` components.
3.  **Deployment:** Deploy the registered model `clinical-readmission-xgb:2` to an Azure ML Online Endpoint for real-time inference.