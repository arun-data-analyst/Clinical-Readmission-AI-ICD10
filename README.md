# 🏥 Clinical Readmission Prediction Pipeline (XGBoost)

This project focuses on building an end-to-end Machine Learning pipeline in Azure ML to predict 30-day hospital readmission risk (binary classification) using clinical data.

## 🎯 Goal
To develop a robust model that can accurately predict 30-day hospital readmission, specifically improving the identification of the **minority class** (readmitted patients), as indicated by the **Balanced Accuracy (BACC)** metric.

## 🛠️ Pipeline Architecture
The solution is orchestrated via an Azure ML Pipeline (`4_pipeline_build.ipynb`) consisting of chained components:

1.  **ICD Enrichment (`enrich.py`):** Feature engineering and creation of the binary target variable (`readmitted_30d_binary`).
2.  **Data Preparation (`prep.py`):** Stratified `train_test_split`.
3.  **Training (`train.py`):** Trains the XGBoost model using **Class Weighting** and logs `balanced_accuracy_custom` via MLflow.

---

## 📊 Model Performance Comparison (Checkpoint: Post Class Weighting Sweep)

The table below summarizes the critical shift in performance after implementing the class weighting strategy.

| Run Type | Primary Metric (AUC) | **Balanced Accuracy (BACC)** | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Pipeline (Unweighted)** | 0.68931 | **0.50688** | Initial benchmark. Model struggled severely with minority class. |
| **XGBoost Sweep Best (BACC Goal + Weighting)** | **0.68006** | **0.62805** | **SUCCESS!** Achieved $\mathbf{+12.1\%}$ improvement in BACC by balancing class errors. |
| **AutoML Best (Voting Ensemble)** | 0.70736 | 0.50336 | Best overall AUC, but BACC confirms its failure on the minority class. |

***Key Finding: The Balanced Accuracy increase to 0.62805 confirms that class weighting is the necessary correction for this risk prediction task.***

---

## ⚙️ Hyperparameter Tuning Details (Best BACC-Optimized Trial)

This configuration achieved the highest Balanced Accuracy of **0.62805**:

| Parameter | Best Value | Optimization Role |
| :--- | :--- | :--- |
| **`scale_pos_weight`** | **10.0** | **Crucial for correcting the imbalance (BACC).** |
| `max_depth` | 9 | Deeper, more complex trees were favored. |
| `learning_rate` | 0.03229 | Slower, more careful learning. |
| `n_estimators` | 200 | Standard number of trees. |
| `subsample` | 0.80967 | |
| `colsample_bytree` | 0.94020 | |

---

## ⏭️ Next Steps (Focus: Final Refinement & Deployment)

1.  **Commit:** Commit these updated documentation and code changes to the repository.
2.  **Refined Sweep:** Run a **final, focused sweep** using a tight search range around the best hyperparameters (`scale_pos_weight` in Uniform [8.0, 13.0], `max_depth` [8, 9, 10], `learning_rate` [0.01, 0.05]) to push the BACC higher.
3.  **Finalize Model:** Train one final production model using the absolute best hyperparameters found.
4.  **Deployment & RAI:** Deploy the final model to an Azure ML Endpoint and generate Responsible AI explanations.