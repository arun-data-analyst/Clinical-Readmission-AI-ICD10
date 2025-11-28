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

## 📊 Model Performance Comparison (Checkpoint: Final Optimized Model)

The table below summarizes the performance improvements achieved after diagnosing and correcting the severe class imbalance problem.

| Run Type | Primary Metric (AUC) | **Balanced Accuracy (BACC)** | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline Pipeline (Unweighted)** | 0.68931 | **0.50688** | Initial benchmark. Model struggled severely with minority class ($\approx$ random guessing). |
| **XGBoost Sweep Best (AUC Goal)** | 0.69256 | TBD | Original sweep goal was incorrect for imbalanced data. |
| **AutoML Best (Voting Ensemble)** | 0.70736 | 0.50336 | Best overall AUC, but BACC confirmed its failure on minority class. |
| **XGBoost Final Best (BACC Goal + Weighting)** | **0.68220** | **0.63507** | **FINAL BEST MODEL.** Achieved a $\mathbf{+12.8\%}$ improvement in BACC through targeted tuning. |

***Key Result: The final BACC of 0.63507 confirms the success of the class weighting strategy, making the model reliably better than random at predicting readmission risk.***

---

## ⚙️ Final Model Hyperparameters (for Production)

This configuration achieved the final best Balanced Accuracy of **0.63507**. This is the exact parameter set that will be used for the final production model training.

| Parameter | Final Best Value (Rounded) | Optimization Insight |
| :--- | :--- | :--- |
| **BACC Score** | **0.63507** | $\mathbf{+12.8\%}$ Improvement over baseline. |
| **`scale_pos_weight`** | $\mathbf{8.593}$ | Optimal penalty found to be close to the theoretical ratio of 7.96. |
| `max_depth` | 9 | Confirms that a complex, deeper model was required. |
| `learning_rate` | 0.04104 | A stable, low learning rate was optimal for convergence. |
| `n_estimators` | 200 | Sufficient number of trees for convergence. |
| `subsample` | 0.80967 | (Fixed) |
| `colsample_bytree` | 0.94020 | (Fixed) |
| **Git Commit** | $\mathbf{7f8c71e477c699e12975105541658ed98fec374f}$ | Source code version used for the best run. |

---

## ⏭️ Tomorrow's Steps (Finalization & Deployment)

1.  **Final Model Training:** Use the exact hyperparameters above to train a single, definitive production model.
2.  **Responsible AI (RAI) Analysis:** Generate the **Model Explanation (SHAP/LIME)** and the **RAI Dashboard** for clinical transparency.
3.  **Deployment:** Deploy the final, validated model to an Azure ML Online Endpoint.

---

**Action:** Update your `README.md` with the text above, commit the change, and push to GitHub. You've completed a major project milestone! See you tomorrow to finish the job!