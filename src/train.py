import argparse   # parse CLI arguments
import os         # work with folders / paths
import glob       # find files matching a pattern

import numpy as np
import pandas as pd                       # tabular data handling
from sklearn.metrics import accuracy_score, roc_auc_score # basic metrics 
from sklearn.metrics import balanced_accuracy_score 
import xgboost as xgb                     # gradient boosting model

import mlflow                             # experiment tracking
import mlflow.xgboost                     # XGBoost autolog support


DEFAULT_TARGET_COL = "readmitted_30d_binary"


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for 30-day readmission prediction."
    )

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Folder that contains train.csv (from prep step).",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Folder that contains test.csv (from prep step).",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Folder where the trained model will be saved.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help=(
            "Name of the target column. "
            "Default: readmitted_30d_binary (1 = <30, 0 = NO or >30). "
            "If set to 'readmitted', the script will binarize it (<30 vs others)."
        ),
    )

    # Hyperparameters (we keep them simple for now; later we sweep them)
    parser.add_argument("--max_depth", type=int, default=6, help="Tree depth.")
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Learning rate."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="Number of trees."
    )
    parser.add_argument(
        "--subsample", type=float, default=1.0, help="Row subsampling rate."
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=1.0,
        help="Column subsampling per tree.",
    )
    # NEW: Argument for controlling class imbalance
    parser.add_argument(
        "--scale_pos_weight",
        type=float,
        default=1.0, # None means we calculate it later based on data
        help="Controls the balance of positive and negative weights. Set to 1 for no weighting.",
)

    return parser.parse_args()


def find_csv(folder: str) -> str:
    """Return the first CSV file found under a folder (recursively)."""
    csv_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {folder}")
    return csv_files[0]


def load_dataset(folder: str) -> pd.DataFrame:
    """Load a dataset from a folder containing a single CSV."""
    csv_path = find_csv(folder)
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, na_values="?", low_memory=False)
    print(f"Shape: {df.shape}")
    return df


def binarize_readmitted(series: pd.Series) -> pd.Series:
    """
    Turn 'readmitted' labels into binary:
    1 = readmitted within 30 days ('<30')
    0 = all other values.
    """
    # ADDED: Use pd.to_numeric().fillna(0) for robustness to non-numeric issues 
    # (which we discussed regarding the 20k missing/unclassified records).
    binary_series = series.apply(lambda v: 1 if str(v).strip() == "<30" else 0)

    # Coerce to numeric, treat errors as NaN, fill NaN with 0, then cast to int
    # This robustly handles any non-numeric values in the binary column by setting them to 0.
    return pd.to_numeric(binary_series, errors='coerce').fillna(0).astype(int)

def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
):
    """
    Split into X/y, one-hot encode, and align train/test columns.

    If target_col == 'readmitted':
        - y = binarize_readmitted(readmitted)

    If target_col == 'readmitted_30d_binary':
        - y = readmitted_30d_binary as created in icd_enrich.py

    Features:
        - All columns except the target, and we always drop the original
          'readmitted' column to avoid leakage when using the binary target.
    """

    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in train data.")

    # --- 1. Separate raw target and features ---
    y_train_raw = train_df[target_col]
    y_test_raw = test_df[target_col]

    # Columns to drop from features
    drop_cols = [target_col]
    # Always drop the original 'readmitted' label if present
    if "readmitted" in train_df.columns and "readmitted" not in drop_cols:
        drop_cols.append("readmitted")

    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    # --- 2. Target mapping depending on target_col ---
    if target_col == "readmitted":
        print("Binarizing 'readmitted' column for 30-day readmission...")
        y_train = binarize_readmitted(y_train_raw)
        y_test = binarize_readmitted(y_test_raw)
    else:
        # Assume column is already binary (e.g. readmitted_30d_binary)
        print(f"Using '{target_col}' as precomputed binary target...")
        # ADDED: Ensure the precomputed binary target is robustly treated as integer 0/1
        y_train = pd.to_numeric(y_train_raw, errors='coerce').fillna(0).astype(int)
        y_test = pd.to_numeric(y_test_raw, errors='coerce').fillna(0).astype(int)

    # --- 3. One-hot encode categorical features ---
    print("One-hot encoding categorical features...")
    X_train_enc = pd.get_dummies(X_train, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, drop_first=False)

    # --- 4. Align columns across train and test (fill missing with 0) ---
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0
    )

    # --- 5. Clean up column names for XGBoost ---
    X_train_enc.columns = [
        c.replace("[", "_")
        .replace("]", "_")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(" ", "_")
        for c in X_train_enc.columns
    ]
    X_test_enc.columns = X_train_enc.columns  # keep same order

    print(f"Encoded train shape: {X_train_enc.shape}")
    print(f"Encoded test shape : {X_test_enc.shape}")

    return X_train_enc, X_test_enc, y_train, y_test


def main():
    args = parse_args()

    print("--- STARTING TRAINING ---")
    print(f"Target column: {args.target_col}")

    # 1. Load train and test data
    train_df = load_dataset(args.train_data)
    test_df = load_dataset(args.test_data)

    # 2. Prepare features and labels
    X_train, X_test, y_train, y_test = prepare_features(
        train_df, test_df, args.target_col
    )
# 3. CLASS WEIGHTING LOGIC: Calculate or use user-provided scale_pos_weight
    # -----------------------------------------------------------------------
    scale_pos_weight = args.scale_pos_weight

    if scale_pos_weight is None:
        # Calculate the weight based on the training data imbalance (Majority / Minority)
        print("Calculating scale_pos_weight for imbalance...")
        y_train_counts = y_train.value_counts()
        neg_count = y_train_counts[0]
        pos_count = y_train_counts[1]
        scale_pos_weight = neg_count / pos_count
        print(f"Calculated scale_pos_weight (Neg/Pos Ratio): {scale_pos_weight:.2f}")

    # 4. Configure MLflow autolog (tracks params, metrics, and the model)
    mlflow.xgboost.autolog()

    # 5. Define the model with CLI hyperparameters and the calculated weight
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=-1,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
    )

    # 6. Train and evaluate inside an MLflow run
    with mlflow.start_run():
        print("Fitting XGBoost model...")
        model.fit(X_train, y_train)

        print("Evaluating on test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # prob of positive class

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        bacc = balanced_accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"AUC     : {auc:.4f}")
        print(f"Balanced Accuracy: {bacc:.4f}")

        # Log custom metrics
        mlflow.log_metric("accuracy_custom", acc)
        mlflow.log_metric("auc_custom", auc)
        mlflow.log_metric("balanced_accuracy_custom", bacc)

        # 6. Save model to the output folder for Azure ML pipeline
        os.makedirs(args.model_output, exist_ok=True)
        model_path = os.path.join(args.model_output, "model.json")
        model.save_model(model_path)
        print(f"Model saved to: {model_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
