import argparse  # helps pass arguments through the CLI
import os        # work with folders and file paths
import glob      # search for files that match a pattern

import pandas as pd  # tabular data handling
from sklearn.model_selection import train_test_split  # split into train / test sets


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare train/test CSVs from the enriched clinical dataset."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input folder that contains the enriched CSV (Azure input mount).",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Output folder where train.csv will be saved.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Output folder where test.csv will be saved.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set (default: 0.2).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("--- STARTING DATA PREP ---")
    print(f"Looking for input CSVs under: {args.data}")

    # 1. Locate the input CSV (handles Azure mounts and subfolders)
    csv_files = glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True)
    if not csv_files:
        folder_contents = os.listdir(args.data)
        raise RuntimeError(
            f"No CSV files found in {args.data}. "
            f"Folder contents: {folder_contents}"
        )

    input_path = csv_files[0]  # use the first CSV found
    print(f"Using input file: {input_path}")

    # 2. Load data (treat '?' as missing; low_memory=False avoids dtype warnings)
    df = pd.read_csv(input_path, na_values="?", low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")

    # 3. Drop identifier + original label columns if present
    drop_cols = ["encounter_id", "patient_nbr", "readmitted"]
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 4. Stratify ONLY on the binary target if it exists
    stratify_col = None
    if "readmitted_30d_binary" in df.columns:
        stratify_col = df["readmitted_30d_binary"]
        print("Stratifying train/test split by 'readmitted_30d_binary'.")
        print("Binary class distribution:")
        print(stratify_col.value_counts())
    else:
        print(
            "Column 'readmitted_30d_binary' not found. "
            "Splitting without stratification."
        )

    # 5. Train/test split
    print(
        f"Splitting data with test_size={args.test_size}, "
        f"random_state={args.random_state} ..."
    )
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify_col,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape : {test_df.shape}")

    # 6. Save outputs in the expected structure for the training component
    print(f"Saving train data to folder: {args.train_data}")
    os.makedirs(args.train_data, exist_ok=True)
    train_path = os.path.join(args.train_data, "train.csv")
    train_df.to_csv(train_path, index=False)

    print(f"Saving test data to folder: {args.test_data}")
    os.makedirs(args.test_data, exist_ok=True)
    test_path = os.path.join(args.test_data, "test.csv")
    test_df.to_csv(test_path, index=False)

    # 7. Quick safety check
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("VICTORY: train.csv and test.csv successfully written.")
    else:
        raise RuntimeError("Failed to save train/test CSVs to disk.")

    print("Data prep complete.")


if __name__ == "__main__":
    main()
