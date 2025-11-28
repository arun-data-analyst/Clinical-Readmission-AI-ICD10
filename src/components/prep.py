import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

# 1. SETUP ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to input data")
parser.add_argument("--train_data", type=str, help="Path to save train data")
parser.add_argument("--test_data", type=str, help="Path to save test data")
args = parser.parse_args()

print("--- STARTING DATA PREP ---")

# 2. LOAD DATA
print(f"Searching for CSVs in: {args.data}")
# KRISHNA'S FIX: Use glob to find any csv, handling potential subfolders or mounting issues
csv_files = glob.glob(os.path.join(args.data, "**", "*.csv"), recursive=True)

if not csv_files:
    raise RuntimeError(f"No CSV files found in {args.data}. Content: {os.listdir(args.data)}")

print(f"Found files: {csv_files}")
df = pd.read_csv(csv_files[0])
print(f"Data Loaded. Shape: {df.shape}")

# 3. PREPROCESSING
cols_to_drop = ['encounter_id', 'patient_nbr'] 
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 4. SPLIT DATA
print("Splitting Data...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 5. SAVE DATA (THE FIX)
# KRISHNA'S FIX: Explicitly create the output directories.
print(f"Saving Train data to: {args.train_data}")
os.makedirs(args.train_data, exist_ok=True) # <--- CRITICAL LINE
train_path = os.path.join(args.train_data, "train.csv")
train_df.to_csv(train_path, index=False)

print(f"Saving Test data to: {args.test_data}")
os.makedirs(args.test_data, exist_ok=True) # <--- CRITICAL LINE
test_path = os.path.join(args.test_data, "test.csv")
test_df.to_csv(test_path, index=False)

# Verify they exist
if os.path.exists(train_path) and os.path.exists(test_path):
    print("VICTORY: Files successfully written to disk.")
else:
    raise RuntimeError("Failed to save output files!")

print("Data Prep Complete!")