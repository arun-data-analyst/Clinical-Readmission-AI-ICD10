import argparse
import os
import pandas as pd


# ---------------------------------------------
# ICD-9 -> ICD-10-CA CHAPTER MAPPING FUNCTION
# ---------------------------------------------
def map_icd9_to_icd10ca_chapter(icd9_code):
    """Map ICD-9 primary diagnosis code to an ICD-10-CA–style chapter name."""
    if pd.isna(icd9_code):
        return "Unknown"
    
    code_str = str(icd9_code)

    # --- HANDLE V-CODES (Factors influencing health status) ---
    if code_str.startswith("V"):
        return "Z00-Z99 (Factors influencing health status)"

    # --- HANDLE E-CODES (External causes of injury) ---
    if code_str.startswith("E"):
        return "V01-Y98 (External causes of morbidity)"

    # --- HANDLE NUMERIC CODES ---
    try:
        code_num = float(code_str)

        # 1. CIRCULATORY SYSTEM (Heart Disease) -> ICD-10 I00-I99
        if 390 <= code_num <= 459 or code_num == 785:
            return "I00-I99 (Diseases of the circulatory system)"

        # 2. DIABETES (Core of this study) -> ICD-10 E08-E13
        elif 249 <= code_num < 251:
            return "E08-E13 (Diabetes mellitus)"

        # 3. RESPIRATORY SYSTEM -> ICD-10 J00-J99
        elif 460 <= code_num <= 519 or code_num == 786:
            return "J00-J99 (Diseases of the respiratory system)"

        # 4. DIGESTIVE SYSTEM -> ICD-10 K00-K93
        elif 520 <= code_num <= 579 or code_num == 787:
            return "K00-K93 (Diseases of the digestive system)"

        # 5. INJURY AND POISONING -> ICD-10 S00-T98
        elif 800 <= code_num <= 999:
            return "S00-T98 (Injury, poisoning and consequences of external causes)"

        # 6. MUSCULOSKELETAL -> ICD-10 M00-M99
        elif 710 <= code_num <= 739:
            return "M00-M99 (Diseases of the musculoskeletal system)"

        # 7. NEOPLASMS (Cancer) -> ICD-10 C00-D48
        elif 140 <= code_num <= 239:
            return "C00-D48 (Neoplasms)"

        else:
            return "Other (Genitourinary, Mental, Skin, etc.)"

    except ValueError:
        return "Unknown"


# ---------------------------------------------
# HbA1c HIGH-RISK FLAG
# ---------------------------------------------
def check_high_risk_a1c(result):
    """Return 1 if HbA1c result is >8 (high risk), else 0."""
    if result == ">8":
        return 1
    return 0


# ---------------------------------------------
# BINARY 30-DAY READMISSION FLAG
# ---------------------------------------------
def make_readmitted_30d_binary(value):
    """
    Map original 'readmitted' values to a binary 30-day readmission flag.

    Original UCI values:
      - "NO"  : no readmission
      - "<30" : readmitted within 30 days
      - ">30" : readmitted after 30 days

    Binary flag:
      - 1 if "<30"
      - 0 if "NO" or ">30" (or anything else)
    """
    value_str = str(value).strip()
    if value_str == "<30":
        return 1
    return 0


# ---------------------------------------------
# MAIN ENRICHMENT LOGIC
# ---------------------------------------------
def main(raw_data_path: str, output_dir: str):
    print("Loading the US Diabetes Dataset...")
    df = pd.read_csv(raw_data_path, na_values="?")
    print(f"Initial Shape: {df.shape}")

    # 1) Clinical ICD mapping on primary diagnosis
    print("Applying Clinical Mapping (ICD-9 -> ICD-10-CA) on diag_1...")
    df["Primary_Diagnosis_Group"] = df["diag_1"].apply(map_icd9_to_icd10ca_chapter)

    # 2) HbA1c high-risk flag
    print("Creating High_Risk_A1C flag from A1Cresult...")
    df["High_Risk_A1C"] = df["A1Cresult"].apply(check_high_risk_a1c)

    # 3) Binary 30-day readmission label
    print("Creating binary 30-day readmission flag (readmitted_30d_binary)...")
    df["readmitted_30d_binary"] = df["readmitted"].apply(make_readmitted_30d_binary)

    # 4) Simple reports for sanity check
    print("-" * 50)
    print("MAPPING REPORT (Primary_Diagnosis_Group):")
    print(df["Primary_Diagnosis_Group"].value_counts())
    print("-" * 50)
    print("READMISSION_30D_BINARY value_counts():")
    print(df["readmitted_30d_binary"].value_counts())
    print("-" * 50)

    # Avoid leakage into AutoML by removing the original multiclass label
    if "readmitted" in df.columns:
        df = df.drop(columns=["readmitted"])
        
    # 5) Save enriched dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "diabetes_clinical_enriched.csv")
    df.to_csv(output_path, index=False)
    print(f"File saved as '{output_path}'. Ready for Azure.")


# ---------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ICD-9 enrichment + HbA1c and 30-day readmission flags "
                    "for the UCI diabetes readmission dataset."
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
        help="Path to raw diabetic_data.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where enriched CSV will be saved",
    )
    args = parser.parse_args()

    main(args.raw_data, args.output_dir)
