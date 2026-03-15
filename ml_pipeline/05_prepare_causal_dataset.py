"""
Step 5: Prepare Causal Dataset
Selects the final set of variables for causal analysis and ML modeling.
Saves the dataset ready for DoWhy and EconML.
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "framingham_engineered.csv"))

print("=" * 60)
print("PREPARING CAUSAL DATASET")
print("=" * 60)

# Selected variables for causal model
TREATMENT = "currentSmoker"
OUTCOME = "TenYearCHD"

CONFOUNDERS = [
    "age",
    "male",
    "BMI",
    "totChol",
    "sysBP",
    "diaBP",
    "glucose",
    "diabetes",
    "heartRate",
    "prevalentHyp",
    "BPMeds",
    "pulsePressure",
    "isObese",
    "highChol"
]

# Build causal dataset with only needed columns
selected_cols = [TREATMENT, OUTCOME] + CONFOUNDERS
causal_df = df[selected_cols].copy()

# Drop any remaining nulls
causal_df.dropna(inplace=True)

print(f"\n[1] Treatment variable : {TREATMENT}")
print(f"[2] Outcome variable   : {OUTCOME}")
print(f"[3] Confounders ({len(CONFOUNDERS)}): {CONFOUNDERS}")
print(f"\n[4] Causal dataset shape: {causal_df.shape}")
print(f"    Treated (smokers)   : {causal_df[TREATMENT].sum()}")
print(f"    Control (non-smokers): {(1-causal_df[TREATMENT]).sum()}")
print(f"    CHD cases           : {causal_df[OUTCOME].sum()}")

output_path = os.path.join(DATA_DIR, "causal_dataset.csv")
causal_df.to_csv(output_path, index=False)
print(f"\n[5] Causal dataset saved to: {output_path}")
print("\nThis dataset is the input for:")
print("  - 06: Propensity score estimation")
print("  - 07: DoWhy causal model building")
print("  - 09: ITE / Meta-learner estimation")
print("  - 12: Risk prediction model")