"""
Step 2: Data Cleaning
Handles missing values, drops duplicates, and saves cleaned dataset.
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "framingham.csv"))

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\n[1] Missing values BEFORE cleaning:")
print(df.isnull().sum())

# Fill missing values with column medians (more robust than mean for skewed data)
cols_to_fill = ["education", "cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
for col in cols_to_fill:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Filled '{col}' missing values with median: {median_val:.2f}")

# Drop rows where critical columns still have nulls
df.dropna(inplace=True)

# Drop duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n[2] Dropped {before - len(df)} duplicate rows.")

print("\n[3] Missing values AFTER cleaning:")
print(df.isnull().sum())

print(f"\n[4] Final dataset shape: {df.shape}")

# Ensure correct dtypes
df["currentSmoker"] = df["currentSmoker"].astype(int)
df["TenYearCHD"] = df["TenYearCHD"].astype(int)
df["male"] = df["male"].astype(int)

# Save cleaned dataset
output_path = os.path.join(DATA_DIR, "framingham_cleaned.csv")
df.to_csv(output_path, index=False)
print(f"\n[5] Cleaned dataset saved to: {output_path}")