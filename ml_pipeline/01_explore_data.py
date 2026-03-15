"""
Step 1: Data Exploration
Loads and explores the Framingham Heart Study dataset.
"""
import pandas as pd
import os

# Determine correct path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "framingham.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("FRAMINGHAM HEART STUDY DATASET - EXPLORATION")
print("=" * 60)

print("\n[1] First 5 rows:")
print(df.head())

print("\n[2] Columns in dataset:")
print(list(df.columns))

print("\n[3] Dataset shape (rows, columns):", df.shape)

print("\n[4] Dataset information:")
print(df.info())

print("\n[5] Missing values per column:")
print(df.isnull().sum())

print("\n[6] Statistical summary:")
print(df.describe())

print("\n[7] Target variable distribution (TenYearCHD):")
print(df["TenYearCHD"].value_counts())
print(f"  CHD Rate: {df['TenYearCHD'].mean():.2%}")

print("\n[8] Treatment variable distribution (currentSmoker):")
print(df["currentSmoker"].value_counts())