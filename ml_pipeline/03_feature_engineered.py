"""
Step 3: Feature Engineering
Creates new clinically meaningful features from the cleaned dataset.
"""
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "framingham_cleaned.csv"))

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# 1. Pulse Pressure (important cardiovascular indicator)
df["pulsePressure"] = df["sysBP"] - df["diaBP"]
print("[1] Added 'pulsePressure' = sysBP - diaBP")

# 2. Obesity indicator
df["isObese"] = (df["BMI"] >= 30).astype(int)
print("[2] Added 'isObese' = BMI >= 30")

# 3. High cholesterol indicator
df["highChol"] = (df["totChol"] > 240).astype(int)
print("[3] Added 'highChol' = totChol > 240")

# 4. Hypertension stage indicator
df["hyperStage"] = 0
df.loc[df["sysBP"] >= 130, "hyperStage"] = 1   # Stage 1
df.loc[df["sysBP"] >= 140, "hyperStage"] = 2   # Stage 2
print("[4] Added 'hyperStage' = 0/1/2 based on sysBP levels")

# 5. Age group
df["ageGroup"] = pd.cut(df["age"], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3]).astype(int)
print("[5] Added 'ageGroup' = 0(<40), 1(40-55), 2(55-70), 3(70+)")

print(f"\nNew columns: {['pulsePressure','isObese','highChol','hyperStage','ageGroup']}")
print(f"Total features now: {df.shape[1]}")
print(df.head())

output_path = os.path.join(DATA_DIR, "framingham_engineered.csv")
df.to_csv(output_path, index=False)
print(f"\nEngineered dataset saved to: {output_path}")