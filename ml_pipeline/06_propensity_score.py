"""
Step 6: Propensity Score Estimation
Estimates P(treatment=1 | covariates) using logistic regression.
Propensity scores are used to:
  - Check covariate overlap between treated/control groups
  - Adjust for selection bias in observational data
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

print("=" * 60)
print("PROPENSITY SCORE ESTIMATION")
print("=" * 60)

# Treatment variable
T = df["currentSmoker"]

# Covariates (confounders)
COVARIATE_COLS = [
    "age", "male", "totChol", "sysBP", "diaBP",
    "BMI", "heartRate", "glucose", "diabetes",
    "prevalentHyp", "BPMeds", "pulsePressure", "isObese", "highChol"
]
X = df[COVARIATE_COLS]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train logistic regression for propensity scores
ps_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
ps_model.fit(X_scaled, T)

# Cross-validation AUC
cv_scores = cross_val_score(ps_model, X_scaled, T, cv=5, scoring="roc_auc")
print(f"\n[1] Propensity model CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Compute propensity scores
df["propensity_score"] = ps_model.predict_proba(X_scaled)[:, 1]

# Check overlap
treated_ps = df.loc[df["currentSmoker"] == 1, "propensity_score"]
control_ps = df.loc[df["currentSmoker"] == 0, "propensity_score"]

print(f"\n[2] Propensity score overlap check:")
print(f"    Treated  group - Mean PS: {treated_ps.mean():.4f}, Range: [{treated_ps.min():.4f}, {treated_ps.max():.4f}]")
print(f"    Control  group - Mean PS: {control_ps.mean():.4f}, Range: [{control_ps.min():.4f}, {control_ps.max():.4f}]")

# Overlap is good if the ranges overlap substantially
overlap_min = max(treated_ps.min(), control_ps.min())
overlap_max = min(treated_ps.max(), control_ps.max())
print(f"    Overlap region : [{overlap_min:.4f}, {overlap_max:.4f}]")
if overlap_min < overlap_max:
    print("    ✓ Good overlap between treated and control groups")
else:
    print("    ✗ Warning: Poor overlap - causal estimates may be unreliable")

# Save propensity score model and scaler
joblib.dump(ps_model, os.path.join(MODEL_DIR, "propensity_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "ps_scaler.pkl"))

# Save dataset with propensity scores
output_path = os.path.join(DATA_DIR, "framingham_propensity.csv")
df.to_csv(output_path, index=False)
print(f"\n[3] Propensity scores saved to: {output_path}")
print(f"[4] Propensity model saved to models/propensity_model.pkl")

print("\n[5] Feature coefficients (top confounders by |effect| on smoking):")
coef_df = pd.DataFrame({
    "feature": COVARIATE_COLS,
    "coefficient": ps_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))