"""
Step 9: Estimate Causal Effect (ATE)
Estimates the Average Treatment Effect using multiple methods:
  - Linear Regression (backdoor adjustment)
  - Propensity Score Stratification
  - Inverse Propensity Weighting (IPW)
Saves ATE results for dashboard display.
"""
import pandas as pd
import numpy as np
import joblib
import os
from dowhy import CausalModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

print("=" * 60)
print("CAUSAL EFFECT ESTIMATION (ATE)")
print("=" * 60)

TREATMENT = "currentSmoker"
OUTCOME = "TenYearCHD"

causal_graph = """
digraph {
    age -> currentSmoker;
    age -> sysBP;
    age -> totChol;
    age -> TenYearCHD;
    male -> currentSmoker;
    male -> TenYearCHD;
    currentSmoker -> totChol;
    currentSmoker -> sysBP;
    currentSmoker -> TenYearCHD;
    BMI -> sysBP;
    BMI -> diabetes;
    BMI -> TenYearCHD;
    sysBP -> TenYearCHD;
    diaBP -> TenYearCHD;
    totChol -> TenYearCHD;
    glucose -> diabetes;
    glucose -> TenYearCHD;
    diabetes -> TenYearCHD;
    heartRate -> TenYearCHD;
    prevalentHyp -> sysBP;
    prevalentHyp -> TenYearCHD;
    BPMeds -> sysBP;
}
"""

model = CausalModel(
    data=df,
    treatment=TREATMENT,
    outcome=OUTCOME,
    graph=causal_graph
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

results = {}

# --- Method 1: Linear Regression ---
print("\n[Method 1] Linear Regression (backdoor adjustment)...")
try:
    estimate_lr = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    ate_lr = float(estimate_lr.value)
    results["linear_regression"] = ate_lr
    print(f"  ATE (Linear Regression): {ate_lr:.6f}")
    print(f"  Interpretation: Smoking changes 10-year CHD risk by {ate_lr:.4f} ({ate_lr*100:.2f}%)")
except Exception as e:
    print(f"  Error: {e}")
    results["linear_regression"] = None

# --- Method 2: Propensity Score Stratification ---
print("\n[Method 2] Propensity Score Stratification...")
try:
    estimate_ps = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_stratification",
        target_units="ate"
    )
    ate_ps = float(estimate_ps.value)
    results["propensity_stratification"] = ate_ps
    print(f"  ATE (Propensity Stratification): {ate_ps:.6f}")
except Exception as e:
    print(f"  Error: {e}")
    results["propensity_stratification"] = None

# --- Method 3: Propensity Score Weighting (IPW) ---
print("\n[Method 3] Inverse Propensity Weighting (IPW)...")
try:
    estimate_ipw = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate"
    )
    ate_ipw = float(estimate_ipw.value)
    results["ipw"] = ate_ipw
    print(f"  ATE (IPW): {ate_ipw:.6f}")
except Exception as e:
    print(f"  Error: {e}")
    results["ipw"] = None

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY OF ATE ESTIMATES")
print("=" * 60)
for method, val in results.items():
    if val is not None:
        print(f"  {method:<35}: {val:.6f}  ({val*100:+.3f}% CHD risk change)")

valid_ates = [v for v in results.values() if v is not None]
if valid_ates:
    consensus_ate = np.mean(valid_ates)
    print(f"\n  Consensus ATE (average): {consensus_ate:.6f} ({consensus_ate*100:+.3f}%)")
    results["consensus_ate"] = float(consensus_ate)

# Save ATE results for Flask app
joblib.dump(results, os.path.join(MODEL_DIR, "ate_results.pkl"))
print(f"\nATE results saved to models/ate_results.pkl")