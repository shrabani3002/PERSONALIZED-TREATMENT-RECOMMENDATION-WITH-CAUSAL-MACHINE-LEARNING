"""
Step 7: Build Causal Model with DoWhy
Creates the formal causal model using the causal DAG.
DoWhy separates causal modeling into 4 stages:
  1. Model  - Define the causal graph
  2. Identify - Find valid identification strategy
  3. Estimate - Compute causal effect
  4. Refute - Test robustness of estimates
"""
import pandas as pd
import joblib
import os
from dowhy import CausalModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

print("=" * 60)
print("BUILDING CAUSAL MODEL (DoWhy)")
print("=" * 60)

TREATMENT = "currentSmoker"
OUTCOME = "TenYearCHD"
CONFOUNDERS = [
    "age", "male", "BMI", "totChol",
    "sysBP", "diaBP", "glucose", "diabetes",
    "heartRate", "prevalentHyp", "BPMeds"
]

# Causal graph as GML
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

# STAGE 1: MODEL
print("\n[Stage 1] Building Causal Model...")
model = CausalModel(
    data=df,
    treatment=TREATMENT,
    outcome=OUTCOME,
    graph=causal_graph
)
print(f"  ✓ Causal model created")
print(f"    Treatment : {TREATMENT}")
print(f"    Outcome   : {OUTCOME}")
print(f"    Data shape: {df.shape}")

# STAGE 2: IDENTIFY
print("\n[Stage 2] Identifying Causal Effect...")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(f"  ✓ Estimand identified")
print(f"  Identification strategy: {identified_estimand}")

# Save the model and estimand info
joblib.dump({
    "treatment": TREATMENT,
    "outcome": OUTCOME,
    "confounders": CONFOUNDERS,
    "graph": causal_graph
}, os.path.join(MODEL_DIR, "causal_model_config.pkl"))

print("\n[Stage 3] Model configuration saved to models/causal_model_config.pkl")
print("\nNext steps:")
print("  Run 08_identify_causal_effect.py -> to see full estimand")
print("  Run 09_estimate_causal_effect.py -> to compute ATE")
print("  Run 11_individual_treatment_effect.py -> to compute ITE per patient")