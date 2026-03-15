"""
Step 8: Identify Causal Effect
Uses DoWhy to formally identify the causal estimand.
Identifies what must be estimated to answer the causal question:
  "What is the effect of smoking on 10-year CHD risk?"
"""
import pandas as pd
import os
from dowhy import CausalModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

print("=" * 60)
print("CAUSAL EFFECT IDENTIFICATION")
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

print("\n[1] Causal question:")
print(f"    What is the Average Treatment Effect (ATE) of '{TREATMENT}' on '{OUTCOME}'?")
print(f"    ATE = E[Y(1) - Y(0)] = E[CHD if smoke] - E[CHD if not smoke]")

print("\n[2] Identifying the estimand...")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

print("\n[3] Identified Estimand:")
print(identified_estimand)

print("\n[4] Explanation:")
print("    Backdoor Criterion: We can adjust for common causes (confounders)")
print("    that affect both treatment (smoking) and outcome (CHD).")
print("    By conditioning on these confounders, we block all backdoor paths")
print("    and isolate the true causal effect of smoking.")

print("\n[5] Confounders being adjusted for:")
confounders = ["age", "male", "BMI", "totChol", "sysBP", "diaBP", "glucose",
               "diabetes", "heartRate", "prevalentHyp", "BPMeds"]
for c in confounders:
    print(f"    - {c}")