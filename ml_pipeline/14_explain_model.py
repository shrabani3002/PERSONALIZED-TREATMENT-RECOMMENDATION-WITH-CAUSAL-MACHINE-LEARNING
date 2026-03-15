"""
Step 14: SHAP Explainability — FIXED VERSION
=============================================
Fix: SHAP output shape varies by model type and sklearn/shap version:
  RandomForest  -> list [class0_array, class1_array]  each (n, f)
  GradientBoost -> 3D array (n, f, 2)  OR  list
  Generic       -> 2D array (n, f)

The old code assumed list format always — crashed on 3D array.
This version handles all 3 cases safely.
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
import shap

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

model        = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "risk_feature_cols.pkl"))

ite_path  = os.path.join(DATA_DIR, "ite_results.csv")
base_path = os.path.join(DATA_DIR, "causal_dataset.csv")
df        = pd.read_csv(ite_path if os.path.exists(ite_path) else base_path)

print("=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

X        = df[FEATURE_COLS]
X_sc     = scaler.transform(X)
X_sc_df  = pd.DataFrame(X_sc, columns=FEATURE_COLS)

# Cap rows for speed
SAMPLE   = min(200, len(X_sc_df))
X_sample = X_sc_df.iloc[:SAMPLE]
print(f"  SHAP sample : {SAMPLE} rows  (full dataset = {len(df)})")
print(f"  Features    : {len(FEATURE_COLS)}")

# ── Build explainer ───────────────────────────────────────────────────────────
t0 = time.time()
print("\n[1] Building TreeExplainer ...")
explainer = shap.TreeExplainer(model)

print(f"[2] Computing SHAP values for {SAMPLE} rows ...")
shap_vals = explainer.shap_values(X_sample)

# ── Safely extract 2D (n_samples, n_features) array ──────────────────────────
# SHAP output varies across model types and library versions:
#   list of 2 arrays  -> RandomForest, old shap versions
#   3D ndarray        -> GradientBoosting, newer shap versions
#   2D ndarray        -> regression models
print(f"\n[3] Parsing SHAP output ...")
print(f"  Raw type  : {type(shap_vals)}")

if isinstance(shap_vals, list):
    # list [class_0_array, class_1_array] — pick class 1 (CHD positive)
    sv = np.array(shap_vals[1])
    print(f"  Format    : list of {len(shap_vals)} arrays -> using class 1")

elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    # shape (n_samples, n_features, n_classes) — pick last class
    sv = shap_vals[:, :, -1]
    print(f"  Format    : 3D array {shap_vals.shape} -> using [:,:,-1]")

else:
    # Already 2D (n_samples, n_features)
    sv = np.array(shap_vals)
    print(f"  Format    : 2D array {sv.shape}")

# Force exactly 2D
sv = np.array(sv)
if sv.ndim == 1:
    sv = sv.reshape(1, -1)
if sv.ndim != 2:
    sv = sv.reshape(SAMPLE, len(FEATURE_COLS))

print(f"  Final shape: {sv.shape}  (expected: {SAMPLE} x {len(FEATURE_COLS)})")
print(f"  Done in {time.time()-t0:.1f}s")

# Sanity check
assert sv.shape == (SAMPLE, len(FEATURE_COLS)), \
    f"Shape mismatch: got {sv.shape}, expected ({SAMPLE}, {len(FEATURE_COLS)})"

# ── Global feature importance ─────────────────────────────────────────────────
mean_abs   = np.abs(sv).mean(axis=0)          # 1D array, length = n_features
importance = pd.DataFrame({
    "feature"       : list(FEATURE_COLS),
    "mean_abs_shap" : mean_abs.tolist(),       # .tolist() = plain Python list = 1D
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\n[4] Global Feature Importance (mean |SHAP value|):")
print(importance.to_string(index=False))

# ── Individual patient example ────────────────────────────────────────────────
print("\n[5] Patient 0 — top contributing features:")
p_shap = pd.Series(sv[0].tolist(), index=FEATURE_COLS)
p_top  = p_shap.abs().sort_values(ascending=False).head(5)
for feat, val in p_top.items():
    direction = "increases" if p_shap[feat] > 0 else "decreases"
    print(f"  {feat:<20} |SHAP|={abs(val):.4f}  -> {direction} CHD risk")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
importance.to_csv(os.path.join(DATA_DIR, "shap_importance.csv"), index=False)

print(f"\nSaved -> models/shap_explainer.pkl")
print(f"Saved -> data/shap_importance.csv")
print(f"\nInterpretation guide:")
print(f"  Higher |SHAP| = stronger influence on CHD risk prediction")
print(f"  Positive SHAP = feature pushes prediction toward CHD")
print(f"  Negative SHAP = feature pushes prediction away from CHD")
print(f"  NOTE: SHAP = correlation-based. Compare with ITE for true causal importance.")
