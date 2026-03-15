"""
Step 11: Individual Treatment Effect (ITE) — FAST VERSION
==========================================================
Speed fixes applied:
  - S-Learner : GradientBoosting n_estimators=50, max_depth=3
  - T-Learner : Same lightweight models
  - X-Learner : Ridge regression for CATE models (near-instant)
  - Sample cap : Uses max 2000 rows for fitting (plenty for ITE estimation)
  Total time  : ~30-60 seconds instead of 3-5 minutes
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

print("=" * 60)
print("INDIVIDUAL TREATMENT EFFECT (ITE) — Meta-Learners")
print("=" * 60)

TREATMENT_COL = "currentSmoker"
OUTCOME_COL   = "TenYearCHD"
FEATURE_COLS  = [c for c in df.columns if c not in [TREATMENT_COL, OUTCOME_COL]]

Y  = df[OUTCOME_COL].values
T  = df[TREATMENT_COL].values
X  = df[FEATURE_COLS].values

# ── Cap sample size for speed ──────────────────────────────────────────────
MAX_ROWS = 2000
if len(df) > MAX_ROWS:
    np.random.seed(42)
    idx = np.random.choice(len(df), MAX_ROWS, replace=False)
    Y_fit, T_fit, X_fit = Y[idx], T[idx], X[idx]
    print(f"  Using {MAX_ROWS} rows for fitting (full dataset = {len(df)} rows)")
else:
    Y_fit, T_fit, X_fit = Y, T, X
    print(f"  Using all {len(df)} rows")

print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Treated : {T_fit.sum()} | Control: {(1-T_fit).sum()}")

# ── Lightweight base model (key speed fix) ─────────────────────────────────
def fast_gb():
    return GradientBoostingRegressor(
        n_estimators=50,    # was 100 — halved
        max_depth=3,        # was 4 — shallower = faster
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )

t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# S-LEARNER: Single model, treatment is just another feature
# ITE(x) = model(x, T=1) - model(x, T=0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[S-Learner] Training single model (treatment as feature)...")
t1 = time.time()
s_learner = SLearner(overall_model=fast_gb())
s_learner.fit(Y_fit, T_fit, X=X_fit)
ite_s = s_learner.effect(X)          # predict on full dataset
print(f"  Done in {time.time()-t1:.1f}s | ITE mean={ite_s.mean():.4f} std={ite_s.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# T-LEARNER: Separate model per treatment group
# ITE(x) = mu_1(x) - mu_0(x)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[T-Learner] Training separate treated/control models...")
t2 = time.time()
t_learner = TLearner(models=[fast_gb(), fast_gb()])
t_learner.fit(Y_fit, T_fit, X=X_fit)
ite_t = t_learner.effect(X)
print(f"  Done in {time.time()-t2:.1f}s | ITE mean={ite_t.mean():.4f} std={ite_t.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# X-LEARNER: Ridge CATE models (near-instant, handles group imbalance)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[X-Learner] Training with Ridge CATE models (fastest)...")
t3 = time.time()
x_learner = XLearner(
    models=[fast_gb(), fast_gb()],
    cate_models=[RidgeCV(), RidgeCV()]   # Ridge = instant
)
x_learner.fit(Y_fit, T_fit, X=X_fit)
ite_x = x_learner.effect(X)
print(f"  Done in {time.time()-t3:.1f}s | ITE mean={ite_x.mean():.4f} std={ite_x.std():.4f}")

# ── Ensemble ITE ─────────────────────────────────────────────────────────────
df["ite_s_learner"] = ite_s
df["ite_t_learner"] = ite_t
df["ite_x_learner"] = ite_x
df["ite_ensemble"]  = (ite_s + ite_t + ite_x) / 3.0

print("\n" + "=" * 60)
print("ITE SUMMARY")
print("=" * 60)
print(f"  Ensemble ITE mean : {df['ite_ensemble'].mean():.4f}")
print(f"  Ensemble ITE std  : {df['ite_ensemble'].std():.4f}")
print(f"  High-risk (>0.05) : {(df['ite_ensemble'] > 0.05).sum()} patients")
print(f"  Low-risk  (<=0)   : {(df['ite_ensemble'] <= 0).sum()} patients")
print(f"  Total time        : {time.time()-t0:.1f}s")

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(s_learner, os.path.join(MODEL_DIR, "s_learner.pkl"))
joblib.dump(t_learner, os.path.join(MODEL_DIR, "t_learner.pkl"))
joblib.dump(x_learner, os.path.join(MODEL_DIR, "x_learner.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_cols.pkl"))

ite_path = os.path.join(DATA_DIR, "ite_results.csv")
df.to_csv(ite_path, index=False)
print(f"\nSaved -> models/s_learner.pkl, t_learner.pkl, x_learner.pkl")
print(f"Saved -> {ite_path}")

print("""
What the ITE tells us:
  ITE > 0 : Smoking INCREASES this patient's CHD risk
  ITE = 0 : Smoking has no effect on this patient
  ITE < 0 : Smoking appears protective (rare, likely confounding)
  Ensemble = average of all 3 learners for stability
""")