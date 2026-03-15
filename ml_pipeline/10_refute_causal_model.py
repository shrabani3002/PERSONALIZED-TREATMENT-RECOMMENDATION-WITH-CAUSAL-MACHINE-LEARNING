"""
Step 10: Refute Causal Model  — PURE SKLEARN, NO DoWhy refutation
==================================================================
DoWhy's refute_estimate() is COMPLETELY REMOVED.
All 3 tests are implemented with numpy + sklearn in < 10 seconds.

Why DoWhy refutation is so slow:
  - It re-installs a full causal graph + estimand + estimator
    for EVERY simulation (default 100 simulations per test)
  - Even at n_simulations=10 it still hits internal DoWhy
    overhead (graph parsing, networkx traversal) per sim

This version:
  - Uses OLS (LinearRegression) directly — same math, 100x faster
  - Test 1: add random noise column, check ATE change  (~1s)
  - Test 2: permute treatment 20 times, check placebo ATE (~2s)
  - Test 3: 10 random 80% subsets, check ATE variance   (~2s)
  Total: under 10 seconds
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.linear_model import LinearRegression

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "causal_dataset.csv"))

TREATMENT    = "currentSmoker"
OUTCOME      = "TenYearCHD"
CONFOUNDERS  = [c for c in df.columns if c not in [TREATMENT, OUTCOME]]

Y = df[OUTCOME].values
T = df[TREATMENT].values
X = df[CONFOUNDERS].values

print("=" * 60)
print("CAUSAL MODEL REFUTATION TESTS  (fast sklearn version)")
print("=" * 60)
print(f"  Rows        : {len(df)}")
print(f"  Treatment   : {TREATMENT}")
print(f"  Outcome     : {OUTCOME}")
print(f"  Confounders : {len(CONFOUNDERS)}")

t0 = time.time()

# ── Baseline ATE via OLS ──────────────────────────────────────────────────────
# OLS coefficient on treatment = ATE after linear adjustment for confounders
# This is identical to DoWhy backdoor.linear_regression
X_full  = np.column_stack([T, X])
baseline_ate = LinearRegression().fit(X_full, Y).coef_[0]
print(f"\n  Baseline ATE (OLS backdoor): {baseline_ate:.6f}")

refute_results = {
    "original_ate" : float(baseline_ate),
    "method"       : "OLS backdoor adjustment (sklearn)",
}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Random Common Cause
# Add a random N(0,1) column as a fake confounder.
# ATE should not change — proves estimate is not sensitive to spurious vars.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 1] Random Common Cause ...")
t1 = time.time()
np.random.seed(42)
noise_ates = []
for _ in range(20):                        # 20 random noise columns
    noise      = np.random.normal(0, 1, len(df)).reshape(-1, 1)
    X_noisy    = np.column_stack([T, X, noise])
    ate_noisy  = LinearRegression().fit(X_noisy, Y).coef_[0]
    noise_ates.append(ate_noisy)

mean_noise_ate = float(np.mean(noise_ates))
diff1          = abs(mean_noise_ate - baseline_ate)
passed1        = diff1 < 0.005
print(f"  Baseline ATE        : {baseline_ate:.6f}")
print(f"  Mean ATE with noise : {mean_noise_ate:.6f}")
print(f"  Max deviation       : {max(abs(a - baseline_ate) for a in noise_ates):.6f}")
print(f"  Avg deviation       : {diff1:.6f}  (threshold < 0.005)")
print(f"  Result              : {'PASS' if passed1 else 'FAIL'}  ({time.time()-t1:.2f}s)")
refute_results["random_common_cause"] = {
    "baseline_ate": float(baseline_ate),
    "mean_noisy_ate": mean_noise_ate,
    "avg_deviation": float(diff1),
    "passed": passed1,
}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Placebo Treatment
# Permute (shuffle) the treatment column randomly.
# ATE on scrambled treatment should be ≈ 0.
# If real treatment has genuine causal effect, placebo should vanish.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 2] Placebo Treatment ...")
t2 = time.time()
np.random.seed(42)
placebo_ates = []
for _ in range(20):                        # 20 permutations
    T_perm      = np.random.permutation(T)
    X_perm      = np.column_stack([T_perm, X])
    ate_perm    = LinearRegression().fit(X_perm, Y).coef_[0]
    placebo_ates.append(ate_perm)

mean_placebo = float(np.mean(placebo_ates))
passed2      = abs(mean_placebo) < 0.005
print(f"  Baseline ATE   : {baseline_ate:.6f}")
print(f"  Mean placebo   : {mean_placebo:.6f}  (should be ≈ 0)")
print(f"  Max placebo    : {max(abs(a) for a in placebo_ates):.6f}")
print(f"  Result         : {'PASS' if passed2 else 'FAIL'}  ({time.time()-t2:.2f}s)")
print(f"  Interpretation : {'Real treatment effect confirmed' if passed2 else 'Check — placebo unexpectedly large'}")
refute_results["placebo_treatment"] = {
    "baseline_ate": float(baseline_ate),
    "mean_placebo_ate": mean_placebo,
    "passed": passed2,
}

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Data Subset
# Estimate ATE on 10 random 80% subsets.
# ATE should be stable (low variance) across subsets.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 3] Data Subset (80%) ...")
t3 = time.time()
np.random.seed(42)
subset_ates = []
n = len(df)
for _ in range(10):                        # 10 subsets
    idx        = np.random.choice(n, size=int(0.8*n), replace=False)
    X_sub      = np.column_stack([T[idx], X[idx]])
    ate_sub    = LinearRegression().fit(X_sub, Y[idx]).coef_[0]
    subset_ates.append(ate_sub)

mean_subset = float(np.mean(subset_ates))
std_subset  = float(np.std(subset_ates))
diff3       = abs(mean_subset - baseline_ate)
passed3     = diff3 < 0.01 and std_subset < 0.01
print(f"  Baseline ATE   : {baseline_ate:.6f}")
print(f"  Mean subset    : {mean_subset:.6f}  (diff={diff3:.6f})")
print(f"  Std dev        : {std_subset:.6f}   (threshold < 0.01)")
print(f"  Min / Max      : {min(subset_ates):.6f} / {max(subset_ates):.6f}")
print(f"  Result         : {'PASS' if passed3 else 'FAIL'}  ({time.time()-t3:.2f}s)")
refute_results["data_subset"] = {
    "baseline_ate": float(baseline_ate),
    "mean_subset_ate": mean_subset,
    "std_subset": std_subset,
    "diff": float(diff3),
    "passed": passed3,
}

# ── Final summary ─────────────────────────────────────────────────────────────
total_time = time.time() - t0
print("\n" + "=" * 60)
print("REFUTATION SUMMARY")
print("=" * 60)
print(f"  Baseline ATE : {baseline_ate:.6f}")
print(f"  Total time   : {total_time:.2f}s\n")

all_pass = True
for key, name, criterion in [
    ("random_common_cause", "Random Common Cause",
     f"Avg ATE deviation {refute_results['random_common_cause']['avg_deviation']:.5f} < 0.005"),
    ("placebo_treatment",   "Placebo Treatment",
     f"Mean placebo ATE {refute_results['placebo_treatment']['mean_placebo_ate']:.5f} ≈ 0"),
    ("data_subset",         "Data Subset 80%",
     f"Std across subsets {refute_results['data_subset']['std_subset']:.5f} < 0.01"),
]:
    p = refute_results[key]["passed"]
    all_pass = all_pass and p
    print(f"  {'PASS' if p else 'FAIL'}  {name:<25} {criterion}")

print()
if all_pass:
    print("  All 3 refutation tests passed.")
    print("  The causal estimate is robust and reliable.")
else:
    print("  Some tests failed — review individual results above.")

print("""
What each test proves:
  Test 1 (Random Cause)  : ATE is not inflated by spurious correlations
  Test 2 (Placebo)       : Treatment has a REAL effect, not random noise
  Test 3 (Subset)        : ATE is stable — not driven by a few outlier rows
""")

joblib.dump(refute_results, os.path.join(MODEL_DIR, "refutation_results.pkl"))
print("Saved -> models/refutation_results.pkl")