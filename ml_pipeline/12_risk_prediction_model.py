"""
Step 12: Risk Prediction Model
Trains a Random Forest classifier to predict 10-year CHD risk.
Uses causally validated features (from causal dataset).
This model powers the Flask app's patient risk prediction.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Use ITE-enriched dataset if available, else base causal dataset
ite_path = os.path.join(DATA_DIR, "ite_results.csv")
base_path = os.path.join(DATA_DIR, "causal_dataset.csv")
df = pd.read_csv(ite_path if os.path.exists(ite_path) else base_path)

print("=" * 60)
print("RISK PREDICTION MODEL TRAINING")
print("=" * 60)

OUTCOME_COL = "TenYearCHD"

# Feature columns (exclude outcome and ITE columns from features)
EXCLUDE_COLS = ["TenYearCHD", "ite_s_learner", "ite_t_learner", "ite_x_learner", "ite_ensemble"]
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE_COLS]

X = df[FEATURE_COLS]
y = df[OUTCOME_COL]

print(f"\n[1] Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"[2] Dataset: {X.shape[0]} patients")
print(f"    CHD positive: {y.sum()} ({y.mean():.1%})")
print(f"    CHD negative: {(1-y).sum()} ({(1-y.mean()):.1%})")

# Train/test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[3] Train size: {len(X_train)} | Test size: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model 1: Random Forest ---
print("\n[4] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

rf_preds = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"  Random Forest - Accuracy: {rf_acc:.4f} | ROC-AUC: {rf_auc:.4f}")

# --- Model 2: Gradient Boosting ---
print("\n[5] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

gb_preds = gb_model.predict(X_test_scaled)
gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
gb_auc = roc_auc_score(y_test, gb_proba)
gb_acc = accuracy_score(y_test, gb_preds)
print(f"  Gradient Boosting - Accuracy: {gb_acc:.4f} | ROC-AUC: {gb_auc:.4f}")

# Select best model
if gb_auc >= rf_auc:
    best_model = gb_model
    best_name = "Gradient Boosting"
    best_auc = gb_auc
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_auc = rf_auc

print(f"\n[6] Best model selected: {best_name} (AUC: {best_auc:.4f})")

# Detailed evaluation
best_proba = best_model.predict_proba(X_test_scaled)[:, 1]
best_preds = best_model.predict(X_test_scaled)
print("\n[7] Classification Report:")
print(classification_report(y_test, best_preds, target_names=["No CHD", "CHD"]))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, scaler.transform(X), y, cv=cv, scoring="roc_auc")
print(f"[8] 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Save best model, scaler, and feature list
joblib.dump(best_model, os.path.join(MODEL_DIR, "risk_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "risk_scaler.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "risk_feature_cols.pkl"))

# Save evaluation metrics
metrics = {
    "model_name": best_name,
    "accuracy": float(accuracy_score(y_test, best_preds)),
    "roc_auc": float(best_auc),
    "cv_auc_mean": float(cv_scores.mean()),
    "cv_auc_std": float(cv_scores.std()),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "feature_cols": FEATURE_COLS
}
joblib.dump(metrics, os.path.join(MODEL_DIR, "model_metrics.pkl"))

print(f"\n[9] Models saved:")
print(f"    models/risk_model.pkl       <- Main prediction model")
print(f"    models/risk_scaler.pkl      <- Feature scaler")
print(f"    models/risk_feature_cols.pkl <- Feature column list")
print(f"    models/model_metrics.pkl    <- Evaluation results")