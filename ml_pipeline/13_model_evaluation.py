"""
Step 13: Model Evaluation
Comprehensive evaluation of the risk prediction model.
Generates performance metrics and saves them for dashboard display.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score,
    average_precision_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load data and model
ite_path = os.path.join(DATA_DIR, "ite_results.csv")
base_path = os.path.join(DATA_DIR, "causal_dataset.csv")
df = pd.read_csv(ite_path if os.path.exists(ite_path) else base_path)

model = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "risk_feature_cols.pkl"))

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

X = df[FEATURE_COLS]
y = df["TenYearCHD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)[:, 1]

print(f"\n[1] Test Set Size: {len(X_test)}")
print(f"    Positive (CHD): {y_test.sum()}")
print(f"    Negative (No CHD): {(1-y_test).sum()}")

print(f"\n[2] Core Metrics:")
print(f"    Accuracy  : {accuracy_score(y_test, preds):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(y_test, proba):.4f}")
print(f"    PR-AUC    : {average_precision_score(y_test, proba):.4f}")
print(f"    Precision : {precision_score(y_test, preds):.4f}")
print(f"    Recall    : {recall_score(y_test, preds):.4f}")
print(f"    F1-Score  : {f1_score(y_test, preds):.4f}")

print(f"\n[3] Confusion Matrix:")
cm = confusion_matrix(y_test, preds)
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

print(f"\n[4] Full Classification Report:")
print(classification_report(y_test, preds, target_names=["No CHD", "CHD"]))

# Save evaluation report
eval_report = {
    "accuracy": float(accuracy_score(y_test, preds)),
    "roc_auc": float(roc_auc_score(y_test, proba)),
    "pr_auc": float(average_precision_score(y_test, proba)),
    "precision": float(precision_score(y_test, preds)),
    "recall": float(recall_score(y_test, preds)),
    "f1": float(f1_score(y_test, preds)),
    "confusion_matrix": cm.tolist(),
    "n_test": len(X_test),
    "n_positive": int(y_test.sum()),
    "n_negative": int((1 - y_test).sum())
}
joblib.dump(eval_report, os.path.join(MODEL_DIR, "eval_report.pkl"))
print("\nEvaluation report saved to models/eval_report.pkl")