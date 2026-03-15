"""
Risk Analysis Route — Enhanced Version
Provides detailed per-feature risk explanation based on actual patient values.
"""
from flask import Blueprint, render_template, request
import numpy as np
import joblib
import os

risk_bp = Blueprint("risk_bp", __name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load models at startup
model        = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))
scaler       = joblib.load(os.path.join(MODEL_DIR, "risk_scaler.pkl"))
FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "risk_feature_cols.pkl"))

# Try loading meta-learners (optional — only if step 11 was run)
try:
    t_learner    = joblib.load(os.path.join(MODEL_DIR, "t_learner.pkl"))
    s_learner    = joblib.load(os.path.join(MODEL_DIR, "s_learner.pkl"))
    x_learner    = joblib.load(os.path.join(MODEL_DIR, "x_learner.pkl"))
    META_AVAILABLE = True
except Exception:
    META_AVAILABLE = False


def compute_ite(patient_dict):
    """Estimate Individual Treatment Effect using ensemble of meta-learners."""
    if not META_AVAILABLE:
        return None, {}
    causal_cols = [c for c in FEATURE_COLS if c != "currentSmoker"]
    X_c = np.array([patient_dict.get(c, 0.0) for c in causal_cols]).reshape(1, -1)
    ites = {}
    for name, learner in [("T-Learner", t_learner),
                           ("S-Learner", s_learner),
                           ("X-Learner", x_learner)]:
        try:
            ites[name] = round(float(learner.effect(X_c)[0]), 4)
        except Exception:
            ites[name] = None
    valid = [v for v in ites.values() if v is not None]
    ensemble = round(float(np.mean(valid)), 4) if valid else None
    return ensemble, ites


def build_detailed_explanation(patient, prob):
    """
    Build a detailed, factor-by-factor explanation of the risk score.
    Returns a list of dicts with: factor, value, status, impact, detail
    """
    factors = []

    # ── Smoking ──────────────────────────────────────────────────────────────
    cigs = patient["cigsPerDay"]
    if cigs > 20:
        factors.append({
            "factor": "Heavy Smoking",
            "value": f"{int(cigs)} cigarettes/day",
            "status": "critical",
            "impact": "Very High",
            "detail": "Smoking >20 cigarettes/day doubles CHD risk. Nicotine raises BP and damages arterial walls.",
            "icon": "🚬"
        })
    elif cigs > 0:
        factors.append({
            "factor": "Smoking",
            "value": f"{int(cigs)} cigarettes/day",
            "status": "danger",
            "impact": "High",
            "detail": "Any smoking causally increases CHD risk by damaging endothelial lining and raising LDL.",
            "icon": "🚬"
        })

    # ── Systolic BP ──────────────────────────────────────────────────────────
    sys = patient["sysBP"]
    if sys >= 160:
        factors.append({
            "factor": "Severe Hypertension",
            "value": f"{sys:.0f} mmHg systolic",
            "status": "critical",
            "impact": "Very High",
            "detail": "Stage 2+ hypertension (≥160 mmHg) significantly stresses the heart and arteries, directly elevating CHD risk.",
            "icon": "🩸"
        })
    elif sys >= 140:
        factors.append({
            "factor": "Stage 2 Hypertension",
            "value": f"{sys:.0f} mmHg systolic",
            "status": "danger",
            "impact": "High",
            "detail": "Systolic BP ≥140 mmHg indicates stage 2 hypertension. A 10 mmHg increase raises CHD risk by ~20%.",
            "icon": "🩸"
        })
    elif sys >= 130:
        factors.append({
            "factor": "Elevated Blood Pressure",
            "value": f"{sys:.0f} mmHg systolic",
            "status": "warning",
            "impact": "Moderate",
            "detail": "Stage 1 hypertension (130–139 mmHg). Monitor and consider lifestyle changes.",
            "icon": "🩸"
        })
    else:
        factors.append({
            "factor": "Blood Pressure",
            "value": f"{sys:.0f} mmHg systolic",
            "status": "normal",
            "impact": "Low",
            "detail": "Systolic BP is within normal range (<130 mmHg). Good cardiovascular indicator.",
            "icon": "🩸"
        })

    # ── Cholesterol ──────────────────────────────────────────────────────────
    chol = patient["totChol"]
    if chol > 280:
        factors.append({
            "factor": "Very High Cholesterol",
            "value": f"{chol:.0f} mg/dL",
            "status": "critical",
            "impact": "Very High",
            "detail": "Total cholesterol >280 mg/dL indicates severe hypercholesterolaemia. Strongly associated with arterial plaque buildup.",
            "icon": "🧪"
        })
    elif chol > 240:
        factors.append({
            "factor": "High Cholesterol",
            "value": f"{chol:.0f} mg/dL",
            "status": "danger",
            "impact": "High",
            "detail": "Total cholesterol >240 mg/dL is a major CHD risk factor. Statins and dietary changes are recommended.",
            "icon": "🧪"
        })
    elif chol > 200:
        factors.append({
            "factor": "Borderline Cholesterol",
            "value": f"{chol:.0f} mg/dL",
            "status": "warning",
            "impact": "Moderate",
            "detail": "Cholesterol 200–240 mg/dL is borderline high. Diet and exercise can help reduce levels.",
            "icon": "🧪"
        })
    else:
        factors.append({
            "factor": "Cholesterol",
            "value": f"{chol:.0f} mg/dL",
            "status": "normal",
            "impact": "Low",
            "detail": "Total cholesterol is in the desirable range (<200 mg/dL).",
            "icon": "🧪"
        })

    # ── Diabetes ─────────────────────────────────────────────────────────────
    if patient["diabetes"] == 1:
        factors.append({
            "factor": "Diabetes",
            "value": "Diagnosed",
            "status": "danger",
            "impact": "High",
            "detail": "Diabetes doubles CHD risk. High blood sugar damages blood vessels and nerves controlling the heart.",
            "icon": "💉"
        })

    # ── Glucose ──────────────────────────────────────────────────────────────
    gluc = patient["glucose"]
    if patient["diabetes"] == 0:
        if gluc > 126:
            factors.append({
                "factor": "High Fasting Glucose",
                "value": f"{gluc:.0f} mg/dL",
                "status": "danger",
                "impact": "High",
                "detail": "Fasting glucose >126 mg/dL suggests undiagnosed diabetes. This significantly raises CHD risk.",
                "icon": "🍬"
            })
        elif gluc > 100:
            factors.append({
                "factor": "Pre-diabetic Glucose",
                "value": f"{gluc:.0f} mg/dL",
                "status": "warning",
                "impact": "Moderate",
                "detail": "Fasting glucose 100–126 mg/dL indicates pre-diabetes. Lifestyle changes can prevent progression.",
                "icon": "🍬"
            })

    # ── BMI ──────────────────────────────────────────────────────────────────
    bmi = patient["BMI"]
    if bmi >= 35:
        factors.append({
            "factor": "Severe Obesity",
            "value": f"BMI {bmi:.1f}",
            "status": "critical",
            "impact": "Very High",
            "detail": "BMI ≥35 (severe obesity) significantly strains the heart, raises BP, and promotes insulin resistance.",
            "icon": "⚖️"
        })
    elif bmi >= 30:
        factors.append({
            "factor": "Obesity",
            "value": f"BMI {bmi:.1f}",
            "status": "danger",
            "impact": "High",
            "detail": "BMI ≥30 (obese) causally increases blood pressure and CHD risk through multiple pathways.",
            "icon": "⚖️"
        })
    elif bmi >= 25:
        factors.append({
            "factor": "Overweight",
            "value": f"BMI {bmi:.1f}",
            "status": "warning",
            "impact": "Moderate",
            "detail": "BMI 25–30 (overweight). Even modest weight loss of 5–10% can meaningfully reduce CHD risk.",
            "icon": "⚖️"
        })
    else:
        factors.append({
            "factor": "BMI",
            "value": f"BMI {bmi:.1f}",
            "status": "normal",
            "impact": "Low",
            "detail": "BMI is in the normal range (18.5–24.9). Healthy weight is protective against CHD.",
            "icon": "⚖️"
        })

    # ── Age ──────────────────────────────────────────────────────────────────
    age = patient["age"]
    if age >= 65:
        factors.append({
            "factor": "Age",
            "value": f"{int(age)} years",
            "status": "warning",
            "impact": "Moderate–High",
            "detail": "Age ≥65 is a major non-modifiable CHD risk factor. Risk accumulates with age due to arterial stiffening.",
            "icon": "📅"
        })
    elif age >= 50:
        factors.append({
            "factor": "Age",
            "value": f"{int(age)} years",
            "status": "warning",
            "impact": "Moderate",
            "detail": "Age 50–65: CHD risk begins rising significantly. Regular screening is recommended.",
            "icon": "📅"
        })

    # ── Sex ──────────────────────────────────────────────────────────────────
    if patient["male"] == 1 and age >= 45:
        factors.append({
            "factor": "Sex & Age",
            "value": "Male ≥45",
            "status": "warning",
            "impact": "Moderate",
            "detail": "Men ≥45 have higher CHD risk than women of the same age. Estrogen in pre-menopausal women is cardioprotective.",
            "icon": "👤"
        })

    # ── BP Meds ──────────────────────────────────────────────────────────────
    if patient["BPMeds"] == 1:
        factors.append({
            "factor": "On BP Medication",
            "value": "Yes",
            "status": "warning",
            "impact": "Indicates history",
            "detail": "Use of BP medication indicates a history of hypertension. While medication helps, underlying vascular risk remains elevated.",
            "icon": "💊"
        })

    # ── Prevalent Stroke ─────────────────────────────────────────────────────
    if patient["prevalentStroke"] == 1:
        factors.append({
            "factor": "Previous Stroke",
            "value": "Yes",
            "status": "critical",
            "impact": "Very High",
            "detail": "A prior stroke indicates severe cerebrovascular disease, which shares risk factors with coronary heart disease.",
            "icon": "🧠"
        })

    # ── Heart Rate ───────────────────────────────────────────────────────────
    hr = patient["heartRate"]
    if hr > 100:
        factors.append({
            "factor": "Elevated Heart Rate",
            "value": f"{int(hr)} bpm",
            "status": "warning",
            "impact": "Moderate",
            "detail": "Resting heart rate >100 bpm (tachycardia) increases cardiac workload and is associated with higher CHD risk.",
            "icon": "💓"
        })

    # ── Protective factors ───────────────────────────────────────────────────
    if cigs == 0:
        factors.append({
            "factor": "Non-Smoker",
            "value": "0 cigarettes/day",
            "status": "good",
            "impact": "Protective",
            "detail": "Not smoking is one of the most impactful protective factors for cardiovascular health.",
            "icon": "✅"
        })
    if sys < 120 and patient["diaBP"] < 80:
        factors.append({
            "factor": "Normal Blood Pressure",
            "value": f"{sys:.0f}/{patient['diaBP']:.0f} mmHg",
            "status": "good",
            "impact": "Protective",
            "detail": "Optimal BP (<120/80 mmHg) significantly lowers CHD risk.",
            "icon": "✅"
        })

    # ── Recommendations ──────────────────────────────────────────────────────
    recommendations = []
    if cigs > 0:
        recommendations.append("Quit smoking — this is the single highest-impact modifiable risk factor")
    if sys >= 130:
        recommendations.append("Consult a doctor about blood pressure management (lifestyle + medication)")
    if chol > 200:
        recommendations.append("Adopt a heart-healthy diet (reduce saturated fats, increase fibre)")
    if bmi >= 25:
        recommendations.append("Target gradual weight loss — even 5kg can lower BP and CHD risk")
    if gluc > 100 or patient["diabetes"] == 1:
        recommendations.append("Monitor blood sugar regularly and maintain low glycaemic diet")
    if hr > 80:
        recommendations.append("Regular aerobic exercise (150 min/week) lowers resting heart rate")
    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle with regular check-ups")
        recommendations.append("Annual cardiovascular screening is recommended")

    return factors, recommendations


@risk_bp.route("/risk-analysis", methods=["GET", "POST"])
def risk_analysis():
    result = None
    error  = None

    if request.method == "POST":
        try:
            # ── Parse form inputs ─────────────────────────────────────────────
            age             = float(request.form["age"])
            male            = float(request.form["sex"])
            cigsPerDay      = float(request.form["cigsPerDay"])
            BPMeds          = float(request.form["BPMeds"])
            prevalentStroke = float(request.form["prevalentStroke"])
            prevalentHyp    = float(request.form["prevalentHyp"])
            diabetes        = float(request.form["diabetes"])
            totChol         = float(request.form["totChol"])
            sysBP           = float(request.form["sysBP"])
            diaBP           = float(request.form["diaBP"])
            BMI             = float(request.form["BMI"])
            heartRate       = float(request.form["heartRate"])
            glucose         = float(request.form["glucose"])

            # ── Derived features (must match training pipeline) ───────────────
            currentSmoker  = 1 if cigsPerDay > 0 else 0
            pulsePressure  = sysBP - diaBP
            isObese        = 1 if BMI >= 30 else 0
            highChol       = 1 if totChol > 240 else 0
            hyperStage     = 2 if sysBP >= 140 else (1 if sysBP >= 130 else 0)
            ageGroup       = 3 if age >= 70 else (2 if age >= 55 else (1 if age >= 40 else 0))

            patient = {
                "age": age, "male": male, "cigsPerDay": cigsPerDay,
                "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
                "prevalentHyp": prevalentHyp, "diabetes": diabetes,
                "totChol": totChol, "sysBP": sysBP, "diaBP": diaBP,
                "BMI": BMI, "heartRate": heartRate, "glucose": glucose,
                "currentSmoker": currentSmoker, "pulsePressure": pulsePressure,
                "isObese": isObese, "highChol": highChol,
                "hyperStage": hyperStage, "ageGroup": ageGroup,
            }

            # ── Build feature vector in correct training order ────────────────
            X_raw  = np.array([[patient.get(col, 0.0) for col in FEATURE_COLS]])
            X_sc   = scaler.transform(X_raw)

            # ── Predict ───────────────────────────────────────────────────────
            prob        = float(model.predict_proba(X_sc)[0][1])
            probability = round(prob * 100, 2)

            # ── Risk level ────────────────────────────────────────────────────
            if prob >= 0.6:
                risk_level = "High Risk"
                risk_color = "critical"
            elif prob >= 0.35:
                risk_level = "Moderate Risk"
                risk_color = "warning"
            else:
                risk_level = "Low Risk"
                risk_color = "safe"

            # ── Detailed explanations ─────────────────────────────────────────
            factors, recommendations = build_detailed_explanation(patient, prob)

            # ── ITE from meta-learners ────────────────────────────────────────
            ensemble_ite, ite_breakdown = compute_ite(patient)

            if ensemble_ite is not None:
                if ensemble_ite > 0.05:
                    ite_text = f"Smoking causally INCREASES this patient's CHD risk by ~{ensemble_ite*100:.1f}%"
                    ite_color = "danger"
                elif ensemble_ite > 0.01:
                    ite_text = f"Smoking has a moderate causal effect on CHD risk (~{ensemble_ite*100:.1f}% increase)"
                    ite_color = "warning"
                elif ensemble_ite > -0.01:
                    ite_text = f"Smoking has minimal direct causal impact for this patient profile (~{ensemble_ite*100:.1f}%)"
                    ite_color = "normal"
                else:
                    ite_text = f"Unusual result — smoking appears to decrease risk by {abs(ensemble_ite)*100:.1f}% (may indicate confounding)"
                    ite_color = "normal"
            else:
                ite_text  = "ITE not available — run step 11 to train meta-learners"
                ite_color = "normal"

            result = {
                "probability"    : probability,
                "risk_level"     : risk_level,
                "risk_color"     : risk_color,
                "factors"        : factors,
                "recommendations": recommendations,
                "ensemble_ite"   : ensemble_ite,
                "ite_breakdown"  : ite_breakdown,
                "ite_text"       : ite_text,
                "ite_color"      : ite_color,
                "is_smoker"      : currentSmoker == 1,
                "patient"        : patient,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"Prediction error: {str(e)}"

    return render_template("risk_analysis.html", result=result, error=error)