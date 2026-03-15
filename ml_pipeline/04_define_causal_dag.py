"""
Step 4: Define Causal DAG
Full clinical DAG matching the project diagram:
  Age / Sex  -->  Blood Pressure, Cholesterol
  Smoking    -->  Blood Pressure, Heart Disease   (TREATMENT)
  Diet       -->  Cholesterol, Blood Sugar
  Blood Pressure --> Shortness of Breath, Chest Pain, Heart Disease
  Physical Activity --> BMI  -->  Blood Pressure, Heart Disease
  Blood Sugar --> Resting ECG, Heart Disease
  Symptoms   --> Heart Disease   (OUTCOME)
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    import pandas as pd
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── 1. Causal graph (GML / dot string for DoWhy) ─────────────────────────────
CAUSAL_GRAPH = """
digraph {
    age           -> sysBP;
    age           -> totChol;
    age           -> TenYearCHD;
    male          -> sysBP;
    male          -> totChol;
    male          -> TenYearCHD;
    currentSmoker -> sysBP;
    currentSmoker -> totChol;
    currentSmoker -> TenYearCHD;
    education     -> totChol;
    education     -> glucose;
    BMI           -> sysBP;
    BMI           -> TenYearCHD;
    glucose       -> diabetes;
    glucose       -> prevalentStroke;
    glucose       -> TenYearCHD;
    diabetes      -> TenYearCHD;
    sysBP         -> prevalentHyp;
    sysBP         -> heartRate;
    sysBP         -> TenYearCHD;
    diaBP         -> TenYearCHD;
    totChol       -> TenYearCHD;
    prevalentHyp  -> TenYearCHD;
    heartRate     -> TenYearCHD;
    prevalentStroke -> TenYearCHD;
    BPMeds        -> sysBP;
    BPMeds        -> diaBP;
}
"""

print("=" * 60)
print("CAUSAL DAG DEFINITION")
print("=" * 60)

# ── 2. DoWhy CausalModel ─────────────────────────────────────────────────────
if DOWHY_AVAILABLE:
    df = pd.read_csv(os.path.join(DATA_DIR, "framingham_engineered.csv"))
    model = CausalModel(
        data=df, treatment="currentSmoker",
        outcome="TenYearCHD", graph=CAUSAL_GRAPH
    )
    print("\n[DoWhy] CausalModel built.  Treatment=currentSmoker  Outcome=TenYearCHD")
    try:
        model.view_model(layout="dot")
        print("[DoWhy] Graph image saved (graphviz).")
    except Exception:
        print("[DoWhy] graphviz not available – skipping DoWhy render.")
else:
    print("[warn] DoWhy not installed – skipping CausalModel.")

# ── 3. Matplotlib DAG image ───────────────────────────────────────────────────
NODE_DEFS = {
    "age":               (4.5,  13.5, "Age",                 "#F5A020"),
    "sex":               (8.5,  13.5, "Sex",                 "#5B9FD5"),
    "smoking":           (1.3,  10.5, "Smoking",             "#E74C3C"),
    "diet":              (11.7, 10.5, "Diet",                "#27AE60"),
    "blood_pressure":    (4.0,  10.0, "Blood\nPressure",     "#C0392B"),
    "cholesterol":       (8.0,  10.0, "Cholesterol",         "#E67E22"),
    "physical_activity": (1.3,   7.2, "Physical\nActivity",  "#16A085"),
    "blood_sugar":       (11.7,  7.8, "Blood\nSugar",        "#8E44AD"),
    "shortness_breath":  (4.0,   6.0, "Shortness\nof Breath","#2980B9"),
    "chest_pain":        (8.0,   6.0, "Chest\nPain",         "#E91E8C"),
    "bmi":               (1.3,   4.5, "BMI",                 "#2C3E50"),
    "resting_ecg":       (11.7,  5.0, "Resting\nECG Result", "#6C3483"),
    "heart_disease":     (6.5,   2.2, "Heart\nDisease",      "#C0392B"),
}
EDGE_DEFS = [
    ("age","blood_pressure",False), ("age","cholesterol",False),
    ("sex","blood_pressure",False), ("sex","cholesterol",False), ("sex","heart_disease",False),
    ("smoking","blood_pressure",False), ("smoking","heart_disease",False),
    ("diet","cholesterol",False), ("diet","blood_sugar",False),
    ("blood_pressure","shortness_breath",False), ("blood_pressure","chest_pain",False),
    ("blood_pressure","heart_disease",False),
    ("cholesterol","heart_disease",False),
    ("physical_activity","bmi",False), ("physical_activity","blood_pressure",True),
    ("blood_sugar","resting_ecg",False), ("blood_sugar","heart_disease",False),
    ("shortness_breath","heart_disease",False), ("chest_pain","heart_disease",False),
    ("bmi","blood_pressure",True), ("bmi","heart_disease",True),
    ("resting_ecg","heart_disease",False),
]
RAD = {n: (0.72 if n == "heart_disease" else 0.58) for n in NODE_DEFS}

fig, ax = plt.subplots(figsize=(13,15))
ax.set_xlim(0,13); ax.set_ylim(0,15); ax.axis('off')
fig.patch.set_facecolor('#F4F6FB'); ax.set_facecolor('#F4F6FB')

def ctr(n): return NODE_DEFS[n][0], NODE_DEFS[n][1]

for src, dst, dsh in EDGE_DEFS:
    x1,y1=ctr(src); x2,y2=ctr(dst)
    dx,dy=x2-x1,y2-y1; d=np.hypot(dx,dy)
    sx,sy=x1+(dx/d)*RAD[src], y1+(dy/d)*RAD[src]
    ex,ey=x2-(dx/d)*RAD[dst], y2-(dy/d)*RAD[dst]
    col="#AAAAAA" if dsh else "#6A8EAE"; lw=1.4 if dsh else 1.9; ls="--" if dsh else "-"
    ax.annotate("",xy=(ex,ey),xytext=(sx,sy),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=lw,linestyle=ls,mutation_scale=16))

for name,(x,y,lbl,bg) in NODE_DEFS.items():
    r=RAD[name]
    ax.add_patch(Circle((x+.06,y-.06),r,color='#00000020',zorder=3))
    ax.add_patch(Circle((x,y),r,color=bg,zorder=4,lw=2.5,ec='white'))
    lines=lbl.split('\n')
    if len(lines)==1:
        ax.text(x,y,lbl,ha='center',va='center',fontsize=9,fontweight='bold',color='white',zorder=5)
    else:
        for i,ln in enumerate(lines):
            ax.text(x,y+.16-i*.33,ln,ha='center',va='center',fontsize=8,fontweight='bold',color='white',zorder=5)

ax.text(6.5,14.65,"Causal DAG",ha='center',fontsize=23,fontweight='bold',color='#2C3E50')
ax.text(6.5,14.15,"for  Heart Disease",ha='center',fontsize=21,fontweight='bold',color='#E91E8C')

for i,(col,lbl,ls) in enumerate([("#6A8EAE","Direct causal link",'-'),("#AAAAAA","Contributing / indirect link",'--')]):
    lx,ly=0.25, 1.3-i*.5
    ax.annotate("",xy=(lx+.8,ly),xytext=(lx,ly),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=1.8,linestyle=ls,mutation_scale=13))
    ax.text(lx+.95,ly,lbl,va='center',fontsize=8.5,color='#2C3E50')

key=[("#F5A020","Demographic"),("#E74C3C","Lifestyle/Treatment"),("#C0392B","Clinical/Outcome"),("#2980B9","Symptom"),("#8E44AD","Biomarker")]
for i,(col,lbl) in enumerate(key):
    cx=5.8+(i%3)*2.5; cy=1.1-(i//3)*.5
    ax.add_patch(Circle((cx,cy),.13,color=col,zorder=5))
    ax.text(cx+.22,cy,lbl,va='center',fontsize=7.5,color='#2C3E50')

ax.text(6.5,.22,"Causal DAG — Clinical Domain Knowledge  |  MCA Major Project",
        ha='center',fontsize=8,color='#95A5A6',style='italic')

plt.tight_layout(pad=0.2)
img_path = os.path.join(DATA_DIR, "causal_dag_heart_disease.png")
plt.savefig(img_path, dpi=180, bbox_inches='tight', facecolor='#F4F6FB')
plt.close()
print(f"\n[Image] Causal DAG saved  →  {img_path}")

# ── 4. Print edge summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CAUSAL EDGE SUMMARY")
print("=" * 60)
for src,dst,reason in [
    ("age",            "sysBP / totChol",   "BP & cholesterol rise with age"),
    ("sex (male)",     "sysBP / totChol",   "Biological sex modulates lipid and BP levels"),
    ("currentSmoker",  "sysBP / totChol",   "Smoking raises BP and LDL cholesterol"),
    ("currentSmoker",  "TenYearCHD",        "DIRECT CAUSAL PATH — treatment → outcome"),
    ("education",      "totChol / glucose", "Diet proxy; affects lipids & blood sugar"),
    ("BMI",            "sysBP",             "Obesity elevates blood pressure"),
    ("BMI",            "TenYearCHD",        "High BMI is an independent CHD risk factor"),
    ("glucose",        "diabetes",          "Elevated glucose → Type 2 diabetes"),
    ("sysBP",          "prevalentHyp",      "Persistent high BP → hypertension (Shortness of Breath)"),
    ("sysBP",          "heartRate",         "Hypertension elevates resting heart rate (Chest Pain)"),
    ("glucose",        "prevalentStroke",   "High blood sugar → vascular events (ECG abnormality)"),
    ("diabetes",       "TenYearCHD",        "Diabetes multiplies CHD risk"),
    ("totChol",        "TenYearCHD",        "Elevated LDL → atherosclerosis → CHD"),
    ("prevalentHyp",   "TenYearCHD",        "Hypertension → cardiac damage → CHD"),
    ("heartRate",      "TenYearCHD",        "Elevated HR (Chest Pain pathway) → CHD"),
    ("prevalentStroke","TenYearCHD",        "Prior vascular event (ECG) → CHD"),
]:
    print(f"  {src:<22}  →  {dst:<22}: {reason}")