from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os, json
from joblib import load

APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(os.path.dirname(APP_DIR), 'models')

def safe_load(path):
    return load(path) if os.path.exists(path) else None

model_rn = safe_load(os.path.join(MODELS_DIR, 'rn_180d.joblib'))
model_os180 = safe_load(os.path.join(MODELS_DIR, 'os_180d.joblib'))
model_os365 = safe_load(os.path.join(MODELS_DIR, 'os_365d.joblib'))

app = FastAPI(title="Brain Mets Digital Twin — MVP")

class RNFeatures(BaseModel):
    margin_dose: Optional[float] = None
    v12_cc: Optional[float] = None
    largest_target_volume_cc: Optional[float] = None
    num_targets: Optional[float] = None
    piv_cc: Optional[float] = None
    beam_on_time_min: Optional[float] = None

class OSFeatures(BaseModel):
    age: Optional[float] = None
    baseline_kps: Optional[float] = None
    sex: Optional[str] = None
    primary_histology: Optional[str] = None
    extracranial_disease: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "rn_loaded": model_rn is not None, "os180_loaded": model_os180 is not None, "os365_loaded": model_os365 is not None}

@app.get("/schema")
def schema():
    return {
        "rn_features": list(RNFeatures.model_fields.keys()),
        "os_features": list(OSFeatures.model_fields.keys())
    }

@app.post("/predict/rn")
def predict_rn(x: RNFeatures):
    if model_rn is None:
        return {"error": "RN model not loaded. Train it first."}
    import pandas as pd
    df = pd.DataFrame([x.model_dump()])
    p = float(model_rn.predict_proba(df)[0,1])
    return {"prob_rn_180d": p}

@app.post("/predict/os180")
def predict_os_180(x: OSFeatures):
    if model_os180 is None:
        return {"error": "OS(180d) model not loaded. Train it first."}
    import pandas as pd
    df = pd.DataFrame([x.model_dump()])
    p = float(model_os180.predict_proba(df)[0,1])
    return {"prob_death_180d": p}

@app.post("/predict/os365")
def predict_os_365(x: OSFeatures):
    if model_os365 is None:
        return {"error": "OS(365d) model not loaded. Train it first."}
    import pandas as pd
    df = pd.DataFrame([x.model_dump()])
    p = float(model_os365.predict_proba(df)[0,1])
    return {"prob_death_365d": p}
