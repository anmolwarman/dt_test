from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any


# ---- Config paths (relative to project root) ----
MODEL_PATH = "models/os_dts.joblib"
PATIENTS_PATH = "data/patients.parquet"
PLANS_PATH = "data/plans_enriched.parquet"

EXCLUDE = {"patient_identifier","days_to_death_from_gk","total_fu_time",
           "label_os_180d","label_os_365d","label_os_any"}

# ---- App ----
app = FastAPI(title="BrainMets Digital Twin API")

# CORS: allow all origins for MVP (tighten later)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ---- Load data/model once ----
_bundle = load(MODEL_PATH)
_MODEL, _COLS, _INTERVALS = _bundle["model"], _bundle["columns"], _bundle["intervals"]

_PAT = pd.read_parquet(PATIENTS_PATH)
_PLN = pd.read_parquet(PLANS_PATH)

def _feature_row(patient_id: str) -> pd.DataFrame:
    pid_str = str(patient_id)
    p = _PAT[_PAT["patient_identifier"].astype(str) == pid_str]
    if p.empty:
        ex = _PAT["patient_identifier"].astype(str).head(10).tolist()
        raise HTTPException(404, f"patient_identifier '{pid_str}' not found. Examples: {ex}")
    row = p.iloc[0].copy()

    pl = _PLN[_PLN["patient_identifier"].astype(str) == pid_str].copy()
    if len(pl):
        num_cols = [c for c in pl.columns if c not in ["patient_identifier","gk_id"]
                    and pd.api.types.is_numeric_dtype(pl[c])]
        aggs = {}
        for c in num_cols:
            if any(k in c for k in ["num_","total_","count"]):
                aggs[c] = ["sum","max"]
            else:
                aggs[c] = ["mean","max"]
        g = pl.groupby("patient_identifier").agg(aggs)
        g.columns = [f"{a}_{b}" for a,b in g.columns]
        g = g.reset_index(drop=True)
        for c in g.columns:
            if c != "patient_identifier":
                row[c] = float(g[c].iloc[0])
    return row.to_frame().T

@app.get("/health")
def health():
    return {"status":"ok","features":len(_COLS),"intervals":_INTERVALS}

@app.get("/os_curve/{patient_id}")
def os_curve(patient_id: str):
    R = _feature_row(patient_id)
    # 1 row per interval
    DF = pd.concat([R.assign(interval_months=m) for m in _INTERVALS], ignore_index=True)

    # numeric + align to training columns
    X = DF.select_dtypes(include=[np.number]).drop(
        columns=[c for c in DF.columns if c in EXCLUDE], errors="ignore"
    )
    for c in _COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[_COLS]

    # predict hazards -> survival & cumulative mortality
    h = _MODEL.predict_proba(X)[:,1]
    S = np.cumprod(1.0 - h)
    risk = 1.0 - S

    return {
        "patient_id": str(patient_id),
        "months": _INTERVALS,
        "hazard": h.tolist(),
        "survival": S.tolist(),
        "cum_mortality": risk.tolist()
    }

# 1) List available patient IDs
@app.get("/patients")
def list_patients(limit: int = 100):
    ids = _PAT["patient_identifier"].astype(str).dropna().unique().tolist()[:limit]
    return {"count": len(ids), "ids": ids}

# 2) Predict from a raw payload (no parquet row required)
class PlanRecord(BaseModel):
    # any numeric plan-level feature you have; optional
    beam_on_time_min: float | None = None
    margin_dose: float | None = None
    # ... add others as needed

class PatientPayload(BaseModel):
    patient_identifier: str = "new_patient"
    # patient-level features
    age: float | None = None
    primary_histology: float | None = None
    extracranial_disease: float | None = None
    kps_pre_gk: float | None = None
    total_fu_time: float | None = None  # optional
    # optional list of plan records to aggregate
    plans: List[Dict[str, Any]] = []

@app.post("/os_curve_from_payload")
def os_curve_from_payload(payload: PatientPayload):
    # start with an empty row using existing PAT columns so we align with training space
    row = pd.Series(index=_PAT.columns, dtype=float)
    row["patient_identifier"] = payload.patient_identifier
    # copy patient fields if names match the training columns
    for k, v in payload.dict().items():
        if k in row.index and v is not None:
            row[k] = v

    # aggregate plan fields if provided
    if payload.plans:
        PLp = pd.DataFrame(payload.plans)
        num_cols = [c for c in PLp.columns if pd.api.types.is_numeric_dtype(PLp[c])]
        aggs = {}
        for c in num_cols:
            if any(k in c for k in ["num_", "total_", "count"]):
                aggs[c] = ["sum", "max"]
            else:
                aggs[c] = ["mean", "max"]
        if aggs:
            g = PLp.agg(aggs)
            g.columns = [f"{a}_{b}" for a, b in g.columns]
            for c in g.index:
                row[c] = float(g.loc[c])

    R = row.to_frame().T  # single-row DF

    DF = pd.concat([R.assign(interval_months=m) for m in _INTERVALS], ignore_index=True)

    # numeric + align
    X = DF.select_dtypes(include=[np.number]).drop(
        columns=[c for c in DF.columns if c in EXCLUDE], errors="ignore"
    )
    for c in _COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[_COLS]

    h = _MODEL.predict_proba(X)[:, 1]
    S = np.cumprod(1.0 - h)
    risk = 1.0 - S
    return {
        "patient_id": payload.patient_identifier,
        "months": _INTERVALS,
        "hazard": h.tolist(),
        "survival": S.tolist(),
        "cum_mortality": risk.tolist(),
    }