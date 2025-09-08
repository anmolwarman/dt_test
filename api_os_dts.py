# api_os_dts.py — FastAPI backend for OS Digital Twin (Python 3.9 compatible)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os

# ---------- Config ----------
ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "os_dts_cv.joblib")
PATIENTS_PARQUET = os.path.join(ROOT, "data", "patients.parquet")
PLANS_PARQUET = os.path.join(ROOT, "data", "plans_enriched.parquet")

# ---------- App ----------
app = FastAPI(title="BrainMets Digital Twin API (OS DTS)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------- Load model bundle ----------
_bundle = joblib.load(MODEL_PATH)

def _get_from_bundle(bundle: Any, keys: List[str], default=None):
    if isinstance(bundle, dict):
        for k in keys:
            if k in bundle:
                return bundle[k]
    return default

_MODEL = _get_from_bundle(_bundle, ["model", "pipe", "estimator"], default=_bundle)
_COLS = _get_from_bundle(_bundle, ["columns", "feature_names", "cols"], default=None)
_INTERVALS = _get_from_bundle(_bundle, ["intervals", "interval_months"], default=[3,6,9,12,15,18,21,24])

if _COLS is None:
    try:
        _COLS = _MODEL.feature_names_in_.tolist()
    except Exception:
        raise RuntimeError("Model columns missing; re-train and save with column metadata.")

# ---------- Load data ----------
_PAT = pd.read_parquet(PATIENTS_PARQUET)
try:
    _PLN = pd.read_parquet(PLANS_PARQUET)
except Exception:
    _PLN = None

_NUMERIC_PLN_COLS = [c for c in _PLN.columns if pd.api.types.is_numeric_dtype(_PLN[c])] if _PLN is not None else []
_EXCLUDE = {"label_os_180d", "label_os_365d", "label_os_any", "patient_identifier"}

# ---------- Pydantic payloads ----------
class PlanRecord(BaseModel):
    margin_dose: Optional[float] = None
    beam_on_time_min: Optional[float] = None
    total_vol_all_targets: Optional[float] = None
    max_dose_largest: Optional[float] = None
    num_gk_targets: Optional[float] = None
    total_num_tumors_gk: Optional[float] = None
    num_lobar_gk: Optional[float] = None
    num_brainstem_gk: Optional[float] = None
    num_thal_bg_gk: Optional[float] = None
    num_cerebellum_gk: Optional[float] = None
    num_skull_gk: Optional[float] = None
    max_linear_dimension_largest: Optional[float] = None
    largest_target_volume_cc: Optional[float] = None
    piv_cc_largest: Optional[float] = None
    v12_cc_largest: Optional[float] = None
    num_isocenters_largest: Optional[float] = None
    margin_dose_largest: Optional[float] = None
    percent_isodose_largest: Optional[float] = None
    pci_largest: Optional[float] = None
    gri_largest: Optional[float] = None
    min_dose_largest: Optional[float] = None
    mean_dose_largest: Optional[float] = None

class PatientPayload(BaseModel):
    patient_identifier: str = "new_patient"
    age: Optional[float] = None
    primary_histology: Optional[float] = None
    extracranial_disease: Optional[float] = None
    kps_pre_gk: Optional[float] = None
    total_fu_time: Optional[float] = None
    plans: List[Dict[str, Any]] = []

# ---------- Helpers ----------
def _aggregate_plans_for_patient(pid: str) -> pd.Series:
    if _PLN is None:
        return pd.Series(dtype=float)
    sub = _PLN[_PLN["patient_identifier"].astype(str) == str(pid)]
    if sub.empty:
        return pd.Series(dtype=float)
    num_cols = [c for c in _NUMERIC_PLN_COLS if c not in _EXCLUDE and c in sub.columns]
    if not num_cols:
        return pd.Series(dtype=float)
    aggs = {}
    for c in num_cols:
        lc = c.lower()
        if lc.startswith("num_") or lc.startswith("total_") or lc.endswith("_count"):
            aggs[c] = ["sum", "max"]
        else:
            aggs[c] = ["mean", "max"]
    G = sub[num_cols].agg(aggs)
    G.columns = [f"{a}_{b}" for a, b in G.columns]
    return G.astype(float)

def _expand_and_align(row_like: pd.Series) -> pd.DataFrame:
    DF = pd.concat([row_like.to_frame().T.assign(interval_months=m) for m in _INTERVALS], ignore_index=True)
    X = DF.select_dtypes(include=[np.number]).drop(columns=[c for c in DF.columns if c in _EXCLUDE], errors="ignore")
    for c in _COLS:
        if c not in X.columns:
            X[c] = np.nan
    return X[_COLS]

def _build_feature_frame_from_existing(pid: str) -> pd.DataFrame:
    Psub = _PAT[_PAT["patient_identifier"].astype(str) == str(pid)]
    if Psub.empty:
        raise HTTPException(status_code=404, detail=f"patient_id {pid} not found")
    row = Psub.iloc[0].copy()
    agg = _aggregate_plans_for_patient(pid)
    for c, v in agg.items():
        row[c] = v
    return _expand_and_align(row)

def _build_feature_frame_from_payload(payload: PatientPayload) -> pd.DataFrame:
    base = pd.Series(index=_PAT.columns, dtype=float)
    base["patient_identifier"] = payload.patient_identifier
    for k, v in payload.dict().items():
        if k in base.index and v is not None and not isinstance(v, list):
            base[k] = v
    if payload.plans:
        PL = pd.DataFrame(payload.plans)
        if not PL.empty:
            ncols = [c for c in PL.columns if pd.api.types.is_numeric_dtype(PL[c])]
            aggs = {}
            for c in ncols:
                lc = c.lower()
                if lc.startswith("num_") or lc.startswith("total_") or lc.endswith("_count"):
                    aggs[c] = ["sum", "max"]
                else:
                    aggs[c] = ["mean", "max"]
            if aggs:
                G = PL.agg(aggs)
                G.columns = [f"{a}_{b}" for a, b in G.columns]
                for c in G.index:
                    base[c] = float(G.loc[c])
    return _expand_and_align(base)

# ---------- Global medians for safe imputation ----------
def _compute_global_medians() -> pd.Series:
    try:
        rows = []
        for pid in _PAT["patient_identifier"].astype(str).unique().tolist():
            try:
                X = _build_feature_frame_from_existing(pid)
                rows.append(X)
            except Exception:
                continue
        if rows:
            A = pd.concat(rows, ignore_index=True)
            meds = A.median(numeric_only=True)
            meds = meds.reindex(_COLS)
            return meds
    except Exception:
        pass
    return pd.Series(index=_COLS, dtype=float)

_MEDIANS = _compute_global_medians()

def _impute_with_medians(X: pd.DataFrame) -> pd.DataFrame:
    Y = X.copy()
    if _MEDIANS is not None and not _MEDIANS.empty:
        # align then fill
        for c in _COLS:
            if c not in Y.columns:
                Y[c] = np.nan
        Y = Y[_COLS]
        Y = Y.fillna(_MEDIANS)
    else:
        Y = Y.fillna(0.0)
    # final safety
    Y = Y.replace([np.inf, -np.inf], 0.0)
    return Y

def _predict_curve_fromX(X: pd.DataFrame) -> Dict[str, Any]:
    X = _impute_with_medians(X)
    h = _MODEL.predict_proba(X)[:, 1]  # per-interval hazard
    S = np.cumprod(1.0 - h)
    risk = 1.0 - S
    return {"months": _INTERVALS, "hazard": h.tolist(), "survival": S.tolist(), "cum_mortality": risk.tolist()}

# ---------- Routes ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "n_features": len(_COLS),
        "intervals": _INTERVALS,
        "patients": int(_PAT["patient_identifier"].nunique()),
        "has_plans": _PLN is not None,
        "medians_nonnull": int(_MEDIANS.notna().sum()),
    }

@app.get("/patients")
def list_patients(limit: int = 200, valid: int = 0):
    ids = _PAT["patient_identifier"].astype(str).dropna().unique().tolist()
    if not valid:
        return {"count": len(ids[:limit]), "ids": ids[:limit]}
    ok = []
    for pid in ids:
        try:
            X = _build_feature_frame_from_existing(pid)
            X = _impute_with_medians(X)
            _ = _MODEL.predict_proba(X.iloc[[0]])
            ok.append(pid)
        except Exception:
            continue
        if len(ok) >= limit:
            break
    # Fallback: if none validated, return unfiltered list so UI isn't empty
    if not ok:
        return {"count": len(ids[:limit]), "ids": ids[:limit]}
    return {"count": len(ok), "ids": ok}

@app.get("/os_curve/{patient_id}")
def os_curve_existing(patient_id: str):
    X = _build_feature_frame_from_existing(patient_id)
    out = _predict_curve_fromX(X)
    out["patient_id"] = patient_id
    return out

@app.post("/os_curve_from_payload")
def os_curve_from_payload(payload: PatientPayload):
    X = _build_feature_frame_from_payload(payload)
    out = _predict_curve_fromX(X)
    out["patient_id"] = payload.patient_identifier
    return out
