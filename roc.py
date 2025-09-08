# roc_interval_cv.py — Reproduce interval-level OOF ROC (like train_os_dts)
# Run:  python roc_interval_cv.py
# Out:  charts/roc_dts_interval_oof.png

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc

# ---------- paths ----------
ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "models", "os_dts.joblib")
PATIENTS_PARQUET = os.path.join(ROOT, "data", "patients.parquet")
PLANS_PARQUET = os.path.join(ROOT, "data", "plans_enriched.parquet")
OUT_DIR = os.path.join(ROOT, "charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- exact interval dataset builder (same as in training) ----------
def _months_to_days(m):
    return float(m) * 30.4375

def make_interval_dataset(patients_df: pd.DataFrame,
                          plans_enriched_df: pd.DataFrame,
                          feature_names: list,
                          intervals=(3,6,9,12,15,18,21,24)):
    DF = patients_df.merge(plans_enriched_df, on="patient_identifier", how="left")

    tdeath = pd.to_numeric(DF.get("days_to_death_from_gk"), errors="coerce")
    fu     = pd.to_numeric(DF.get("total_fu_time"), errors="coerce")

    Ycols = []
    for m in intervals:
        thr = _months_to_days(m)
        pos = tdeath.notna() & (tdeath <= thr)
        neg = fu.notna() & (fu >= thr) & ((tdeath.isna()) | (tdeath > thr))
        y = pd.Series(np.nan, index=DF.index, dtype=float)
        y[pos] = 1.0
        y[neg] = 0.0
        DF[f"y_{int(m)}m"] = y
        Ycols.append(f"y_{int(m)}m")

    # Expand rows per interval
    blocks = []
    for m in intervals:
        b = DF.copy()
        b["interval_months"] = float(m)
        b["y"] = DF[f"y_{int(m)}m"].values
        blocks.append(b)
    Wide = pd.concat(blocks, ignore_index=True)

    # Build X with model's expected features
    X = pd.DataFrame(index=Wide.index, columns=feature_names, dtype=float)
    for c in feature_names:
        if c in Wide.columns:
            X[c] = pd.to_numeric(Wide[c], errors="coerce")
        elif c == "interval_months":
            X[c] = Wide["interval_months"].astype(float)
        else:
            X[c] = np.nan

    y = Wide["y"].values  # shape (n_rows,)

    # group by patient for GroupKFold
    try:
        groups = Wide["patient_identifier"].values
    except KeyError:
        raise RuntimeError("patient_identifier must be present to group by patient.")

    return X, y, groups

# ---------- main ----------
def main():
    # Load the trained bundle to get feature list & intervals & hyper-params
    bundle = joblib.load(MODEL_PATH)
    model_in_bundle = bundle.get("model", bundle)

    feature_names = bundle.get("feature_names") or bundle.get("columns")
    if feature_names is None:
        try:
            feature_names = model_in_bundle.feature_names_in_.tolist()
        except Exception:
            raise RuntimeError("Cannot resolve model feature names. Save them in the joblib.")
    intervals = bundle.get("intervals") or bundle.get("interval_months") or [3,6,9,12,15,18,21,24]
    intervals = [int(float(m)) for m in intervals]

    # Try to read the logistic hyper-params from bundle; fall back to sensible defaults
    # (If your bundle stored metadata, use it. Otherwise, these match what we used earlier.)
    meta = {}
    for k in ("C", "penalty", "solver", "max_iter", "class_weight", "fit_intercept"):
        if hasattr(model_in_bundle, k):
            meta[k] = getattr(model_in_bundle, k)
    # Defaults (only used if not detectable from the stored model)
    C = meta.get("C", 1.0)
    penalty = meta.get("penalty", "l2")
    solver = meta.get("solver", "lbfgs")
    max_iter = int(meta.get("max_iter", 200))
    class_weight = meta.get("class_weight", "balanced")
    fit_intercept = bool(meta.get("fit_intercept", True))

    print("[config]")
    print(" intervals:", intervals)
    print(" features:", len(feature_names))
    print(" logistic:", dict(C=C, penalty=penalty, solver=solver,
                             max_iter=max_iter, class_weight=class_weight,
                             fit_intercept=fit_intercept))

    # Load data
    P = pd.read_parquet(PATIENTS_PARQUET)
    PLN = pd.read_parquet(PLANS_PARQUET)

    # Build interval dataset
    X, y, groups = make_interval_dataset(P, PLN, feature_names, intervals=intervals)

    # Mask to rows with defined labels
    mask = ~np.isnan(y)
    X = X.loc[mask].reset_index(drop=True)
    y = y[mask].astype(int)
    groups = groups[mask]

    # Build the same pipeline as training (Imputer -> Scaler -> Logistic)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            C=C, penalty=penalty, solver=solver, max_iter=max_iter,
            class_weight=class_weight, fit_intercept=fit_intercept, n_jobs=None, random_state=42
        ))
    ])

    # GroupKFold OOF prediction (5 folds by patient)
    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y), dtype=float)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y[tr]
        pipe.fit(Xtr, ytr)
        oof_pred[te] = pipe.predict_proba(Xte)[:, 1]
        print(f"[fold {fold}] train={len(tr)}  test={len(te)}  done")

    # ROC/AUC on pooled OOF
    fpr, tpr, _ = roc_curve(y, oof_pred)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"OOF pooled intervals (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — DTS Interval-level OOF (GroupKFold)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "roc_dts_interval_oof.png")
    plt.savefig(out_png, dpi=300)
    print(f"[saved] {out_png}  AUC={roc_auc:.3f}")

    # Also save a small JSON for record
    with open(os.path.join(OUT_DIR, "roc_dts_interval_oof.json"), "w") as f:
        json.dump({"auc_oof": float(roc_auc), "n_rows": int(len(y)), "intervals": intervals}, f, indent=2)

if __name__ == "__main__":
    main()