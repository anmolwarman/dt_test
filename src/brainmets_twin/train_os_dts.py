# src/brainmets_twin/train_os_dts.py
# Python 3.9-compatible trainer for OS DTS (discrete-time survival)
# Usage:
#   python -m src.brainmets_twin.train_os_dts \
#     --patients data/patients.parquet \
#     --plans data/plans_enriched.parquet \
#     --model_out models/os_dts_cv.joblib \
#     --charts_dir charts

import os
import json
import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

import joblib

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc


# -----------------------------
# Global config
# -----------------------------
INTERVALS: List[int] = [3, 6, 9, 12, 15, 18, 21, 24]
RANDOM_STATE: int = 42


# -----------------------------
# Helpers
# -----------------------------
def months_to_days(m: float) -> float:
    """Convert months to days using 365.25/12."""
    return float(m) * 30.4375


def hygiene_drop(
    X: pd.DataFrame, min_nonnull_frac: float = 0.05
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Drop unusable columns (all-missing, very high missing rate, constant numerical columns).
    Returns cleaned X and a dict describing drops.
    """
    dropped = {"all_missing": [], "high_missing": [], "low_variance_num": []}

    # All-missing columns
    for c in list(X.columns):
        if X[c].isna().all():
            dropped["all_missing"].append(c)
    X = X.drop(columns=dropped["all_missing"], errors="ignore")

    # High missing rate columns
    if len(X) > 0:
        nnfrac = 1.0 - (X.isna().sum() / len(X))
        hi_missing = nnfrac[nnfrac < min_nonnull_frac].index.tolist()
    else:
        hi_missing = []
    dropped["high_missing"] = hi_missing
    X = X.drop(columns=hi_missing, errors="ignore")

    # Low-variance numeric columns (std == 0)
    num_cols = X.select_dtypes(include=[np.number]).columns
    low_var = []
    for c in num_cols:
        s = X[c].dropna()
        if len(s) > 0 and float(np.nanstd(s.values)) == 0.0:
            low_var.append(c)
    dropped["low_variance_num"] = low_var
    X = X.drop(columns=low_var, errors="ignore")

    return X, dropped


def recompute_interval_labels(
    patients_df: pd.DataFrame, intervals: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Add y_{m}m columns to patients_df based on days_to_death_from_gk and total_fu_time.
    y=1 if death_time <= threshold
    y=0 if followup >= threshold and (no death or death_time > threshold)
    y=NaN otherwise (censored before threshold)
    """
    if intervals is None:
        intervals = INTERVALS

    DF = patients_df.copy()
    tdeath = pd.to_numeric(DF.get("days_to_death_from_gk"), errors="coerce")
    fu = pd.to_numeric(DF.get("total_fu_time"), errors="coerce")

    for m in intervals:
        thr = months_to_days(m)
        pos = tdeath.notna() & (tdeath <= thr)
        neg = fu.notna() & (fu >= thr) & ((tdeath.isna()) | (tdeath > thr))
        y = pd.Series(np.nan, index=DF.index, dtype=float)
        y[pos] = 1.0
        y[neg] = 0.0
        DF["y_%dm" % m] = y

    return DF


def _name_looks_like_leak(col_lower: str) -> bool:
    """
    Heuristic filter for leakage-prone columns:
    - survival/outcome signals
    - follow-up/censoring
    - explicit labels
    - post-event treatments (subseq_*)
    - obviously post-index KPS/etc (_fu)
    """
    if col_lower in {"y", "death"}:
        return True

    # direct survival fields and their variants
    patterns = [
        "days_to_death",      # includes days_to_death_from_gk/_diag
        "cause_of_death",
        "total_fu_time",      # follow-up duration
        "follow_up", "followup", "_fu",  # any FU derived
        "label_",             # any precomputed labels
        "y_",
        "oof", "auc",         # any evaluation artifacts
        "subseq_",            # post-index interventions
        "kps_last",           # post-index KPS
        "rn_any_ever",        # outcomes after index
    ]
    return any(p in col_lower for p in patterns)


def build_interval_dataset(
    patients_df: pd.DataFrame,
    plans_enriched_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    intervals: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Construct the interval-expanded dataset used for DTS training/eval.

    Returns:
      X          : DataFrame (one row per patient-interval) with features in final order
      y          : ndarray of binary labels per row (NaN rows not yet filtered)
      groups     : ndarray of patient_identifier for GroupKFold
      feat_list  : the feature list actually used
      meta_rows  : DataFrame with ['patient_identifier','horizon_m'] aligned with X/y
    """
    if intervals is None:
        intervals = INTERVALS

    # Merge patient-level with plan-aggregates
    DF = patients_df.merge(plans_enriched_df, on="patient_identifier", how="left")

    # Build labels on the patient table (so per-interval blocks can pull them)
    DF = recompute_interval_labels(DF, intervals)

    # Expand per-interval
    blocks = []
    for m in intervals:
        b = DF.copy()
        b["interval_months"] = float(m)
        b["y"] = DF["y_%dm" % m].values
        b["horizon_m"] = m
        blocks.append(b)
    Wide = pd.concat(blocks, ignore_index=True)

    # Decide on features: if not provided, take numeric columns (excluding labels + survival/outcome fields) + interval_months
    if feature_names is None:
        candidates = []
        for c in Wide.columns:
            if pd.api.types.is_numeric_dtype(Wide[c]):
                cl = c.lower()
                if not _name_looks_like_leak(cl):
                    candidates.append(c)
        # Always include interval_months (already numeric & safe)
        if "interval_months" not in candidates:
            candidates.append("interval_months")
        feature_names = sorted(set(candidates))

    # Build X in declared order
    X = pd.DataFrame(index=Wide.index, columns=feature_names, dtype=float)
    for c in feature_names:
        if c in Wide.columns:
            X[c] = pd.to_numeric(Wide[c], errors="coerce")
        else:
            X[c] = np.nan

    y = Wide["y"].values
    groups = Wide["patient_identifier"].values
    meta_rows = Wide[["patient_identifier", "horizon_m"]].copy()

    return X, y, groups, feature_names, meta_rows


# Backwards-compat alias (some scripts import this name)
def make_interval_dataset(
    patients_df: pd.DataFrame,
    plans_enriched_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    intervals: Optional[List[int]] = None,
):
    return build_interval_dataset(patients_df, plans_enriched_df, feature_names, intervals)


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, out_png: str, title: str) -> float:
    """Plot ROC and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label="OOF (AUC=%.3f)" % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    return float(roc_auc)


# -----------------------------
# Train CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Train OS DTS (Discrete-Time Survival) with GroupKFold OOF.")
    ap.add_argument("--patients", required=True, help="Path to patients.parquet")
    ap.add_argument("--plans", required=True, help="Path to plans_enriched.parquet")
    ap.add_argument("--model_out", default="models/os_dts_cv.joblib", help="Output model bundle path")
    ap.add_argument("--charts_dir", default="charts", help="Where to write ROC/metrics")
    ap.add_argument("--min_nonnull_frac", type=float, default=0.05, help="Drop columns with < this non-null fraction")
    ap.add_argument("--folds", type=int, default=5, help="GroupKFold splits")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.charts_dir, exist_ok=True)

    # Load data
    patients = pd.read_parquet(args.patients)
    plans = pd.read_parquet(args.plans)

    # Build interval-expanded dataset (auto feature list first pass with leakage guard)
    X_raw, y_raw, groups_raw, feat_list, meta_rows = build_interval_dataset(
        patients, plans, feature_names=None, intervals=INTERVALS
    )

    # Hygiene drops to stabilize features
    X_clean, dropped = hygiene_drop(X_raw, min_nonnull_frac=args.min_nonnull_frac)
    feature_names_final = list(X_clean.columns)

    # Keep only rows with defined labels
    mask = ~np.isnan(y_raw)
    X = X_clean.loc[mask].reset_index(drop=True)
    y = y_raw[mask].astype(int)
    groups = groups_raw[mask]
    meta = meta_rows.loc[mask].reset_index(drop=True)

    # GroupKFold setup
    unique_patients = pd.unique(groups)
    n_unique = len(unique_patients)
    n_splits = max(2, min(args.folds, n_unique))
    if n_unique < 2:
        raise RuntimeError("Need at least 2 unique patients for GroupKFold; found %d." % n_unique)

    # Pipeline (median impute -> standardize -> logistic)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=200, fit_intercept=True,
            random_state=RANDOM_STATE
        )),
    ])

    # Grouped OOF predictions by patient
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(y), dtype=float)

    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        Xtr, ytr = X.iloc[tr], y[tr]
        Xte = X.iloc[te]
        pipe.fit(Xtr, ytr)
        oof_pred[te] = pipe.predict_proba(Xte)[:, 1]
        print("[fold %d/%d] train=%d test=%d" % (i, n_splits, len(tr), len(te)))

    # Save OOF table for auditing
    oof_df = meta.copy()
    oof_df["y"] = y
    oof_df["y_pred"] = oof_pred
    oof_path = os.path.join(args.charts_dir, "os_dts_oof.parquet")
    oof_df.to_parquet(oof_path, index=False)

    # Pooled OOF ROC
    roc_png = os.path.join(args.charts_dir, "roc_dts_interval_oof.png")
    auc_oof = plot_roc(y, oof_pred, roc_png, "ROC — DTS Interval-level OOF (GroupKFold)")

    # Fit final model on all rows (for deployment)
    pipe.fit(X, y)

    # Save bundle with metadata to ensure reproducibility
    bundle = {
        "model": pipe,
        "feature_names": feature_names_final,
        "intervals": INTERVALS,
        "dropped_columns": dropped,
        "auc_oof": float(auc_oof),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(bundle, args.model_out)

    # Metrics JSON
    with open(os.path.join(args.charts_dir, "roc_dts_interval_oof.json"), "w") as f:
        json.dump({
            "auc_oof": float(auc_oof),
            "n_rows": int(len(y)),
            "n_features": int(len(feature_names_final)),
            "intervals": INTERVALS,
            "dropped_columns": dropped
        }, f, indent=2)

    # Features JSON (manifest)
    features_json_path = os.path.join(args.charts_dir, "os_dts_features.json")
    with open(features_json_path, "w") as f:
        json.dump({
            "feature_names": feature_names_final,
            "intervals": INTERVALS,
            "dropped_columns": dropped
        }, f, indent=2)

    print("\n---- Summary ----")
    print("AUC (OOF pooled): %.3f" % auc_oof)
    print("OOF table        :", oof_path)
    print("ROC plot         :", roc_png)
    print("Model saved      :", args.model_out)
    print("Features JSON    :", features_json_path)


if __name__ == "__main__":
    main()
