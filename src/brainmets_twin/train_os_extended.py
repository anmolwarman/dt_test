import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import dump

# ----------------------------
# Config
# ----------------------------

# Patient-level columns we want to keep (if present)
PATIENT_BASE_VARS = [
    "age", "primary_histology", "extracranial_disease", "kps_pre_gk"
]

# Plan-level numeric features to aggregate up to patient level (subset will be auto-detected)
PLAN_NUM_CANDIDATES = [
    # global plan dose/volume/time
    "margin_dose", "beam_on_time_min", "piv_cc", "v12_cc",
    "largest_target_volume_cc", "num_targets", "total_vol_all_targets",
    # regional counts / burden
    "num_gk_targets", "total_num_tumors_gk",
    "num_lobar_gk", "num_brainstem_gk", "num_thal_bg_gk",
    "num_cerebellum_gk", "num_skull_gk",
    # largest-target geometry/dose
    "max_linear_dimension_largest",
    "xdimension_largest", "ydimension_largest", "zdimension_largest",
    "piv_cc_largest", "v12_cc_largest",
    "margin_dose_largest", "percent_isodose_largest",
    "pci_largest", "gri_largest",
    "max_dose_largest", "min_dose_largest", "mean_dose_largest",
]

# Columns we must never use as features (labels, identifiers, follow-up, etc.)
PATIENT_EXCLUDE = {
    "patient_identifier",
    "label_os_180d", "label_os_365d", "label_os_any",
    "days_to_death_from_gk", "total_fu_time",
    # other potentially leaky aggregates if present
    "kps_last", "kps_min", "kps_max", "rn_any_ever",
}

# Drop rules (preprocessing hygiene)
MISSING_FRACTION_DROP = 0.995  # drop columns with >=99.5% missing
MIN_NUMERIC_UNIQUE    = 2      # drop numeric columns with <2 unique non-null values


# ----------------------------
# Helpers
# ----------------------------

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r'[^0-9eE+\-\.]', '', regex=True),
        errors='coerce'
    )

def agg_plans_to_patient(plans: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-plan features to one row per patient (pre-GK features only)."""
    if plans.empty:
        return pd.DataFrame(columns=["patient_identifier"])

    df = plans.copy()
    if "patient_identifier" not in df.columns:
        raise SystemExit("plans_enriched missing 'patient_identifier'")

    present = [c for c in PLAN_NUM_CANDIDATES if c in df.columns]
    for c in present:
        df[c] = to_num(df[c])

    # burden/counters
    burden_cols = [
        "num_targets", "num_gk_targets", "total_num_tumors_gk",
        "num_lobar_gk", "num_brainstem_gk", "num_thal_bg_gk",
        "num_cerebellum_gk", "num_skull_gk",
        "total_vol_all_targets"
    ]
    burden_cols = [c for c in burden_cols if c in present]

    # dose/volume/time
    dosevol_cols = [c for c in present if c not in burden_cols]

    aggs = {}
    for c in burden_cols:
        aggs[c] = ["sum", "max"]
    for c in dosevol_cols:
        aggs[c] = ["mean", "max"]

    if not aggs:
        return df[["patient_identifier"]].drop_duplicates()

    g = df.groupby("patient_identifier").agg(aggs)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    return g

def drop_bad_cols(X: pd.DataFrame):
    """
    Drop:
      - all-missing columns,
      - ultra-sparse columns (>=99.5% missing),
      - numeric columns with <2 unique non-null values.
    Return cleaned X and a dict of what was dropped.
    """
    dropped = {"all_missing": [], "high_missing": [], "low_variance_num": []}

    # all-missing
    all_missing = [c for c in X.columns if X[c].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)
        dropped["all_missing"] = all_missing

    # ultra-sparse (but not already all-missing)
    high_missing = [c for c in X.columns
                    if X[c].isna().mean() >= MISSING_FRACTION_DROP]
    if high_missing:
        X = X.drop(columns=high_missing)
        dropped["high_missing"] = high_missing

    # numeric low-variance (<=1 unique non-null)
    low_var = []
    for c in X.select_dtypes(include=[np.number]).columns:
        nn = X[c].dropna().nunique()
        if nn < MIN_NUMERIC_UNIQUE:
            low_var.append(c)
    if low_var:
        X = X.drop(columns=low_var)
        dropped["low_variance_num"] = low_var

    return X, dropped

def build_design_matrix(patients: pd.DataFrame,
                        plan_aggs: pd.DataFrame,
                        label: str):
    """Merge base patient vars and aggregated plan features; return X, y, num, cat, dropped."""
    for need in ["patient_identifier", label]:
        if need not in patients.columns:
            raise SystemExit(f"patients.parquet missing '{need}'")

    keep_base = [c for c in PATIENT_BASE_VARS if c in patients.columns]
    P = patients[["patient_identifier", label] + keep_base].copy()

    if not plan_aggs.empty:
        P = P.merge(plan_aggs, on="patient_identifier", how="left")

    y = P[label].dropna().astype(int)
    X = P.loc[y.index].drop(columns=[label])

    # Safety: exclude accidental leaky/identifier columns
    X = X.drop(columns=[c for c in X.columns if c in PATIENT_EXCLUDE], errors="ignore")

    # Hygiene: drop all-missing / ultra-sparse / constant numeric columns
    X, dropped = drop_bad_cols(X)

    # Split numeric vs categorical
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols and X[c].dropna().nunique() <= 50]

    return X, y, num_cols, cat_cols, dropped

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="data/patients.parquet")
    ap.add_argument("--plans", default="data/plans_enriched.parquet")
    ap.add_argument("--label", default="label_os_365d",
                    help="One of: label_os_180d | label_os_365d | label_os_any")
    ap.add_argument("--model_out", default="models/os_365d_ext.joblib")
    args = ap.parse_args()

    pats = pd.read_parquet(args.patients)
    plans = pd.read_parquet(args.plans)

    plan_aggs = agg_plans_to_patient(plans)
    X, y, num_cols, cat_cols, dropped = build_design_matrix(pats, plan_aggs, args.label)

    print(f"Label: {args.label}")
    print(f"Patients: {len(y)}  (positives={int((y==1).sum())}, negatives={int((y==0).sum())})")
    print(f"Dropped columns (hygiene): {json.dumps(dropped, indent=2)}")
    base_present = [c for c in PATIENT_BASE_VARS if c in X.columns]
    print(f"Base patient vars present: {base_present}")
    extra_feats = sorted([c for c in X.columns if c not in base_present and c != "patient_identifier"])
    preview = extra_feats[:20] + (["..."] if len(extra_feats) > 20 else [])
    print(f"Plan aggregate features ({len(extra_feats)}): {preview}")
    print(f"Using {len(num_cols)} numeric and {len(cat_cols)} categorical features.")

    pipe = build_pipeline(num_cols, cat_cols)

    # Stratified CV (patient-level rows are already 1 per patient)
    aucs, auprs, briers = [], [], []
    n_splits = max(2, min(5, int(y.value_counts().min())))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr, te in skf.split(np.zeros(len(y)), y):
        if y.iloc[tr].nunique() < 2:
            continue
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        try: aucs.append(roc_auc_score(y.iloc[te], p))
        except: aucs.append(np.nan)
        try: auprs.append(average_precision_score(y.iloc[te], p))
        except: auprs.append(np.nan)
        try: briers.append(brier_score_loss(y.iloc[te], p))
        except: briers.append(np.nan)

    report = {
        "label_used": args.label,
        "cv_auc_mean": float(np.nanmean(aucs)) if aucs else None,
        "cv_auc_std": float(np.nanstd(aucs)) if aucs else None,
        "cv_aupr_mean": float(np.nanmean(auprs)) if auprs else None,
        "cv_brier_mean": float(np.nanmean(briers)) if briers else None,
        "n_samples": int(len(y)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "features_num": num_cols[:60],
        "features_cat": cat_cols[:60],
        "dropped_columns": dropped,
        "total_features_after_drop": int(len(num_cols) + len(cat_cols)),
    }
    print(json.dumps(report, indent=2))

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump(pipe, args.model_out)
    with open(args.model_out.replace(".joblib", "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
