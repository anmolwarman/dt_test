import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from joblib import dump
from xgboost import XGBClassifier

# ---------- config ----------
PATIENT_BASE_VARS = ["age", "primary_histology", "extracranial_disease", "kps_pre_gk"]
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

# Columns we should never use as features (labels, identifiers, follow-up, etc.)
EXCLUDE = {
    "patient_identifier",
    "label_os_180d", "label_os_365d", "label_os_any",
    "days_to_death_from_gk", "total_fu_time",
    "kps_last", "kps_min", "kps_max", "rn_any_ever",
}

# Drop rules
MISSING_FRACTION_DROP = 0.995  # drop columns with >=99.5% missing
MIN_NUMERIC_UNIQUE    = 2      # drop numeric columns with <2 unique non-null values


# ---------- helpers ----------
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

    burden_cols = [
        "num_targets", "num_gk_targets", "total_num_tumors_gk",
        "num_lobar_gk", "num_brainstem_gk", "num_thal_bg_gk",
        "num_cerebellum_gk", "num_skull_gk",
        "total_vol_all_targets"
    ]
    burden_cols = [c for c in burden_cols if c in present]
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

    # ultra-sparse
    high_missing = [c for c in X.columns if X[c].isna().mean() >= MISSING_FRACTION_DROP]
    if high_missing:
        X = X.drop(columns=high_missing)
        dropped["high_missing"] = high_missing

    # numeric low-variance
    low_var = [c for c in X.select_dtypes(include=[np.number]).columns
               if X[c].dropna().nunique() < MIN_NUMERIC_UNIQUE]
    if low_var:
        X = X.drop(columns=low_var)
        dropped["low_variance_num"] = low_var

    return X, dropped

def build_design(patients: pd.DataFrame, plans: pd.DataFrame, label: str):
    """
    Merge base patient vars + aggregated plan features; return numeric X (DataFrame),
    y (Series), column order, and dropped-col info.
    """
    if label not in patients.columns or patients[label].dropna().nunique() < 2:
        raise SystemExit(f"Label {label} missing or has <2 classes.")

    plan_aggs = agg_plans_to_patient(plans)

    keep_base = [c for c in PATIENT_BASE_VARS if c in patients.columns]
    P = patients[["patient_identifier", label] + keep_base].merge(
        plan_aggs, on="patient_identifier", how="left"
    )

    y = P[label].dropna().astype(int)
    X = P.loc[y.index].drop(columns=[label])

    # exclude leaky/ids
    X = X.drop(columns=[c for c in X.columns if c in EXCLUDE], errors="ignore")

    # numeric-only matrix for XGB; hygiene BEFORE imputation
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    X_num = X[num_cols].copy()
    X_num, dropped = drop_bad_cols(X_num)

    # impute AFTER dropping, then rebuild DataFrame to keep col order
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X_num)
    X_num = pd.DataFrame(X_arr, columns=X_num.columns, index=X_num.index)

    cols = list(X_num.columns)
    return X_num, y, cols, dropped

def build_xgb(monotone_constraints: str) -> XGBClassifier:
    # Sensible defaults
    return XGBClassifier(
        n_estimators=2000,          # upper bound; we’ll pick n with CV
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        objective="binary:logistic",
        # set eval metric in params for older APIs
        eval_metric="auc",
        tree_method="hist",  # fast on CPU
        random_state=42,
        monotone_constraints=monotone_constraints
    )
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="data/patients.parquet")
    ap.add_argument("--plans",    default="data/plans_enriched.parquet")
    ap.add_argument("--label",    default="label_os_365d")
    ap.add_argument("--model_out", default="models/os_365d_xgb.joblib")
    args = ap.parse_args()

    pats  = pd.read_parquet(args.patients)
    plans = pd.read_parquet(args.plans)

    # Build design matrix
    X, y, cols, dropped = build_design(pats, plans, args.label)

    # Monotone constraints vector: +1 for age, -1 for kps_pre_gk, 0 otherwise
    mono = []
    for c in cols:
        if c == "age": mono.append(1)
        elif c == "kps_pre_gk": mono.append(-1)
        else: mono.append(0)
    monotone_constraints = "(" + ",".join(str(v) for v in mono) + ")"

    # CV without early stopping: try a few n_estimators and pick best by AUC
    n_splits = min(5, int(y.value_counts().min()))
    n_splits = max(n_splits, 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    candidate_n = [200, 400, 800, 1200, 1600]
    aucs, auprs, briers, best_rounds = [], [], [], []

    for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        best_auc = -np.inf
        best_n = candidate_n[0]
        best_model = None

        for n in candidate_n:
            mdl = build_xgb(monotone_constraints)
            mdl.set_params(n_estimators=n)
            mdl.fit(X_tr, y_tr, verbose=False)

            p_val = mdl.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, p_val)
            except Exception:
                auc = np.nan

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_n = n
                best_model = mdl

        best_rounds.append(int(best_n))
        p = best_model.predict_proba(X_te)[:, 1]

        try: aucs.append(roc_auc_score(y_te, p))
        except: aucs.append(np.nan)
        try: auprs.append(average_precision_score(y_te, p))
        except: auprs.append(np.nan)
        try: briers.append(brier_score_loss(y_te, p))
        except: briers.append(np.nan)

    report = {
        "label_used": args.label,
        "cv_auc_mean": float(np.nanmean(aucs)),
        "cv_auc_std": float(np.nanstd(aucs)),
        "cv_aupr_mean": float(np.nanmean(auprs)),
        "cv_brier_mean": float(np.nanmean(briers)),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": cols[:60],
        "dropped_columns": dropped,
        "monotone_constraints": {"age": "+", "kps_pre_gk": "-", "others": 0},
        "best_rounds_list": [int(b) for b in best_rounds],
        "best_rounds_median": int(np.nanmedian(best_rounds)) if len(best_rounds) else None,
        "cv_folds": int(n_splits),
        "candidate_n_estimators": candidate_n,
    }
    print(json.dumps(report, indent=2))

    # Final fit on full data using median of best rounds
    final_n_estimators = max(200, int(np.nanmedian(best_rounds)))
    final_model = build_xgb(monotone_constraints)
    final_model.set_params(n_estimators=final_n_estimators)
    final_model.fit(X, y, verbose=False)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump({"model": final_model, "columns": cols, "report": report}, args.model_out)
    with open(args.model_out.replace(".joblib","_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
