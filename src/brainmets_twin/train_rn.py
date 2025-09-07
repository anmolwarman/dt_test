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

# -----------------------
# Config
# -----------------------

EXCLUDE_COLS = {
    "patient_identifier",  # used for grouping only
    "gk_id",
    # labels
    "label_rn_180d", "label_rn_365d", "label_rn_any",
    # known post-index / unsafe
    "kps_last", "kps_min", "kps_max", "total_fu_time", "rn_any_ever",
}

BAD_SUBSTRINGS = [
    "label_", "to_death", "days_to_", "outcome", "followup", "fu_", "_fu_",
    "censor", "rn_any", "mrn", "id_", "_id", "identifier",
]

MAX_CAT_CARD = 50

# -----------------------
# Helpers
# -----------------------

def is_leak_col(col: str) -> bool:
    cl = col.lower()
    return any(s in cl for s in BAD_SUBSTRINGS)

def drop_all_nan_cols(X: pd.DataFrame) -> pd.DataFrame:
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
    return X

def select_features(X: pd.DataFrame):
    drop_cols = {c for c in X.columns if c in EXCLUDE_COLS or is_leak_col(c)}
    Xf = X.drop(columns=list(drop_cols & set(X.columns)), errors="ignore")
    Xf = drop_all_nan_cols(Xf)

    num_cols = [c for c in Xf.select_dtypes(include=[np.number]).columns]
    cand_cat = [c for c in Xf.columns if c not in num_cols]

    cat_cols = []
    for c in cand_cat:
        if c in EXCLUDE_COLS or is_leak_col(c):
            continue
        uniq = Xf[c].dropna().unique()
        if len(uniq) <= MAX_CAT_CARD:
            cat_cols.append(c)
    return Xf, num_cols, cat_cols

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])

def patient_level_counts(df: pd.DataFrame, label: str):
    """Return number of positive/negative patients for a label."""
    if "patient_identifier" not in df.columns or label not in df.columns:
        return 0, 0
    tmp = df[["patient_identifier", label]].dropna()
    g = tmp.groupby("patient_identifier")[label].max().astype(int)
    pos = int((g == 1).sum())
    neg = int((g == 0).sum())
    return pos, neg

def pick_best_label(df: pd.DataFrame):
    """
    Prefer a label with ≥2 positive patients to allow leak-safe CV.
    Try 180d, then 365d, then 'any'. If none have ≥2 positive patients,
    return the first available label with ≥2 classes and warn.
    """
    candidates = ["label_rn_180d", "label_rn_365d", "label_rn_any"]
    details = {}
    for c in candidates:
        if c in df.columns and df[c].dropna().nunique() >= 2:
            pos, neg = patient_level_counts(df, c)
            details[c] = {"pos_patients": pos, "neg_patients": neg}
    if not details:
        raise SystemExit("No RN label with ≥2 classes found (checked 180d, 365d, any).")

    # choose best with ≥2 positive patients
    for c in candidates:
        if c in details and details[c]["pos_patients"] >= 2:
            return c, details[c], False  # leak_warning=False

    # fallback: pick first available (but warn we will do row-level CV)
    first = next(iter(details.keys()))
    return first, details[first], True  # leak_warning=True

# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to plans or plans_enriched parquet")
    ap.add_argument("--model_out", default="models/rn_180d.joblib")
    ap.add_argument("--label", default=None, help="Force a specific label (label_rn_180d|label_rn_365d|label_rn_any)")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile)

    # choose label
    if args.label:
        if args.label not in df.columns or df[args.label].dropna().nunique() < 2:
            raise SystemExit(f"{args.label} not available or lacks ≥2 classes.")
        label_col = args.label
        pos_p, neg_p = patient_level_counts(df, label_col)
        leak_warning = pos_p < 2  # if <2 positive patients, we can't do group CV
        details = {"pos_patients": pos_p, "neg_patients": neg_p}
    else:
        label_col, details, leak_warning = pick_best_label(df)

    print(f"Using RN label column: {label_col}")
    print(f"Patient-level counts: {details}")

    # target/feature frame
    y = df[label_col].dropna().astype(int)
    X = df.loc[y.index].copy()

    # overall class check
    if y.nunique() < 2:
        raise SystemExit("Training data has only one class overall; cannot fit a classifier.")

    # features
    Xf, num_cols, cat_cols = select_features(X)
    print(f"Using {len(num_cols)} numeric and {len(cat_cols)} categorical features.")

    pipe = build_pipeline(num_cols, cat_cols)

    # -----------------------
    # CV strategy
    # -----------------------
    aucs, auprs, briers = [], [], []
    cv_kind = "group_stratified"  # or "row_stratified_fallback"

    if "patient_identifier" in X.columns and not leak_warning:
        # We have ≥2 positive patients → do group-stratified CV at patient level
        # Build patient-level splits by stratifying on patient max label
        tmp = pd.DataFrame({"patient": X["patient_identifier"], "y": y})
        y_group = tmp.groupby("patient")["y"].max().astype(int)
        patients = y_group.index.to_numpy()
        ygp = y_group.to_numpy()

        pos_g = int((ygp == 1).sum())
        neg_g = int((ygp == 0).sum())
        n_splits = min(5, pos_g, neg_g, len(ygp))
        if n_splits < 2:
            # Shouldn't happen due to leak_warning guard, but double-check
            leak_warning = True
            cv_kind = "row_stratified_fallback"
        else:
            # map rows by patient
            rows_by_patient = {}
            for i, pat in enumerate(X["patient_identifier"]):
                rows_by_patient.setdefault(pat, []).append(i)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            for tr_pat_idx, te_pat_idx in skf.split(patients, ygp):
                tr_pats = set(patients[tr_pat_idx])
                te_pats = set(patients[te_pat_idx])
                tr_rows = np.array([i for p in tr_pats for i in rows_by_patient[p]], dtype=int)
                te_rows = np.array([i for p in te_pats for i in rows_by_patient[p]], dtype=int)

                y_tr, y_te = y.iloc[tr_rows], y.iloc[te_rows]
                if y_tr.nunique() < 2:
                    # extremely rare edge; skip this fold
                    continue
                pipe.fit(Xf.iloc[tr_rows], y_tr)
                p = pipe.predict_proba(Xf.iloc[te_rows])[:, 1]
                try: aucs.append(roc_auc_score(y_te, p))
                except ValueError: aucs.append(np.nan)
                try: auprs.append(average_precision_score(y_te, p))
                except ValueError: auprs.append(np.nan)
                try: briers.append(brier_score_loss(y_te, p))
                except ValueError: briers.append(np.nan)

    if leak_warning:
        # Row-level stratified CV (NOT leak-safe, but provides a sanity estimate)
        cv_kind = "row_stratified_fallback"
        skf = StratifiedKFold(n_splits=min(5, y.value_counts().min(), 5), shuffle=True, random_state=42)
        for tr, te in skf.split(np.zeros(len(y)), y):
            if y.iloc[tr].nunique() < 2:
                continue
            pipe.fit(Xf.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(Xf.iloc[te])[:, 1]
            try: aucs.append(roc_auc_score(y.iloc[te], p))
            except ValueError: aucs.append(np.nan)
            try: auprs.append(average_precision_score(y.iloc[te], p))
            except ValueError: auprs.append(np.nan)
            try: briers.append(brier_score_loss(y.iloc[te], p))
            except ValueError: briers.append(np.nan)

    # -----------------------
    # Report + fit final
    # -----------------------
    report = {
        "label_used": label_col,
        "cv_kind": cv_kind,
        "patient_pos": int(details.get("pos_patients", 0)),
        "patient_neg": int(details.get("neg_patients", 0)),
        "cv_auc_mean": float(np.nanmean(aucs)) if len(aucs) else None,
        "cv_auc_std": float(np.nanstd(aucs)) if len(aucs) else None,
        "cv_aupr_mean": float(np.nanmean(auprs)) if len(auprs) else None,
        "cv_brier_mean": float(np.nanmean(briers)) if len(briers) else None,
        "n_samples": int(len(y)),
        "n_features_num": len(num_cols),
        "n_features_cat": len(cat_cols),
        "features_num": num_cols[:60],
        "features_cat": cat_cols[:60],
        "leak_warning": bool(leak_warning),
        "notes": "Row-level CV used for metrics due to <2 positive patients; numbers may be optimistic."
                 if leak_warning else "Group-stratified patient-level CV.",
    }
    print(json.dumps(report, indent=2))

    # Fit final model on ALL data (classes verified ≥2 overall)
    pipe.fit(Xf, y)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump(pipe, args.model_out)
    with open(args.model_out.replace(".joblib", "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
