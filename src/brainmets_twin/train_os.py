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

SAFE_EXCLUDE = {
    # labels & identifiers
    "patient_identifier",
    "label_os_180d","label_os_365d","label_os_any",
    # post-index or leak-prone
    "total_fu_time","days_to_death_from_gk",
    "kps_last","kps_min","kps_max","rn_any_ever",
}

LEAKY_SUBSTR = ["label_","follow","fu","_fu_","to_death","days_to_","outcome","mrn","_id","id_","identifier"]

def is_bad(c): 
    cl=c.lower()
    return any(s in cl for s in LEAKY_SUBSTR)

def select_features(df):
    X = df.drop(columns=[c for c in df.columns if c in SAFE_EXCLUDE or is_bad(c)], errors="ignore")
    # Numeric / categorical split
    num = list(X.select_dtypes(include=[np.number]).columns)
    cat = [c for c in X.columns if c not in num and X[c].dropna().nunique() <= 50]
    # Keep a small, interpretable set if present
    prefer = [c for c in ["age","primary_histology","extracranial_disease","kps_pre_gk"] if c in X.columns]
    # Put preferred first (ordering for readability)
    num = [c for c in prefer if c in num] + [c for c in num if c not in prefer]
    cat = [c for c in prefer if c in cat] + [c for c in cat if c not in prefer]
    return X, num, cat

def build_pipeline(num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("pre", pre), ("clf", clf)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/patients.parquet")
    ap.add_argument("--label", default="label_os_365d")
    ap.add_argument("--model_out", default="models/os_365d.joblib")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile)
    if args.label not in df.columns or df[args.label].dropna().nunique()<2:
        raise SystemExit(f"Label {args.label} missing or lacks ≥2 classes.")
    y = df[args.label].dropna().astype(int)
    Xfull = df.loc[y.index].copy()

    X, num, cat = select_features(Xfull)

    pipe = build_pipeline(num, cat)

    # Stratified CV
    aucs, auprs, briers = [], [], []
    skf = StratifiedKFold(n_splits=min(5, y.value_counts().min(), 5), shuffle=True, random_state=42)
    for tr, te in skf.split(np.zeros(len(y)), y):
        if y.iloc[tr].nunique()<2:
            continue
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:,1]
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
        "n_features_num": len(num),
        "n_features_cat": len(cat),
        "features_num": num[:60],
        "features_cat": cat[:60],
    }
    print(json.dumps(report, indent=2))

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump(pipe, args.model_out)
    with open(args.model_out.replace(".joblib","_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
