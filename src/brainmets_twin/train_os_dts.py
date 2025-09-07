import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from joblib import dump

# ---------------- Config ----------------
INTERVAL_MONTHS = [3, 6, 9, 12, 15, 18, 21, 24]
MISSING_FRAC_DROP = 0.995
LOW_VAR_MIN_UNIQUE = 2

EXCLUDE_ALWAYS = {
    "patient_identifier", "days_to_death_from_gk", "total_fu_time",
    "label_os_180d", "label_os_365d", "label_os_any",
}

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r'[^0-9eE+\-\.]', '', regex=True),
        errors='coerce'
    )

def load_data(patients_path: str, plans_path: str) -> pd.DataFrame:
    P = pd.read_parquet(patients_path)
    PL = pd.read_parquet(plans_path)
    num_cols = [c for c in PL.columns
                if c not in ["patient_identifier", "gk_id"]
                and pd.api.types.is_numeric_dtype(PL[c])]
    aggs = {}
    for c in num_cols:
        if any(k in c for k in ["num_", "total_", "count"]):
            aggs[c] = ["sum", "max"]
        else:
            aggs[c] = ["mean", "max"]
    if aggs:
        G = PL.groupby("patient_identifier").agg(aggs)
        G.columns = [f"{a}_{b}" for a, b in G.columns]
        G = G.reset_index()
        P = P.merge(G, on="patient_identifier", how="left")
    return P

def make_person_period(P: pd.DataFrame,
                       label_days_col: str = "days_to_death_from_gk") -> pd.DataFrame:
    if label_days_col not in P.columns:
        raise SystemExit(f"patients.parquet is missing {label_days_col}")
    rows = []
    for _, row in P.iterrows():
        t_death = row[label_days_col]
        fu = row.get("total_fu_time", np.nan)
        for m in INTERVAL_MONTHS:
            end_days = int(m * 30.4375)
            censored = (pd.notna(fu) and (fu < end_days) and (pd.isna(t_death) or fu < t_death))
            if pd.notna(t_death) and t_death <= end_days:
                y = 1
            elif censored:
                y = np.nan
            else:
                y = 0
            r = row.to_dict()
            r["interval_months"] = m
            r["y"] = y
            rows.append(r)
    DF = pd.DataFrame(rows)
    DF = DF[DF["y"].notna()].copy()
    DF["y"] = DF["y"].astype(int)
    return DF

def drop_bad_cols(X: pd.DataFrame):
    dropped = {"all_missing": [], "high_missing": [], "low_variance_num": []}
    all_missing = [c for c in X.columns if X[c].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing); dropped["all_missing"] = all_missing
    high_missing = [c for c in X.columns if X[c].isna().mean() >= MISSING_FRAC_DROP]
    if high_missing:
        X = X.drop(columns=high_missing); dropped["high_missing"] = high_missing
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    low_var = [c for c in num_cols if X[c].dropna().nunique() < LOW_VAR_MIN_UNIQUE]
    if low_var:
        X = X.drop(columns=low_var); dropped["low_variance_num"] = low_var
    return X, dropped

def build_Xy(DF: pd.DataFrame):
    y = DF["y"].astype(int)
    cand = DF.select_dtypes(include=[np.number]).columns.tolist()
    cand = [c for c in cand if c != "y" and c not in EXCLUDE_ALWAYS]
    X = DF[cand].copy()
    X, dropped = drop_bad_cols(X)
    # NOTE: no imputation here; imputer is inside the Pipeline.
    cols_kept = list(X.columns)
    return X, y, dropped, cols_kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", default="data/patients.parquet")
    ap.add_argument("--plans", default="data/plans_enriched.parquet")
    ap.add_argument("--model_out", default="models/os_dts.joblib")
    args = ap.parse_args()

    P = load_data(args.patients, args.plans)
    DF = make_person_period(P)
    X, y, dropped, cols_kept = build_Xy(DF)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0,
            class_weight="balanced", random_state=42
        ))
    ])

    skf = StratifiedKFold(
        n_splits=min(5, int(y.value_counts().min())) if y.nunique()==2 else 5,
        shuffle=True, random_state=42
    )
    if skf.n_splits < 2:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        mdl = Pipeline(pipe.steps)
        mdl.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]

    report = {
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_brier": float(brier_score_loss(y, oof)),
        "interval_months": INTERVAL_MONTHS,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "dropped_columns": dropped
    }
    print(json.dumps(report, indent=2))

    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    dump({"model": pipe, "columns": cols_kept, "intervals": INTERVAL_MONTHS, "report": report},
         args.model_out)
    with open(args.model_out.replace(".joblib", "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved model to {args.model_out}")

if __name__ == "__main__":
    main()
