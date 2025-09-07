import argparse, json, numpy as np, pandas as pd, os, sys
from joblib import load
import matplotlib.pyplot as plt

EXCLUDE = {
    "patient_identifier","days_to_death_from_gk","total_fu_time",
    "label_os_180d","label_os_365d","label_os_any"
}

def abspath(p):
    return os.path.abspath(p) if p else None

def load_feature_row(p_path, pl_path, patient_id):
    P = pd.read_parquet(p_path)
    PLN = pd.read_parquet(pl_path)

    pid_str = str(patient_id)
    mask = P["patient_identifier"].astype(str) == pid_str
    if not mask.any():
        sample = P["patient_identifier"].astype(str).head(20).to_list()
        raise ValueError(f"[predict] patient_identifier '{pid_str}' not found. Example IDs: {sample}")

    row = P.loc[mask].iloc[0].copy()

    pl_mask = PLN["patient_identifier"].astype(str) == pid_str
    PLp = PLN.loc[pl_mask].copy()
    if len(PLp):
        num_cols = [c for c in PLp.columns
                    if c not in ["patient_identifier","gk_id"]
                    and pd.api.types.is_numeric_dtype(PLp[c])]
        aggs = {}
        for c in num_cols:
            if any(k in c for k in ["num_","total_","count"]):
                aggs[c] = ["sum","max"]
            else:
                aggs[c] = ["mean","max"]
        g = PLp.groupby("patient_identifier").agg(aggs)
        g.columns = [f"{a}_{b}" for a,b in g.columns]
        g = g.reset_index(drop=True)
        for c in g.columns:
            row[c] = float(g[c].iloc[0])

    return row.to_frame().T  # single-row DataFrame

def predict_curve(model_path, patients_path, plans_path, patient_id, out_png=None, out_json=None):
    print(f"[predict] CWD={os.getcwd()}")
    print(f"[predict] model={abspath(model_path)}")
    print(f"[predict] patients={abspath(patients_path)}")
    print(f"[predict] plans={abspath(plans_path)}")
    print(f"[predict] patient_id={patient_id}")

    bundle = load(model_path)  # {"model","columns","intervals"}
    model, cols, intervals = bundle["model"], bundle["columns"], bundle["intervals"]
    print(f"[predict] loaded model with {len(cols)} features and intervals={intervals}")

    R = load_feature_row(patients_path, plans_path, patient_id)

    # build person-period records
    DF = pd.concat([R.assign(interval_months=m) for m in intervals], ignore_index=True)

    # numeric selection and column align
    X = DF.select_dtypes(include=[np.number]).drop(
        columns=[c for c in DF.columns if c in EXCLUDE], errors="ignore"
    )
    # add any missing expected columns as NaN
    missing = [c for c in cols if c not in X.columns]
    if missing:
        print(f"[predict] adding {len(missing)} missing columns as NaN")
        for c in missing:
            X[c] = np.nan
    # ensure exact column order
    X = X[cols]

    # predict hazards and convert to survival
    h = model.predict_proba(X)[:,1]
    S = np.cumprod(1.0 - h)
    risk = 1.0 - S

    curve = {
        "patient_id": str(patient_id),
        "months": intervals,
        "hazard": h.tolist(),
        "survival": S.tolist(),
        "cum_mortality": risk.tolist()
    }

    # Write JSON (absolute path)
    if out_json:
        out_json_abs = abspath(out_json)
        os.makedirs(os.path.dirname(out_json_abs), exist_ok=True)
        with open(out_json_abs, "w") as f:
            json.dump(curve, f, indent=2)
        print(f"[predict] Saved JSON -> {out_json_abs}")

    # Plot PNG (absolute path)
    if out_png:
        out_png_abs = abspath(out_png)
        os.makedirs(os.path.dirname(out_png_abs), exist_ok=True)
        plt.figure(figsize=(6,4))
        plt.plot(intervals, risk, marker="o", label="Predicted cumulative mortality")
        plt.plot(intervals, S, marker="o", linestyle="--", label="Predicted survival")
        plt.xlabel("Months since GK")
        plt.ylabel("Probability")
        plt.ylim(0,1)
        plt.grid(alpha=.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png_abs, dpi=160)
        plt.close()
        print(f"[predict] Saved PNG  -> {out_png_abs}")

    return curve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/os_dts.joblib")
    ap.add_argument("--patients", default="data/patients.parquet")
    ap.add_argument("--plans", default="data/plans_enriched.parquet")
    ap.add_argument("--patient_id", required=True)
    ap.add_argument("--out_png", default="charts/os_curve.png")
    ap.add_argument("--out_json", default="charts/os_curve.json")
    args = ap.parse_args()
    try:
        predict_curve(args.model, args.patients, args.plans, args.patient_id, args.out_png, args.out_json)
    except Exception as e:
        print(f"[predict] ERROR: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
