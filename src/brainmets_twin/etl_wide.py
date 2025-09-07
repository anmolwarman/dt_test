import argparse, os, json, numpy as np, pandas as pd

# ---- helpers ---------------------------------------------------------------
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r'[^0-9eE+\-\.]','', regex=True),
                         errors='coerce')

def read_csv_or_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

# Timeseries features we want from YOUR file (do NOT include gk_id/patient_identifier here)
TS_KEEP = [
    # region counts & totals
    "total_vol_all_targets","num_gk_targets","total_num_tumors_gk","num_lobar_gk",
    "num_brainstem_gk","num_thal_bg_gk","num_cerebellum_gk","num_skull_gk",
    # largest-target geometry/dose
    "xdimension_largest","ydimension_largest","zdimension_largest",
    "max_linear_dimension_largest","target_volume_largest",
    "planned_isodose_volume_largest","coverage_largest","pci_largest","gri_largest",
    "margin_dose_largest","percent_isodose_largest","volume_12gy_largest",
    "num_isocenters_largest",
    # some sheets have these without the "_largest" suffix
    "max_dose","min_dose","mean_dose",
    # planning/runtime
    "beam_on_time",
    # clinical at GK (keep if present; we’ll also bring KPS from individual)
    "kps_at_gks"
]

# GK details you have; we’ll normalize names
GK_KEEP_RAW = [
    "patient_identifier","gk_id","days_after_initial_gk","gk_count","target_count",
    "total_targets","largest_target_in_plan","target_working_label","target_volume",
    "planned_isodose_volume","gtvpiv","pci","gri","margin_dose","percent_isodose",
    "volume_12gy","max_dose","beam_on_time","total_vol_all_targets"
]

# Individual base fields to carry through (others will be auto-detected later if you wish)
INDIV_BASE = [
    "patient_identifier","age_diagnosis","primary_cancer","extracranial_disease_status",
    "kps_at_gks","total_fu_time","days_to_death_from_gk"
]

# ---- ETL pieces ------------------------------------------------------------
def clean_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate-named columns, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]

def normalize_gkdetails(df: pd.DataFrame) -> pd.DataFrame:
    """Rename GK details to normalized model names."""
    keep = [c for c in GK_KEEP_RAW if c in df.columns]
    df = df[keep].copy()
    ren = {
        "planned_isodose_volume": "piv_cc",
        "gtvpiv": "piv_cc",                        # if both exist, we’ll dedup after
        "volume_12gy": "v12_cc",
        "beam_on_time": "beam_on_time_min",
        "target_count": "num_targets",
        "max_dose": "max_dose_plan"               # distinguish plan-level max dose
    }
    for src, dst in ren.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})
    # Deduplicate if both piv_cc columns slipped in
    df = clean_duplicate_columns(df)
    # Numeric coercion (except ids)
    for c in df.columns:
        if c not in ("patient_identifier","gk_id","target_working_label"):
            df[c] = to_num(df[c])
    return df

def normalize_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """Select & rename timeseries features; ensure no duplicate gk_id/patient_identifier columns."""
    cols = ["gk_id"]
    if "patient_identifier" in ts.columns:
        cols.append("patient_identifier")
    # add kept features that actually exist
    cols += [c for c in TS_KEEP if c in ts.columns]
    ts = ts[cols].copy()

    # If largest-target dose fields are missing but generic exist, map them
    ren = {}
    if "max_dose_largest" not in ts.columns and "max_dose" in ts.columns:
        ren["max_dose"] = "max_dose_largest"
    if "min_dose_largest" not in ts.columns and "min_dose" in ts.columns:
        ren["min_dose"] = "min_dose_largest"
    if "mean_dose_largest" not in ts.columns and "mean_dose" in ts.columns:
        ren["mean_dose"] = "mean_dose_largest"
    if "planned_isodose_volume_largest" in ts.columns:
        ren["planned_isodose_volume_largest"] = "piv_cc_largest"
    if "volume_12gy_largest" in ts.columns:
        ren["volume_12gy_largest"] = "v12_cc_largest"

    if ren:
        ts = ts.rename(columns=ren)

    # numeric coercion (except ids)
    for c in ts.columns:
        if c not in ("patient_identifier","gk_id"):
            ts[c] = to_num(ts[c])

    # ensure no duplicate names
    ts = clean_duplicate_columns(ts)
    return ts

def build_plans_enriched(ts: pd.DataFrame, gk: pd.DataFrame) -> pd.DataFrame:
    gk = normalize_gkdetails(gk)
    ts = normalize_timeseries(ts)
    # safe merge (unique 'gk_id' on both sides)
    out = gk.merge(ts, on="gk_id", how="left", suffixes=("", "_ts"))
    # if both gk and ts have patient_identifier, prefer gk's; drop duplicate
    if "patient_identifier_ts" in out.columns:
        out = out.drop(columns=["patient_identifier_ts"])
    return out

def build_patients_wide(indiv: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in INDIV_BASE if c in indiv.columns]
    P = indiv[keep].copy()
    # rename to modeling names
    ren = {
        "age_diagnosis": "age",
        "primary_cancer": "primary_histology",
        "extracranial_disease_status": "extracranial_disease",
        "kps_at_gks": "kps_pre_gk",
    }
    P = P.rename(columns={k:v for k,v in ren.items() if k in P.columns})

    # create labels if possible
    if "days_to_death_from_gk" in P.columns:
        d = P["days_to_death_from_gk"]
        P["label_os_180d"] = (d.notna() & (d <= 180)).astype(int)
        P["label_os_365d"] = (d.notna() & (d <= 365)).astype(int)
        P["label_os_any"]  = (d.notna() & (d > 0)).astype(int)

    # numeric coercion (except id and histology)
    for c in P.columns:
        if c in ("patient_identifier","primary_histology"):
            continue
        P[c] = to_num(P[c])
    # one row per patient
    P = P.sort_values("patient_identifier").drop_duplicates("patient_identifier", keep="first")
    return P

# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--gkdetails", required=True)
    ap.add_argument("--individual", required=True)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ts  = read_csv_or_parquet(args.timeseries)
    gk  = read_csv_or_parquet(args.gkdetails)
    ind = read_csv_or_parquet(args.individual)

    plans    = build_plans_enriched(ts, gk)
    patients = build_patients_wide(ind)

    plans.to_parquet(os.path.join(args.outdir, "plans_enriched.parquet"))
    patients.to_parquet(os.path.join(args.outdir, "patients.parquet"))

    report = {
        "n_plans": int(len(plans)),
        "n_patients": int(len(patients)),
        "plans_cols": list(plans.columns)[:120],
        "patients_cols": list(patients.columns)[:120]
    }
    with open(os.path.join(args.outdir, "etl_wide_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()