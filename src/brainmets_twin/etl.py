import os
import json
import numpy as np
import pandas as pd
import argparse
from typing import Optional, List, Dict


# ----------------------------
# Utilities
# ----------------------------

def read_csv_smart(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r'[^0-9eE+\-\.]', '', regex=True),
        errors='coerce'
    )

def get_ci(cols_lower: Dict[str, str], *cands) -> Optional[str]:
    for cand in cands:
        cl = cand.lower()
        if cl in cols_lower:
            return cols_lower[cl]
    return None


# ----------------------------
# Events from timeseries.csv
# ----------------------------

def build_events(ts: pd.DataFrame) -> pd.DataFrame:
    """Return tidy events dataframe with key columns."""
    if ts is None or ts.empty:
        return pd.DataFrame(columns=[
            "patient_identifier", "visit_class", "time_from_gk_days",
            "ctcae_cns_necrosis", "kps"
        ])

    df = norm_cols(ts)
    cols_lower = {c.lower(): c for c in df.columns}

    pid = get_ci(cols_lower, "patient_identifier", "patient_id", "mrn", "subject_id")
    vc  = get_ci(cols_lower, "visit_class")
    tfg = get_ci(cols_lower, "time_from_gk_days", "days_from_gk", "days_from_srs")
    nec = get_ci(cols_lower, "ctcae_cns_necrosis", "cns_necrosis_grade", "rn_grade")
    kps = get_ci(cols_lower, "kps", "kps_at_gks")

    mapping = {
        "patient_identifier": pid,
        "visit_class": vc,
        "time_from_gk_days": tfg,
        "ctcae_cns_necrosis": nec,
        "kps": kps
    }
    print("Column mapping:")
    for k, v in mapping.items():
        print(f"  {k:<30} -> {v}")

    keep_src = [v for v in mapping.values() if v is not None]
    out = df[keep_src].rename(columns={v: k for k, v in mapping.items() if v is not None})

    if "time_from_gk_days" in out.columns:
        out["time_from_gk_days"] = to_num(out["time_from_gk_days"])
    if "ctcae_cns_necrosis" in out.columns:
        out["ctcae_cns_necrosis"] = to_num(out["ctcae_cns_necrosis"])
    if "kps" in out.columns:
        out["kps"] = to_num(out["kps"])

    return out


# ----------------------------
# Patients from individual.csv (+ timeseries for kps_pre_gk)
# ----------------------------

def build_patients(individual: pd.DataFrame, ts_raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    if individual is None or individual.empty:
        return pd.DataFrame(columns=[
            "patient_identifier", "age", "primary_histology", "extracranial_disease",
            "days_to_death_from_gk", "total_fu_time",
            "label_os_180d", "label_os_365d", "label_os_any", "kps_pre_gk"
        ])

    df = norm_cols(individual)
    cols_lower = {c.lower(): c for c in df.columns}

    pid = get_ci(cols_lower, "patient_identifier", "patient_id", "mrn", "subject_id")
    age = get_ci(cols_lower, "age_diagnosis", "age")
    hist = get_ci(cols_lower, "primary_cancer", "primary_histology")
    ec   = get_ci(cols_lower, "extracranial_disease_status", "extracranial_disease")
    d2d  = get_ci(cols_lower, "days_to_death_from_gk", "days_to_death")
    fut  = get_ci(cols_lower, "total_fu_time", "follow_up_days")

    mapping = {
        "patient_identifier": pid,
        "age": age,
        "primary_histology": hist,
        "extracranial_disease": ec,
        "days_to_death_from_gk": d2d,
        "total_fu_time": fut,
    }
    print("Individual mapping:")
    print(json.dumps(mapping, indent=2))

    keep_src = [v for v in mapping.values() if v is not None]
    pat = df[keep_src].rename(columns={v: k for k, v in mapping.items() if v is not None}).copy()

    # Coerce numerics
    for c in ["age", "days_to_death_from_gk", "total_fu_time"]:
        if c in pat.columns:
            pat[c] = to_num(pat[c])

    # OS labels
    def make_os_label(h):
        died_in_h = (pat["days_to_death_from_gk"].notna()) & (pat["days_to_death_from_gk"] >= 0) & (pat["days_to_death_from_gk"] <= h)
        alive_past_h = (pat["total_fu_time"].notna()) & (pat["total_fu_time"] >= h)
        lab = np.where(died_in_h, 1.0, np.where(alive_past_h, 0.0, np.nan))
        return lab

    pat["label_os_180d"] = make_os_label(180)
    pat["label_os_365d"] = make_os_label(365)
    pat["label_os_any"] = np.where(
        pat["days_to_death_from_gk"].notna(), 1.0,
        np.where(pat["total_fu_time"].notna(), 0.0, np.nan)
    )

    # kps_pre_gk from timeseries (closest at/before GK; else closest overall)
    def kps_pre_from_timeseries(ts):
        if ts is None or ts.empty:
            return pd.Series(index=pat.index, dtype="float64")
        t = norm_cols(ts).copy()
        if "patient_identifier" not in t.columns:
            return pd.Series(index=pat.index, dtype="float64")
        # tolerant column detection
        cols_lower_t = {c.lower(): c for c in t.columns}
        tfg = get_ci(cols_lower_t, "time_from_gk_days", "days_from_gk", "days_from_srs")
        kps = get_ci(cols_lower_t, "kps", "kps_at_gks")
        if tfg is None or kps is None:
            return pd.Series(index=pat.index, dtype="float64")
        t["time_from_gk_days"] = to_num(t[tfg])
        t["kps"] = to_num(t[kps])

        pre_vals = {}
        for pid_val, grp in t.groupby("patient_identifier"):
            grp = grp.sort_values("time_from_gk_days")
            pre = grp.loc[grp["time_from_gk_days"] <= 0]
            if not pre.empty:
                pre_vals[pid_val] = pre.iloc[-1]["kps"]
            else:
                grp = grp.assign(absd=np.abs(grp["time_from_gk_days"]))
                pre_vals[pid_val] = grp.loc[grp["absd"].idxmin()]["kps"]

        m = pat[["patient_identifier"]].copy()
        m["kps_pre_gk"] = m["patient_identifier"].map(pre_vals)
        return m["kps_pre_gk"]

    pat["kps_pre_gk"] = kps_pre_from_timeseries(ts_raw)

    return pat


# ----------------------------
# Plans from gkdetails + merges
# ----------------------------

def build_plans(gk: pd.DataFrame, events: pd.DataFrame, ts_raw: Optional[pd.DataFrame], patients: pd.DataFrame) -> pd.DataFrame:
    if gk is None or gk.empty:
        return pd.DataFrame()

    df = norm_cols(gk)
    cols_lower = {c.lower(): c for c in df.columns}

    pid_col       = get_ci(cols_lower, "patient_identifier", "patient_id", "mrn", "subject_id")
    gk_id_col     = get_ci(cols_lower, "gk_id", "plan_id", "srs_id")
    margin_col    = get_ci(cols_lower, "margin_dose", "margin_dose_gy", "margindose_gy", "dose_margin")
    v12_col       = get_ci(cols_lower, "v12_cc", "v12", "v12cc")
    largest_vol   = get_ci(cols_lower, "largest_target_volume_cc", "target_volume_largest", "max_target_volume_cc")
    num_targets   = get_ci(cols_lower, "num_targets", "n_targets", "num_gk_targets")
    piv_col       = get_ci(cols_lower, "piv_cc", "piv", "piv_volume_cc")
    beam_on_col   = get_ci(cols_lower, "beam_on_time_min", "beam_on_time", "beamontime", "beam_on_minutes")

    mapping = {
        "patient_identifier": pid_col,
        "gk_id": gk_id_col,
        "margin_dose": margin_col,
        "v12_cc": v12_col,
        "largest_target_volume_cc": largest_vol,
        "num_targets": num_targets,
        "piv_cc": piv_col,
        "beam_on_time_min": beam_on_col,
    }
    print("GK/plan mapping:")
    print(json.dumps(mapping, indent=2))

    keep_src = [v for v in mapping.values() if v is not None]
    out = df[keep_src].rename(columns={v: k for k, v in mapping.items() if v is not None}).copy()

    for c in ["margin_dose","v12_cc","largest_target_volume_cc","num_targets","piv_cc","beam_on_time_min"]:
        if c in out.columns:
            out[c] = to_num(out[c])

    # ---- Merge plan-level features that actually live in timeseries.csv ----
    if ts_raw is not None and not ts_raw.empty:
        t = norm_cols(ts_raw).copy()
        # normalize to lower_underscore for matching
        t.columns = [c.strip().lower().replace(' ', '_') for c in t.columns]
        ts_alias = {
            # regional / counts
            'num_gk_targets'                : 'num_gk_targets',
            'total_num_tumors_gk'           : 'total_num_tumors_gk',
            'num_lobar_gk'                  : 'num_lobar_gk',
            'num_brainstem_gk'              : 'num_brainstem_gk',
            'num_thal_bg_gk'                : 'num_thal_bg_gk',
            'num_cerebellum_gk'             : 'num_cerebellum_gk',
            'num_skull_gk'                  : 'num_skull_gk',
            # largest-target geom/dose
            'xdimension_largest'            : 'xdimension_largest',
            'ydimension_largest'            : 'ydimension_largest',
            'zdimension_largest'            : 'zdimension_largest',
            'max_linear_dimension_largest'  : 'max_linear_dimension_largest',
            'target_volume_largest'         : 'largest_target_volume_cc',
            'planned_isodose_volume_largest': 'piv_cc_largest',
            'volume_12gy_largest'           : 'v12_cc_largest',
            'num_isocenters_largest'        : 'num_isocenters_largest',
            'margin_dose_largest'           : 'margin_dose_largest',
            'percent_isodose_largest'       : 'percent_isodose_largest',
            'pci_largest'                   : 'pci_largest',
            'gri_largest'                   : 'gri_largest',
            'max_dose'                      : 'max_dose_largest',
            'min_dose'                      : 'min_dose_largest',
            'mean_dose'                     : 'mean_dose_largest',
        }
        # choose join keys
        join_on = None
        t_has = set(t.columns)
        if 'gk_id' in out.columns and 'gk_id' in t_has:
            join_on = ['gk_id']
        elif 'patient_identifier' in out.columns and 'patient_identifier' in t_has:
            if 'gk_count' in out.columns and 'gk_count' in t_has:
                join_on = ['patient_identifier','gk_count']
            else:
                # pick ts row closest to index (time_from_gk_days ~ 0) per patient
                if 'time_from_gk_days' in t_has:
                    t = t.sort_values('time_from_gk_days').copy()
                    t = t.loc[t.groupby('patient_identifier')['time_from_gk_days'].apply(lambda s: s.abs().idxmin())]
                    join_on = ['patient_identifier']

        ts_keep_raw = [raw for raw in ts_alias.keys() if raw in t.columns]
        if join_on and ts_keep_raw:
            ts_sub = t[join_on + ts_keep_raw].rename(columns=ts_alias).copy()
            for c in ts_sub.columns:
                if c not in join_on:
                    ts_sub[c] = to_num(ts_sub[c])
            out = out.merge(ts_sub, on=join_on, how='left')
            print(f"Merged timeseries features on {join_on}: {list(ts_alias[r] for r in ts_keep_raw)}")
        else:
            print("Timeseries merge skipped (no join keys or no matching columns).")

    # ---- RN labels from events ----
    rn_180 = pd.Series(index=out.index, dtype='float64')
    rn_365 = pd.Series(index=out.index, dtype='float64')
    rn_any = pd.Series(index=out.index, dtype='float64')

    if events is not None and not events.empty and \
       'patient_identifier' in events.columns and 'ctcae_cns_necrosis' in events.columns:
        ev = events.copy()
        has_days = 'time_from_gk_days' in ev.columns and ev['time_from_gk_days'].notna().any()
        grp = ev.groupby('patient_identifier')

        for i, row in out.iterrows():
            pid_val = row['patient_identifier'] if 'patient_identifier' in out.columns else None
            if pid_val is None or pid_val not in grp.groups:
                rn_180.iloc[i] = np.nan
                rn_365.iloc[i] = np.nan
                rn_any.iloc[i] = np.nan
                continue

            sub = grp.get_group(pid_val)
            sub = sub.dropna(subset=['ctcae_cns_necrosis'])
            if sub.empty:
                rn_180.iloc[i] = np.nan
                rn_365.iloc[i] = np.nan
                rn_any.iloc[i] = np.nan
                continue

            nec = pd.to_numeric(sub['ctcae_cns_necrosis'], errors='coerce')
            if has_days:
                days = pd.to_numeric(sub['time_from_gk_days'], errors='coerce')
                in180 = (days >= 0) & (days <= 180)
                in365 = (days >= 0) & (days <= 365)
                rn_180.iloc[i] = float(1.0) if (nec[in180] >= 2).any() else (0.0 if in180.any() else np.nan)
                rn_365.iloc[i] = float(1.0) if (nec[in365] >= 2).any() else (0.0 if in365.any() else np.nan)
                rn_any.iloc[i] = float(1.0) if (nec >= 2).any() else (0.0 if nec.notna().any() else np.nan)
            else:
                # No timing info: use any-time flag
                val_any = float(1.0) if (nec >= 2).any() else (0.0 if nec.notna().any() else np.nan)
                rn_180.iloc[i] = val_any
                rn_365.iloc[i] = val_any
                rn_any.iloc[i] = val_any

    out['label_rn_180d'] = rn_180
    out['label_rn_365d'] = rn_365
    out['label_rn_any']  = rn_any

    # ---- Merge OS labels + kps_pre_gk from patients onto plans (convenience)
    if patients is not None and not patients.empty:
        os_cols = [c for c in ['label_os_180d','label_os_365d','label_os_any','kps_pre_gk'] if c in patients.columns]
        if 'patient_identifier' in out.columns and os_cols:
            out = out.merge(patients[['patient_identifier'] + os_cols], on='patient_identifier', how='left')

    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--imaging", required=False)
    ap.add_argument("--individual", required=True)
    ap.add_argument("--gkdetails", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ts = read_csv_smart(args.timeseries)
    im = read_csv_smart(args.imaging) if args.imaging else pd.DataFrame()
    ind = read_csv_smart(args.individual)
    gk = read_csv_smart(args.gkdetails)

    # Build artifacts
    events = build_events(ts)
    patients = build_patients(ind, ts_raw=ts)
    plans = build_plans(gk, events, ts_raw=ts, patients=patients)

    # Save
    events.to_parquet(os.path.join(args.outdir, "events.parquet"))
    patients.to_parquet(os.path.join(args.outdir, "patients.parquet"))
    plans.to_parquet(os.path.join(args.outdir, "plans_enriched.parquet"))

    print("Saved:")
    print(f" - {os.path.join(args.outdir, 'events.parquet')}")
    print(f" - {os.path.join(args.outdir, 'patients.parquet')}")
    print(f" - {os.path.join(args.outdir, 'plans_enriched.parquet')}")

if __name__ == "__main__":
    main()
