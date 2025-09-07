# BrainMets Digital Twin MVP

This repository contains a **Digital Twin MVP** for patients undergoing Gamma Knife radiosurgery (GKRS) for brain metastases.  
It provides a full pipeline: ETL (extract–transform–load), survival model training (OS), radionecrosis prediction (RN), advanced models (XGBoost, DTS), API serving, and interactive visualization.

---

## Table of Contents
1. Project Overview
2. Requirements
3. Environment Setup
4. Data Sources
5. ETL: Converting CSV → Parquet
6. Training RN Models
7. Training OS Models (Logistic, Extended, XGB, DTS)
8. Evaluating Models
9. Handling Class Imbalance
10. Feature Engineering (Comorbidities, Radiomics, Systemic Therapy)
11. Prediction and Plotting Curves
12. Serving API with FastAPI/Uvicorn
13. UI Options (HTML vs React/Vite)
14. Deployment on Mac vs Cloud
15. Troubleshooting Common Errors
16. Adding New Features Later
17. Maintainer Notes

---

## 1. Project Overview
The MVP builds **digital twins** for patients by predicting:
- **Overall Survival (OS):** probability of survival over time (e.g., 12 months).
- **Radionecrosis (RN):** probability of treatment‑related necrosis.

Outputs include **charts of survival vs cumulative hazard**, like a Kaplan–Meier curve, but individualized.

---

## 2. Requirements
- Python 3.9+
- Virtual environment (`venv`)
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `fastparquet`, `xgboost`, `joblib`, `matplotlib`, `fastapi`, `uvicorn`, `chart.js` (for HTML UI).
- (Optional) Node.js + npm for React frontend.

---

## 3. Environment Setup
```bash
cd /Users/anmolwarman/Desktop/brainmets_twin_mvp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `pyarrow` fails:  
```bash
pip install --only-binary=:all: pyarrow==16.1.0
```

---

## 4. Data Sources
You provided 4 CSVs from `nyumets.org/docs/brainapi/`:

- `timeseries.csv` → longitudinal visits + radiomics
- `imaging.csv` → imaging metadata
- `individual.csv` → baseline patient info (age, comorbidity, systemic therapy)
- `gkdetails.csv` → per-GKRS plan details

---

## 5. ETL: Converting CSV → Parquet
Run ETL to normalize columns and merge into Parquet:

```bash
python -m src.brainmets_twin.etl   --timeseries ~/Downloads/20220825_timeseries(in).csv   --imaging   ~/Downloads/20220825_imaging(in).csv   --individual ~/Downloads/20220804_individual(in).csv   --gkdetails ~/Downloads/20220804_gkdetails(in).csv   --outdir data
```

Generates:
- `data/events.parquet`
- `data/plans.parquet`
- `data/patients.parquet`

Extended ETL (`etl_wide.py`) merges **extra radiomics + comorbidities**.

---

## 6. Training RN Models
Radionecrosis labels:
- `label_rn_180d`, `label_rn_365d`, `label_rn_any`

Run baseline RN:
```bash
python -m src.brainmets_twin.train_rn   --infile data/plans_enriched.parquet   --model_out models/rn_180d.joblib
```

⚠ RN suffers from **low positives** (few patients). Better use `label_rn_any`.

---

## 7. Training OS Models
### Logistic Regression (baseline)
```bash
python -m src.brainmets_twin.train_os   --infile data/patients.parquet   --label label_os_365d   --model_out models/os_365d.joblib
```

### Extended Logistic Regression (with aggregated plan features)
```bash
python -m src.brainmets_twin.train_os_ext   --infile data/patients.parquet   --plans data/plans_enriched.parquet   --label label_os_365d   --model_out models/os_365d_ext.joblib
```

### XGBoost Model
```bash
python -m src.brainmets_twin.train_os_xgb   --infile data/patients.parquet   --plans data/plans_enriched.parquet   --label label_os_365d   --model_out models/os_365d_xgb.joblib
```

### DTS Model (per‑interval survival curve)
```bash
python -m src.brainmets_twin.train_os_dts   --patients data/patients.parquet   --plans data/plans_enriched.parquet   --label label_os_365d   --model_out models/os_dts.joblib
```

---

## 8. Evaluating Models
Metrics printed after training:
- AUC (ROC)
- AUPR
- Brier score
- # features used
- CV folds

Check JSON logs in stdout.

---

## 9. Handling Class Imbalance
- RN labels are very rare → use `label_rn_any` or survival instead.
- Possible fixes:
  - Oversample positives (SMOTE)
  - Undersample negatives
  - Use focal loss or class weights

---

## 10. Feature Engineering
Features include:
- **Patient-level:** age, histology, comorbidity, systemic therapy
- **Plan-level:** margin dose, beam‑on time, conformity indices, tumor counts
- **Aggregations:** mean, max, sum across multiple GKRS

We drop:
- Columns with 100% missing
- Columns with no variance

Categorical variables (histology, extracranial disease) → encoded numerically (0/1 or one-hot).

---

## 11. Prediction and Plotting Curves
Predict a single patient:
```bash
python -m src.brainmets_twin.predict_plot_os_dts   --model models/os_dts.joblib   --patients data/patients.parquet   --plans data/plans_enriched.parquet   --patient_id 10092334   --out_png charts/os_curve_10092334.png   --out_json charts/os_curve_10092334.json
```

Outputs:
- `.png` chart
- `.json` probabilities per interval

---

## 12. Serving API with FastAPI/Uvicorn
Create `api_os_dts.py`:
```bash
uvicorn api_os_dts:app --reload --port 8080
```

Endpoints:
- `/predict/{patient_id}` → returns survival curve JSON
- `/health` → health check

---

## 13. UI Options
### A. Simple HTML
Located in `charts/os_view.html`. Uses Chart.js to fetch API and plot curves.

### B. React/Vite UI
1. Install Node:
   ```bash
   brew install node
   ```
2. Create project:
   ```bash
   npm create vite@latest digital-twin-ui -- --template react
   cd digital-twin-ui
   npm install recharts
   ```
3. Create `src/OsTwinChart.jsx` with chart code.
4. Run:
   ```bash
   npm run dev
   ```

---

## 14. Deployment on Mac vs Cloud
- Mac is fine for MVP (small dataset).
- For scaling (more patients, GPU radiomics), use:
  - AWS EC2 (p3 or g4 instances)
  - GCP or Azure equivalents

---

## 15. Troubleshooting
- **`pyarrow` missing:** `pip install pyarrow --only-binary=:all:`
- **ROC AUC NaN:** only one positive label → switch to OS or RN_any
- **Index out-of-bounds:** patient_id not found
- **NaN in predict:** ensure imputers handle missing values

---

## 16. Adding New Features Later
- Comorbidity data (from `individual.csv`) → extend patient parquet
- Systemic therapy (chemo, immuno) → binary encodings
- Radiomics features → already partially integrated, add more in ETL

---

## 17. Maintainer Notes
- Code lives in `src/brainmets_twin/`
- Data stored in `data/`
- Models stored in `models/`
- Charts (PNG + JSON) in `charts/`
- When opening a new thread, paste patient IDs + JSON outputs for fast debugging
- Always regenerate Parquet after changing ETL
- Keep this README up to date with new features

---

**Maintainer:**  
@Anmol Warman  
Digital Twin for BrainMets MVP  
