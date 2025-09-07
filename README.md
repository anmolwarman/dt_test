# Brain Mets Digital Twin — MVP (No Radiomics)

This MVP builds two deployable predictors from the NYU Brain Mets CSVs:

1. **RN risk classifier** after SRS/GK (binary at 180 days; falls back to "any time" if timing columns not found).
2. **Overall survival** (OS) classifier at 180 and 365 days from first GK.

It includes:
- ETL to normalize the four CSVs and derive features/labels.
- Training scripts (scikit-learn) that save models to `models/`.
- A FastAPI service exposing JSON endpoints for predictions.

> No radiomics are used. Imaging files are not required for this MVP.

## 0) Setup

```bash
# Create and activate a virtual environment (choose one)
python3 -m venv .venv
source .venv/bin/activate

# Or with conda:
# conda create -y -n twin python=3.11
# conda activate twin

pip install -r requirements.txt
```

## 1) Put your CSVs somewhere and point to them

You can either set paths via a `.env` file at repo root:

```
TIMESERIES_CSV=/path/to/20220825_timeseries(in).csv
IMAGING_CSV=/path/to/20220825_imaging(in).csv
INDIVIDUAL_CSV=/path/to/20220804_individual(in).csv
GKDETAILS_CSV=/path/to/20220804_gkdetails(in).csv
```

Or pass them via CLI flags (see below).

## 2) Run ETL

```bash
python -m src.brainmets_twin.etl   --timeseries "$TIMESERIES_CSV"   --imaging "$IMAGING_CSV"   --individual "$INDIVIDUAL_CSV"   --gkdetails "$GKDETAILS_CSV"   --outdir data
```

This will produce:
- `data/patients.parquet`  (1 row per patient)
- `data/plans.parquet`     (1 row per GK plan, with features + RN labels)
- `data/events.parquet`    (long table of events/visits; optional for MVP)

The ETL prints a **mapping report** so you can see which columns were auto-detected.

## 3) Train models

```bash
# RN risk classifier (180 days)
python -m src.brainmets_twin.train_rn --infile data/plans.parquet --model_out models/rn_180d.joblib

# OS classifier (180d and 365d); outputs two models
python -m src.brainmets_twin.train_os --infile data/patients.parquet --outdir models
```

Each script prints metrics and saves a `*_report.json` and `*.joblib`.

## 4) Run the API

```bash
uvicorn api.app:app --reload --port 8000
```

### Endpoints

- `POST /predict/rn` — RN risk after SRS at 180 days (expects GK plan features)
- `POST /predict/os180` — 180-day mortality
- `POST /predict/os365` — 365-day mortality
- `GET /schema` — JSON describing the expected feature names for each endpoint

See `api/example_client.py` for a working example.

## Notes & Assumptions

- The scripts **auto-detect** column names via fuzzy matching. They will list what they found and which fallbacks were used.
- RN label prefers a 0–180 day window if a "days-from-GK" column is found in the time-series. If not, it uses "ever RN".
- OS labels require a numeric `days_to_death_from_gk` column; rows missing this are dropped for OS training.
- This is a minimal MVP intended to prove end-to-end deployment and allow iteration on features.
