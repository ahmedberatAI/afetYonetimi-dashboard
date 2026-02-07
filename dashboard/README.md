# Dashboard (Streamlit)

This is a lightweight prototype dashboard for exploring model predictions
without requiring a database.

## 1) Create Predictions CSV (Colab Recommended)

Example (Drive):

```bash
python scripts/predict_need_classifier.py \
  --model-dir models/need_classification_silver_63k/final \
  --input data/processed/emergency_geolocated_96k.csv \
  --thresholds-json models/need_classification_silver_63k/thresholds.json \
  --output data/predictions/need_predictions_geolocated_63k.csv \
  --dedup-by-id
```

## 2) Run Dashboard (Local)

Install:

```powershell
pip install -r requirements_dashboard.txt
```

Run:

```powershell
streamlit run dashboard/app.py
```

Then set "Predictions CSV yolu" in the sidebar if you saved it elsewhere.

