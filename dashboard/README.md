# Dashboard (Streamlit)

This dashboard now targets the **canonical final v2** prediction output and reads the companion metadata JSON when available.

## Expected canonical files

- `need_predictions_geolocated_v2_final.csv`
- `need_predictions_geolocated_v2_final.meta.json`

If the modeling repo sits next to this dashboard repo, the app auto-detects the canonical pair from:

- `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.csv`
- `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.meta.json`

The old `need_predictions_geolocated_63k.csv` file is still readable, but it is shown as a historical preview artifact rather than a canonical output.

## Run locally

```powershell
pip install -r requirements_dashboard.txt
streamlit run dashboard/app.py
```

## Optional repo-local sync

```powershell
python scripts/sync_canonical_prediction.py --overwrite
```

Use this when you want the canonical CSV/meta pair copied into `data/predictions/` for a standalone dashboard checkout or Streamlit Cloud deploy.

## Manual override

- Sidebar `Predictions CSV yolu`: point to any compatible CSV
- Sidebar metadata option: auto-detect `*.meta.json` next to the CSV, or set the JSON path manually
