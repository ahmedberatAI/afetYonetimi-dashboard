# Predictions Folder

This folder may contain a repo-local copy of the canonical final output:

- `need_predictions_geolocated_v2_final.csv`
- `need_predictions_geolocated_v2_final.meta.json`
  Optional local copy of the canonical final output. These files are gitignored by default.

If the modeling repo lives next to this dashboard repo, the app can read the canonical files directly from:

- `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.csv`
- `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.meta.json`

If you want a repo-local copy instead, run:

```powershell
python scripts/sync_canonical_prediction.py --overwrite
```
