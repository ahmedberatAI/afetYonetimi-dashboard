# AfetYonetimi Dashboard (Streamlit)

Bu repo, deprem tweet verisinden uretilmis model tahminlerini (pseudo/silver) harita uzerinde kesfetmek icin Streamlit dashboard icerir.

## Lokal Calistirma

```powershell
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Streamlit Community Cloud

1. Streamlit Cloud panelinde "New app".
2. Repo: `ahmedberatAI/afetYonetimi-dashboard`
3. Main file path: `dashboard/app.py`
4. Deploy.

## Kullanilan Dosyalar

- Predictions CSV (default): `data/predictions/need_predictions_geolocated_63k.csv`
- Gazetteer (lat/lon): `data/gazetteer/earthquake_region_neighborhoods.csv`

Sidebar'dan CSV yolunu degistirebilirsin.
