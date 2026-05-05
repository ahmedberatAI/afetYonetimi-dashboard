# AfetYonetimi Dashboard (Streamlit)

Bu repo, need-classification prediction ciktilarini Streamlit uzerinden incelemek icin kullanilan dashboard'u icerir. Dashboard artik historical `63k silver` preview yerine **canonical final v2** output'u tercih eder ve metadata/provenance bilgisini arayuzde gosterir.

## Canonical vs Historical

- Canonical final output:
  - `need_predictions_geolocated_v2_final.csv`
  - `need_predictions_geolocated_v2_final.meta.json`
- Historical preview artifact:
  - `need_predictions_geolocated_63k.csv`

Historical `63k` dosyasi repo icinde tutulsa da canonical olarak gosterilmez. Dashboard canonical source bulunamazsa bunu acik bir fallback olarak etiketler.

## Default Source Resolution

Dashboard su sirayla predictions kaynagi arar:

1. `AFETYONETIMI_PREDICTIONS_CSV` / `AFETYONETIMI_PREDICTIONS_META` environment variable override
2. Dashboard repo icindeki canonical local copy:
   - `data/predictions/need_predictions_geolocated_v2_final.csv`
   - `data/predictions/need_predictions_geolocated_v2_final.meta.json`
3. Sibling modeling repo canonical output:
   - `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.csv`
   - `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.meta.json`
4. Historical fallback:
   - `data/predictions/need_predictions_geolocated_63k.csv`

## Metadata Support

Dashboard metadata varsa su alanlari goruntuler:

- selected experiment key
- selected model dir
- threshold source
- threshold type
- row count
- duplicate removal summary
- pred positives per label
- generated_at

Metadata yoksa app CSV header'indan schema kesfeder ve CSV-only fallback modunda calisir.

## Lokal Calistirma

```powershell
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Canonical Local Example

Iki repo yan yana ise ek bir kopya gerekmiyor. Dashboard otomatik olarak su dosyalari kullanir:

- `C:\Users\omen\Desktop\afetYonetimi_colab\data\predictions\need_predictions_geolocated_v2_final.csv`
- `C:\Users\omen\Desktop\afetYonetimi_colab\data\predictions\need_predictions_geolocated_v2_final.meta.json`

## Optional Sync For Repo-Local Copy

Streamlit Cloud veya tek-repo kurulum icin canonical CSV/meta dosyalarini dashboard repo icine kopyalamak istersen:

```powershell
python scripts/sync_canonical_prediction.py --overwrite
```

Bu script `data/predictions/need_predictions_geolocated_v2_final.*` dosyalarini dashboard repo'ya kopyalar. Bu dosyalar `.gitignore` icinde tutulur.

## Manual / Custom Source

- Sidebar'daki `Predictions CSV yolu` alaniyla herhangi bir CSV secilebilir.
- `CSV yanindaki metadata dosyasini otomatik ara` aciksa app `*.meta.json` dosyasini otomatik dener.
- Istersen metadata yolunu manual olarak da verebilirsin.

## Tweet Test Sekmesi (Canli Inference)

`Tweet Test` sekmesi, kanonik leak-free model
(`exp3_silver_then_gold_v3_exgold`) ile elle yazilan bir cumlenin 9 ihtiyac
etiketi icin olasilik ve CV-tuned esik tahminlerini canli sekilde gosterir.

Model artefaktlari (~440 MB) repo icinde tutulmaz; su sirayla aranir:

1. `AFETYONETIMI_MODEL_DIR` (ve istege bagli `AFETYONETIMI_LABELS_JSON`,
   `AFETYONETIMI_THRESHOLDS_JSON`) ortam degiskenleri.
2. Yan repo: `../afetYonetimi_colab/models/exp3_silver_then_gold_v3_exgold/final`
   ile `label_columns.json` / `thresholds_cv.json`.
3. Yan reponun `models/final/selection.json` cikti yolu (en yetkili).
4. Dashboard repo icine elle kopyalanmis bir kopya.

Ek bagimliliklar (yalnizca bu sekme icin gerekli, varsayilan kurulumda yok):

```powershell
pip install torch transformers
```

Streamlit Community Cloud'da bu sekmeyi calistirmak icin yeterli bellek
yoktur; yerel calistirma icin tasarlanmistir. Streamlit Cloud kurulumunda
sekme acilir ama `model_dir` bulunamadigi icin acik bir hata mesaji ile
kapatilir, geri kalan sekmeler etkilenmez.

## Streamlit Community Cloud

1. Streamlit Cloud panelinde "New app".
2. Repo: `ahmedberatAI/afetYonetimi-dashboard`
3. Main file path: `dashboard/app.py`
4. Canonical CSV/meta dosyalarini repo-local olarak sync et veya deployment ortaminda env var ile goster.

## Kullanilan Diger Dosyalar

- Gazetteer (lat/lon): `data/gazetteer/earthquake_region_neighborhoods.csv`
