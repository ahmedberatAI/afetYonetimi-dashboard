# AfetYonetimi Dashboard (Streamlit)

Bu repo, need-classification prediction ciktilarini Streamlit uzerinden incelemek icin kullanilan dashboard'u icerir. Dashboard **canonical final v2** output'u ve metadata/provenance bilgisini arayuzde gosterir.

## Canonical Output

- Canonical final output:
  - `need_predictions_geolocated_v2_final.csv`
  - `need_predictions_geolocated_v2_final.meta.json`

## Default Source Resolution

Dashboard su sirayla predictions kaynagi arar:

1. `AFETYONETIMI_PREDICTIONS_CSV` / `AFETYONETIMI_PREDICTIONS_META` environment variable override
2. Dashboard repo icindeki canonical local copy:
   - `data/predictions/need_predictions_geolocated_v2_final.csv`
   - `data/predictions/need_predictions_geolocated_v2_final.meta.json`
3. Sibling modeling repo canonical output:
   - `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.csv`
   - `../afetYonetimi_colab/data/predictions/need_predictions_geolocated_v2_final.meta.json`

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

## Stand / Sunum Modu

Dashboard varsayilan olarak stand modunda acilir: teknik sidebar, yerel dosya
yollari ve Streamlit chrome'u gizlenir; harita en yogun saatten baslar ve
Tweet Test modeli onceden isitilir. Gelistirme/teknik kontrol icin eski
sidebar'li modu acmak istersen:

```powershell
$env:AFETYONETIMI_STAND_MODE=0
streamlit run dashboard/app.py
```

Tweet Test modelini acilista isitmak istemezsen:

```powershell
$env:AFETYONETIMI_PRELOAD_MODEL=0
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
Dashboard, modelleme reposunda OOF + validation ile dogrulanan `info_v1`
postprocess profilini de uygular: guclu bilgi-paylasimi dili varsa ve
`prob_bilgi_paylasimi >= 0.20` ise `bilgi_paylasimi` etiketi eklenir. Exact
raw-threshold davranisi icin `AFETYONETIMI_DISABLE_INFO_POSTPROCESS=1`
ortam degiskeni kullanilabilir.

Model checkpoint'i (~440 MB) repo icinde tutulmaz. Etiket + esik JSON'lari
ise kucuk oldugu icin repoya bundle edildi (`data/model_meta/`). Iki dagitim
modu desteklenir:

### Mod A - HuggingFace Hub (Streamlit Cloud icin onerilir)

1. Modeli HF Hub'a yukle (bir kerelik):

   ```powershell
   pip install huggingface-hub
   huggingface-cli login   # token gir (write yetkili)

   huggingface-cli repo create afet-need-classifier --type model
   huggingface-cli upload `
     ahmedberatAI/afet-need-classifier `
     C:\Users\omen\Desktop\afetYonetimi_colab\models\exp3_silver_then_gold_v3_exgold\final `
     . --repo-type model
   ```

   Repo herkese acik veya `--private` olabilir; private ise asagiya gore
   token tanimla.

2. Streamlit Cloud panelinde *Settings -> Secrets*'a ekle:

   ```toml
   AFETYONETIMI_MODEL_HF_REPO = "ahmedberatAI/afet-need-classifier"
   # Private repo icin:
   # AFETYONETIMI_HF_TOKEN = "hf_xxx"
   ```

   Lokal'de aynisi env var olarak da calisir:
   `set AFETYONETIMI_MODEL_HF_REPO=ahmedberatAI/afet-need-classifier`.

3. App restart et. Tweet Test sekmesindeki "Model kaynagi" alani otomatik
   `ahmedberatAI/afet-need-classifier` ile dolar; ilk tahminde checkpoint
   HF Hub'dan ephemeral diske inip cache'lenir.

### Mod B - Lokal disk

Model artefaktlari su sirayla aranir:

1. `AFETYONETIMI_MODEL_HF_REPO` (HF Hub repo id).
2. `AFETYONETIMI_MODEL_DIR` (lokal yol). Istege bagli `AFETYONETIMI_LABELS_JSON`,
   `AFETYONETIMI_THRESHOLDS_JSON` env var'larini ayri tutabilirsin (yoksa
   `data/model_meta/` icindeki bundle kullanilir).
3. Yan repo: `../afetYonetimi_colab/models/exp3_silver_then_gold_v3_exgold/final`.
4. Yan reponun `models/final/selection.json` cikti yolu (en yetkili).
5. Dashboard repo icine elle kopyalanmis bir kopya.

### Bagimliliklar

`requirements_dashboard.txt` icinde `torch` (CPU build) + `transformers` +
`huggingface-hub` yer aliyor; Streamlit Cloud kurulumunda otomatik kurulur.
Lokal'de ekstra islem gerekmez:

```powershell
pip install -r requirements.txt
streamlit run dashboard/app.py
```

> **Not:** Streamlit Community Cloud'da torch CPU wheel + 442 MB checkpoint
> + tokenizer toplamda ~750 MB disk + ~700 MB RAM kullanir; free tier
> uyarilari gelebilir. Memory yetmezse sekme acik bir hata ile kapanir,
> geri kalan 4 sekme etkilenmez.

## Streamlit Community Cloud

1. Streamlit Cloud panelinde "New app".
2. Repo: `ahmedberatAI/afetYonetimi-dashboard`
3. Main file path: `dashboard/app.py`
4. Canonical CSV/meta dosyalarini repo-local olarak sync et veya deployment ortaminda env var ile goster.

## Kullanilan Diger Dosyalar

- Gazetteer (lat/lon): `data/gazetteer/earthquake_region_neighborhoods.csv`
