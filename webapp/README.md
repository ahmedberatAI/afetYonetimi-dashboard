# Afet Yönetimi Local Web App

Bu klasör Streamlit arayüzüne paralel çalışan, sadece local kullanım için
tasarlanmış web uygulamasını içerir.

- Backend: FastAPI (`127.0.0.1:8787`)
- Frontend: Vite + React + TypeScript (`127.0.0.1:5173`)
- Veri kaynağı çözümü Streamlit dashboard ile aynı mantığı izler: env var,
  repo-local canonical CSV, ardından sibling `afetYonetimi_colab` canonical CSV.

## Kurulum

Repo kökünden:

```powershell
python -m pip install -r webapp/backend/requirements.txt
Set-Location webapp/frontend
npm install
Set-Location ../..
```

## Çalıştırma

Repo kökünden:

```powershell
.\webapp\start-local.ps1
```

Arayüz: http://127.0.0.1:5173

API dokümanı: http://127.0.0.1:8787/api/docs

## Manuel Çalıştırma

Terminal 1:

```powershell
python -m uvicorn webapp.backend.app.main:app --host 127.0.0.1 --port 8787 --reload
```

Terminal 2:

```powershell
Set-Location webapp/frontend
$env:VITE_API_BASE = "http://127.0.0.1:8787"
npm run dev
```

## Production Build

```powershell
Set-Location webapp/frontend
npm run build
Set-Location ../..
python -m uvicorn webapp.backend.app.main:app --host 127.0.0.1 --port 8787
```

Build üretildiyse FastAPI aynı process içinde statik frontend dosyalarını da
sunabilir. Geliştirme için Vite dev server önerilir.
