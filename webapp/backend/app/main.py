from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data_service import (
    Filters,
    build_hotspots,
    build_options,
    build_overview,
    clear_caches,
    get_dataset,
    parse_csv_param,
    parse_date,
    repo_root,
)


app = FastAPI(
    title="Afet Yönetimi Local Web API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CsvParam = Annotated[str | None, Query(description="Comma-separated values")]


def _filters(
    start_date: str | None,
    end_date: str | None,
    provinces: str | None,
    districts: str | None,
    labels: str | None,
    label_mode: str,
    urgency_min: float | None,
    search: str | None,
) -> Filters:
    return Filters(
        start_date=parse_date(start_date),
        end_date=parse_date(end_date),
        provinces=parse_csv_param(provinces),
        districts=parse_csv_param(districts),
        labels=parse_csv_param(labels),
        label_mode=(label_mode or "ANY").upper(),
        urgency_min=urgency_min,
        search=search,
    )


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        bundle = get_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "rows": len(bundle.df), "source": str(bundle.source.csv_path)}


@app.post("/api/refresh")
def refresh() -> dict[str, object]:
    clear_caches()
    bundle = get_dataset()
    return {"ok": True, "rows": len(bundle.df), "source": str(bundle.source.csv_path)}


@app.get("/api/options")
def options() -> dict[str, object]:
    try:
        return build_options(get_dataset())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/overview")
def overview(
    start_date: str | None = None,
    end_date: str | None = None,
    provinces: CsvParam = None,
    districts: CsvParam = None,
    labels: CsvParam = None,
    label_mode: str = "ANY",
    urgency_min: float | None = None,
    search: str | None = None,
) -> dict[str, object]:
    try:
        filters = _filters(start_date, end_date, provinces, districts, labels, label_mode, urgency_min, search)
        return build_overview(get_dataset(), filters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/hotspots")
def hotspots(
    start_date: str | None = None,
    end_date: str | None = None,
    provinces: CsvParam = None,
    districts: CsvParam = None,
    labels: CsvParam = None,
    label_mode: str = "ANY",
    urgency_min: float | None = None,
    search: str | None = None,
    hour: str | None = None,
    level: str = "province",
    signal_mode: str = "count_any_need",
) -> dict[str, object]:
    try:
        filters = _filters(start_date, end_date, provinces, districts, labels, label_mode, urgency_min, search)
        return build_hotspots(get_dataset(), filters, hour, level, signal_mode)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


_FRONTEND_DIST = repo_root() / "webapp" / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    assets = _FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        requested = (_FRONTEND_DIST / full_path).resolve()
        if requested.is_file() and _FRONTEND_DIST in requested.parents:
            return FileResponse(requested)
        return FileResponse(_FRONTEND_DIST / "index.html")
