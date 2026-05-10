import base64
import datetime as dt
import html
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# Streamlit Cloud / `streamlit run dashboard/app.py` ile çalıştırıldığında
# `dashboard/` klasörü sys.path'e eklenir ama repo kökü eklenmez. Böylece
# `from dashboard.utils import ...` ImportError verir. Repo kökünü manuel ekle.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import qrcode

    QRCODE_AVAILABLE = True
except Exception:
    QRCODE_AVAILABLE = False

try:
    from dashboard.utils import (
        PredictionSchema,
        PROVINCE_CENTROID,
        build_prediction_schema,
        canonical_limitations,
        classify_prediction_source,
        discover_default_source,
        format_generated_at,
        format_path,
        infer_meta_path,
        load_prediction_metadata,
        load_predictions_csv,
        maybe_fix_mojibake,
        normalize_location_value,
        pretty_label,
        source_kind_label,
        source_kind_note,
    )
except ModuleNotFoundError:
    from utils import (  # type: ignore[no-redef]
        PredictionSchema,
        PROVINCE_CENTROID,
        build_prediction_schema,
        canonical_limitations,
        classify_prediction_source,
        discover_default_source,
        format_generated_at,
        format_path,
        infer_meta_path,
        load_prediction_metadata,
        load_predictions_csv,
        maybe_fix_mojibake,
        normalize_location_value,
        pretty_label,
        source_kind_label,
        source_kind_note,
    )

try:
    from dashboard.inference import (
        ENV_HF_REPO,
        ModelBundle,
        ModelLocation,
        PredictionResult,
        describe_candidates,
        discover_model_location,
        load_bundle,
        looks_like_hf_repo_id,
        make_user_location,
        predict_one,
    )
except ModuleNotFoundError:
    from inference import (  # type: ignore[no-redef]
        ENV_HF_REPO,
        ModelBundle,
        ModelLocation,
        PredictionResult,
        describe_candidates,
        discover_model_location,
        load_bundle,
        looks_like_hf_repo_id,
        make_user_location,
        predict_one,
    )


REPO_DASHBOARD = "https://github.com/ahmedberatAI/afetYonetimi-dashboard"
REPO_MAIN = "https://github.com/ahmedberatAI/afet-aciliyet-sinyalleri"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PALETTE = ["#dc2626", "#f97316", "#fbbf24", "#22d3ee", "#60a5fa", "#a78bfa", "#f472b6", "#34d399", "#facc15"]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().casefold() not in {"0", "false", "no", "off", ""}


STAND_MODE = _env_flag("AFETYONETIMI_STAND_MODE", True)
PRELOAD_MODEL = _env_flag("AFETYONETIMI_PRELOAD_MODEL", STAND_MODE)
SHOW_TECHNICAL_DETAILS = _env_flag("AFETYONETIMI_SHOW_TECHNICAL_DETAILS", not STAND_MODE)

DEMO_TICKER_ITEMS: list[tuple[str, str, str]] = [
    ("arama_kurtarma", "Anonim acil çağrı: enkaz altında kalan aile için arama-kurtarma desteği gerekiyor.", "Hatay"),
    ("gida_su", "Saha notu: toplanma alanında su, bebek maması ve temel gıda ihtiyacı yükseliyor.", "Kahramanmaraş"),
    ("barinma", "Anonim bildirim: gece için çadır, battaniye ve ısınma desteği talep ediliyor.", "Adıyaman"),
    ("saglik", "Saha notu: kronik ilaç ve ilk yardım malzemesi ihtiyacı işaretlendi.", "Malatya"),
    ("lojistik", "Koordinasyon notu: yardım araçları için açık rota ve dağıtım noktası bilgisi gerekiyor.", "Gaziantep"),
    ("altyapi", "Saha notu: elektrik, iletişim ve yol erişimi kesintileri izleme listesine alındı.", "Hatay"),
]


st.set_page_config(
    page_title="AfetYönetimi | İhtiyaç Sinyalleri",
    layout="wide",
    initial_sidebar_state="collapsed" if STAND_MODE else "expanded",
    page_icon="🛟",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"], .stApp, .block-container {
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.75rem;
            max-width: 1500px;
        }

        .stApp {
            background:
                radial-gradient(ellipse at top left, rgba(220, 38, 38, 0.10), transparent 55%),
                radial-gradient(ellipse at bottom right, rgba(37, 99, 235, 0.12), transparent 55%),
                #0b1220;
        }

        /* HERO */
        .hero {
            position: relative;
            border-radius: 20px;
            padding: 1.6rem 1.8rem 1.5rem 1.8rem;
            margin-bottom: 1.1rem;
            background: linear-gradient(125deg, rgba(127, 29, 29, 0.85) 0%, rgba(15, 23, 42, 0.92) 55%, rgba(30, 64, 175, 0.55) 100%);
            border: 1px solid rgba(220, 38, 38, 0.35);
            box-shadow: 0 30px 70px rgba(0, 0, 0, 0.45);
            overflow: hidden;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 90% 10%, rgba(248, 113, 113, 0.18), transparent 45%);
            pointer-events: none;
        }
        .hero-grid {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: wrap;
            gap: 1.4rem;
            position: relative;
        }
        .hero-left { max-width: 760px; }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 700;
            color: #fecaca;
            background: rgba(220, 38, 38, 0.15);
            border: 1px solid rgba(220, 38, 38, 0.5);
            padding: 0.32rem 0.8rem;
            border-radius: 999px;
        }
        .hero-eyebrow .pulse {
            width: 8px; height: 8px; border-radius: 50%;
            background: #ef4444;
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            animation: pulse 1.6s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75); }
            70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .hero-title {
            color: #f8fafc;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0.7rem 0 0.45rem 0;
            letter-spacing: -0.01em;
        }
        .hero-sub {
            color: #cbd5e1;
            font-size: 1.02rem;
            line-height: 1.5;
            max-width: 700px;
        }
        .hero-meta {
            margin-top: 0.85rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .hero-chip {
            font-size: 0.78rem;
            color: #e2e8f0;
            background: rgba(148, 163, 184, 0.14);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 0.28rem 0.75rem;
        }
        .hero-right {
            display: flex;
            gap: 0.85rem;
            flex-wrap: wrap;
        }

        /* BIG STAT CARDS */
        .stat-card {
            min-width: 168px;
            border-radius: 16px;
            padding: 0.95rem 1.05rem;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.7) 0%, rgba(15, 23, 42, 0.45) 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(8px);
        }
        .stat-card .label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
        }
        .stat-card .value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin-top: 0.18rem;
            line-height: 1.0;
            letter-spacing: -0.02em;
        }
        .stat-card .delta {
            margin-top: 0.35rem;
            font-size: 0.82rem;
            color: #fbbf24;
        }
        .stat-card.accent-red    { border-color: rgba(220, 38, 38, 0.55); }
        .stat-card.accent-orange { border-color: rgba(249, 115, 22, 0.55); }
        .stat-card.accent-blue   { border-color: rgba(59, 130, 246, 0.55); }
        .stat-card.accent-green  { border-color: rgba(34, 197, 94, 0.55); }
        .stat-card.accent-purple { border-color: rgba(168, 85, 247, 0.55); }

        /* SEVERITY LEGEND */
        .severity-legend {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.4rem 0 0.6rem 0;
        }
        .severity-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.78rem;
            color: #e2e8f0;
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
        }
        .severity-pill .dot {
            width: 10px; height: 10px; border-radius: 50%;
            box-shadow: 0 0 8px currentColor;
        }
        .sev-critical { color: #7f0000; }
        .sev-high     { color: #cb181d; }
        .sev-medium   { color: #ef3b2c; }
        .sev-watch    { color: #fd8d3c; }

        /* TICKER */
        .ticker-wrap {
            position: relative;
            overflow: hidden;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(15, 23, 42, 0.6) 100%);
            margin-bottom: 1.0rem;
        }
        .ticker-wrap::before, .ticker-wrap::after {
            content: "";
            position: absolute;
            top: 0; bottom: 0;
            width: 80px;
            z-index: 2;
            pointer-events: none;
        }
        .ticker-wrap::before { left: 0; background: linear-gradient(90deg, #0b1220, transparent); }
        .ticker-wrap::after  { right: 0; background: linear-gradient(270deg, #0b1220, transparent); }
        .ticker-track {
            display: inline-flex;
            gap: 1.6rem;
            padding: 0.6rem 1rem;
            white-space: nowrap;
            animation: tickerScroll 90s linear infinite;
        }
        .ticker-item {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 0.88rem;
            color: #e2e8f0;
        }
        .ticker-item .badge {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            color: #0b1220;
            background: #fbbf24;
        }
        .ticker-item .badge.b-arama_kurtarma { background: #dc2626; color: #fff; }
        .ticker-item .badge.b-saglik         { background: #ec4899; color: #fff; }
        .ticker-item .badge.b-barinma        { background: #f97316; color: #0b1220; }
        .ticker-item .badge.b-gida_su        { background: #22d3ee; color: #0b1220; }
        .ticker-item .badge.b-altyapi        { background: #a78bfa; color: #0b1220; }
        .ticker-item .badge.b-guvenlik       { background: #facc15; color: #0b1220; }
        .ticker-item .badge.b-lojistik       { background: #34d399; color: #0b1220; }
        .ticker-item .badge.b-psikolojik     { background: #60a5fa; color: #0b1220; }
        .ticker-item .badge.b-bilgi_paylasimi{ background: #94a3b8; color: #0b1220; }
        @keyframes tickerScroll {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }

        /* SOURCE BANNER (kept, restyled) */
        .source-banner {
            border-radius: 14px;
            padding: 0.85rem 1.1rem;
            margin: 0.2rem 0 1.0rem 0;
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: rgba(15, 23, 42, 0.6);
        }
        .source-banner .eyebrow {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
        }
        .source-banner .title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-top: 0.15rem;
            color: #f8fafc;
        }
        .source-banner .body {
            margin-top: 0.3rem;
            font-size: 0.88rem;
            color: #cbd5e1;
            line-height: 1.45;
        }
        .source-banner .meta {
            margin-top: 0.4rem;
            font-size: 0.78rem;
            color: #94a3b8;
        }
        .source-banner.canonical  { border-left: 4px solid #10b981; }
        .source-banner.candidate  { border-left: 4px solid #3b82f6; }
        .source-banner.historical { border-left: 4px solid #f97316; }
        .source-banner.custom     { border-left: 4px solid #94a3b8; }

        /* TAB STYLE */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(15, 23, 42, 0.55);
            padding: 0.4rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 0.5rem 1.1rem;
            color: #cbd5e1;
            font-weight: 600;
            font-size: 0.92rem;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%);
            color: #fff !important;
            box-shadow: 0 6px 18px rgba(220, 38, 38, 0.35);
        }

        /* QR CARDS */
        .qr-card {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.7rem;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(15, 23, 42, 0.55);
            margin-bottom: 0.55rem;
        }
        .qr-card img { border-radius: 8px; background: #fff; padding: 4px; }
        .qr-card .qr-label { font-size: 0.78rem; font-weight: 600; color: #f8fafc; }
        .qr-card .qr-link  { font-size: 0.7rem; color: #94a3b8; word-break: break-all; }

        /* MISC */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0f1d 0%, #0b1220 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        h2, h3, h4 { color: #f1f5f9 !important; }
        .signal-hero {
            background: linear-gradient(120deg, rgba(127, 29, 29, 0.55) 0%, rgba(30, 64, 175, 0.45) 100%);
            border: 1px solid rgba(220, 38, 38, 0.35);
            border-radius: 14px;
            padding: 14px 18px;
            margin: 0.35rem 0 0.85rem 0;
        }
        .signal-hero-title { color: #f8fafc; font-size: 1.05rem; font-weight: 700; }
        .signal-hero-sub   { color: #fecaca; font-size: 0.91rem; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if STAND_MODE:
        st.markdown(
            """
            <style>
            #MainMenu, footer, header [data-testid="stToolbar"],
            [data-testid="stDeployButton"], [data-testid="stDecoration"],
            [data-testid="stStatusWidget"], [data-testid="collapsedControl"] {
                display: none !important;
                visibility: hidden !important;
            }
            section[data-testid="stSidebar"] {
                display: none !important;
            }
            .block-container {
                max-width: 1680px;
                padding-top: 0.8rem;
                padding-left: 2.2rem;
                padding-right: 2.2rem;
            }
            .hero {
                padding: 1.9rem 2.1rem 1.8rem 2.1rem;
                margin-bottom: 1.2rem;
            }
            .hero-left { max-width: 880px; }
            .hero-title { font-size: 2.45rem; }
            .hero-sub { font-size: 1.12rem; }
            .stat-card {
                min-width: 184px;
                padding: 1.05rem 1.15rem;
            }
            .stat-card .value { font-size: 2.25rem; }
            .ticker-item { font-size: 0.95rem; }
            .stTabs [data-baseweb="tab"] {
                font-size: 1rem;
                padding: 0.62rem 1.3rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_location_index(path_str: str = "data/gazetteer/earthquake_region_neighborhoods.csv"):
    path = Path(path_str)
    empty_neigh = pd.DataFrame(columns=["province", "district", "neighborhood_clean", "lat", "lon"])
    empty_dist = pd.DataFrame(columns=["province", "district", "lat", "lon"])
    empty_prov = pd.DataFrame(columns=["province", "lat", "lon"])
    if not path.exists():
        return empty_neigh, empty_dist, empty_prov

    g = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    for column in ["province", "district", "neighborhood_clean", "lat", "lon"]:
        if column not in g.columns:
            g[column] = ""

    g["province"] = g["province"].map(normalize_location_value)
    g["district"] = g["district"].map(normalize_location_value)
    g["neighborhood_clean"] = (
        g["neighborhood_clean"]
        .astype("string")
        .fillna("")
        .map(maybe_fix_mojibake)
        .str.strip()
        .str.lower()
    )
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    neigh = g[["province", "district", "neighborhood_clean", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    dist = g.groupby(["province", "district"], dropna=False)[["lat", "lon"]].mean().reset_index()
    prov = g.groupby(["province"], dropna=False)[["lat", "lon"]].mean().reset_index()
    return neigh, dist, prov


@st.cache_data(show_spinner=False)
def _qr_data_uri(payload: str) -> str | None:
    if not QRCODE_AVAILABLE:
        return None
    try:
        img = qrcode.make(payload)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _display_path(path: str | Path | None, stand_label: str) -> str:
    if STAND_MODE:
        return stand_label
    return format_path(path)


def _filter_df(df: pd.DataFrame, schema: PredictionSchema) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        date_values = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        date_values = out["created_at_parsed"].dt.date

    valid_dates = date_values.dropna()
    if not valid_dates.empty:
        min_d = valid_dates.min()
        max_d = valid_dates.max()
    else:
        min_d = dt.date(2023, 2, 6)
        max_d = dt.date(2023, 2, 13)

    date_range = st.sidebar.date_input("Tarih aralığı", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    elif isinstance(date_range, list) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range
    out = out[(date_values >= start) & (date_values <= end)]

    if "province" in out.columns:
        provinces = sorted([value for value in out["province"].dropna().unique().tolist() if value])
        selected_provinces = st.sidebar.multiselect("İl", options=provinces, default=[])
        if selected_provinces:
            out = out[out["province"].isin(selected_provinces)]

    if "district" in out.columns:
        districts = sorted([value for value in out["district"].dropna().unique().tolist() if value])
        selected_districts = st.sidebar.multiselect("İlçe", options=districts, default=[])
        if selected_districts:
            out = out[out["district"].isin(selected_districts)]

    if "neighborhood" in out.columns:
        neighborhoods = sorted([value for value in out["neighborhood"].dropna().unique().tolist() if value])
        selected_neighborhoods = st.sidebar.multiselect("Mahalle (neighborhood)", options=neighborhoods, default=[])
        if selected_neighborhoods:
            out = out[out["neighborhood"].isin(selected_neighborhoods)]

    if "urgency_score" in out.columns and not out.empty:
        urgency_numeric = pd.to_numeric(out["urgency_score"], errors="coerce").fillna(0.0)
        lo = int(np.floor(float(urgency_numeric.min())))
        hi = int(np.ceil(float(urgency_numeric.max())))
        urgency_min = st.sidebar.slider("Aciliyet skoru (min)", min_value=lo, max_value=max(lo, hi), value=lo, step=1)
        out = out[urgency_numeric >= urgency_min]

    selected_labels = st.sidebar.multiselect(
        "İhtiyaç etiketleri",
        options=schema.labels,
        default=[],
        format_func=pretty_label,
    )
    label_mode = st.sidebar.radio("Etiket filtresi modu", options=["ANY", "ALL"], index=0)
    if selected_labels:
        selected_columns = [schema.label_to_pred[label] for label in selected_labels if label in schema.label_to_pred]
        if selected_columns:
            if label_mode == "ALL":
                out = out[out[selected_columns].sum(axis=1) == len(selected_columns)]
            else:
                out = out[out[selected_columns].sum(axis=1) > 0]

    query = st.sidebar.text_input("Metin ara")
    if query:
        query_lower = query.lower()
        tweet_clean = (
            out["tweet_clean"].astype("string").fillna("")
            if "tweet_clean" in out.columns
            else pd.Series([""] * len(out), index=out.index)
        )
        tweet_raw = (
            out["tweet"].astype("string").fillna("")
            if "tweet" in out.columns
            else pd.Series([""] * len(out), index=out.index)
        )
        mask = tweet_clean.str.lower().str.contains(query_lower, na=False) | tweet_raw.str.lower().str.contains(
            query_lower,
            na=False,
        )
        out = out[mask]

    return out


def _label_counts(df: pd.DataFrame, schema: PredictionSchema) -> pd.DataFrame:
    rows = []
    for label in schema.labels:
        pred_column = schema.label_to_pred.get(label)
        if pred_column and pred_column in df.columns:
            rows.append({"label": label, "count": int(df[pred_column].sum())})
    if not rows:
        return pd.DataFrame(columns=["label", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _label_prevalence(df_all: pd.DataFrame, df_filtered: pd.DataFrame, metadata: dict | None, schema: PredictionSchema) -> pd.DataFrame:
    meta_row_count = metadata.get("row_count") if metadata else None
    try:
        full_row_count = int(meta_row_count) if meta_row_count is not None else int(len(df_all))
    except (TypeError, ValueError):
        full_row_count = int(len(df_all))

    meta_pred_positives = metadata.get("pred_positives", {}) if metadata else {}
    rows = []
    for label in schema.labels:
        pred_column = schema.label_to_pred.get(label)
        if not pred_column or pred_column not in df_all.columns:
            continue

        full_positive = meta_pred_positives.get(label)
        if full_positive is None:
            full_positive = int(df_all[pred_column].sum())
        filtered_positive = int(df_filtered[pred_column].sum()) if pred_column in df_filtered.columns else 0
        full_rate = (float(full_positive) / full_row_count * 100.0) if full_row_count else 0.0
        filtered_rate = (float(filtered_positive) / len(df_filtered) * 100.0) if len(df_filtered) else 0.0
        rows.append(
            {
                "label": label,
                "full_positive": int(full_positive),
                "full_rate_pct": round(full_rate, 2),
                "filtered_positive": filtered_positive,
                "filtered_rate_pct": round(filtered_rate, 2),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "full_positive", "full_rate_pct", "filtered_positive", "filtered_rate_pct"])
    return pd.DataFrame(rows).sort_values("full_positive", ascending=False).reset_index(drop=True)


def _province_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if "province" not in df.columns:
        return pd.DataFrame(columns=["province", "count", "lat", "lon"])
    non_empty = df[df["province"].astype("string").fillna("").str.strip() != ""]
    grouped = non_empty.groupby("province", dropna=False).size().reset_index(name="count")
    grouped["lat"] = np.nan
    grouped["lon"] = np.nan
    for index, row in grouped.iterrows():
        province = str(row["province"])
        if province in PROVINCE_CENTROID:
            lat, lon = PROVINCE_CENTROID[province]
            grouped.at[index, "lat"] = lat
            grouped.at[index, "lon"] = lon
    return grouped.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def _render_source_banner(source_kind: str, metadata: dict | None, default_note: str) -> None:
    banner_class = {
        "canonical_final": "canonical",
        "canonical_candidate": "candidate",
        "historical": "historical",
    }.get(source_kind, "custom")

    meta_line = []
    if metadata and metadata.get("selected_experiment_key"):
        meta_line.append(f"Experiment: {metadata['selected_experiment_key']}")
    if metadata and metadata.get("threshold_source"):
        threshold_type = metadata.get("threshold_type", "n/a")
        meta_line.append(f"Eşikler: {metadata['threshold_source']} / {threshold_type}")
    if metadata and metadata.get("generated_at"):
        meta_line.append(f"Üretildi: {format_generated_at(metadata.get('generated_at'))}")

    meta_text = " | ".join(meta_line)
    banner_body = source_kind_note(source_kind)
    if default_note:
        banner_body = f"{banner_body} {default_note}"

    st.markdown(
        f"""
        <div class="source-banner {banner_class}">
            <div class="eyebrow">Veri Provenance</div>
            <div class="title">{html.escape(source_kind_label(source_kind))}</div>
            <div class="body">{html.escape(banner_body)}</div>
            <div class="meta">{html.escape(meta_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stat_card_html(label: str, value: str, accent: str = "blue", delta: str | None = None) -> str:
    delta_html = f'<div class="delta">{html.escape(delta)}</div>' if delta else ""
    return (
        f'<div class="stat-card accent-{accent}">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'{delta_html}'
        f'</div>'
    )


def _render_hero(df_all: pd.DataFrame, df_filtered: pd.DataFrame, schema: PredictionSchema, metadata: dict | None) -> None:
    label_counts = _label_counts(df_filtered, schema)
    top_label = pretty_label(label_counts.iloc[0]["label"]) if not label_counts.empty else "n/a"
    top_label_value = int(label_counts.iloc[0]["count"]) if not label_counts.empty else 0

    any_need_total = 0
    if "pred_any_need" in df_filtered.columns and len(df_filtered):
        any_need_total = int(pd.to_numeric(df_filtered["pred_any_need"], errors="coerce").fillna(0).sum())

    province_count = 0
    if "province" in df_filtered.columns:
        province_count = int(df_filtered["province"].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique())

    urgency_mean = 0.0
    if "urgency_score" in df_filtered.columns and len(df_filtered):
        urgency_mean = float(pd.to_numeric(df_filtered["urgency_score"], errors="coerce").fillna(0.0).mean())

    generated_at = format_generated_at(metadata.get("generated_at")) if metadata else None
    experiment = metadata.get("selected_experiment_key") if metadata else None

    cards = [
        _stat_card_html("Toplam tweet", f"{len(df_all):,}", "blue"),
        _stat_card_html("Görünen tweet", f"{len(df_filtered):,}", "purple"),
        _stat_card_html("İhtiyaç sinyali", f"{any_need_total:,}", "red"),
        _stat_card_html("Kapsanan il", f"{province_count}", "green"),
        _stat_card_html("Ort. aciliyet", f"{urgency_mean:.2f}", "orange"),
        _stat_card_html("En kritik etiket", top_label, "red", delta=f"{top_label_value:,} sinyal" if top_label_value else None),
    ]

    chip_meta = []
    if experiment:
        chip_meta.append(f"Experiment: {experiment}")
    if generated_at:
        chip_meta.append(f"Üretildi: {generated_at}")
    chip_meta.append("Veri: 5-13 Şubat 2023 | 10 il + konumu belirsiz kayıtlar")

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-grid">
                <div class="hero-left">
                    <span class="hero-eyebrow"><span class="pulse"></span> CANLI DEMO  -  AFET YÖNETİMİ</span>
                    <div class="hero-title">Afet Aciliyet Sinyalleri | Sosyal Medya Tabanlı İhtiyaç Tespiti</div>
                    <div class="hero-sub">
                        6 Şubat 2023 Kahramanmaraş depremi sonrası atılan tweet'leri 9 ihtiyaç kategorisine sınıflıyor,
                        konum çıkarımı ile saatlik sıcak nokta haritası üretiyoruz. Amaç: kriz anında en acil sinyallere
                        erken yanıt verebilmek.
                    </div>
                    <div class="hero-meta">
                        {''.join(f'<span class="hero-chip">{html.escape(c)}</span>' for c in chip_meta)}
                    </div>
                </div>
                <div class="hero-right">
                    {''.join(cards)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_ticker(df: pd.DataFrame, schema: PredictionSchema, max_items: int = 18) -> None:
    if STAND_MODE:
        items_html = [
            f'<span class="ticker-item"><span class="badge b-{html.escape(label)}">'
            f"{html.escape(pretty_label(label))}</span>"
            f"<span>{html.escape(text)} | {html.escape(province)}</span></span>"
            for label, text, province in DEMO_TICKER_ITEMS
        ]
        track_html = "".join(items_html)
        st.markdown(
            f"""
            <div class="ticker-wrap">
                <div class="ticker-track">{track_html}{track_html}{track_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if df.empty or "tweet_clean" not in df.columns:
        return

    samples = df.copy()
    if "urgency_score" in samples.columns:
        samples["__u"] = pd.to_numeric(samples["urgency_score"], errors="coerce").fillna(0.0)
        samples = samples.sort_values("__u", ascending=False)
    samples = samples.head(max_items * 2)

    items_html = []
    for _, row in samples.iterrows():
        text = str(row.get("tweet_clean") or "").strip().replace("\n", " ")
        if not text:
            continue
        if len(text) > 140:
            text = text[:137] + "..."

        labels = []
        for label in schema.labels:
            pred_column = schema.label_to_pred.get(label)
            if pred_column and pred_column in row.index:
                try:
                    if int(row[pred_column]) == 1:
                        labels.append(label)
                except (ValueError, TypeError):
                    continue
            if len(labels) >= 2:
                break

        province = str(row.get("province") or "").strip()
        loc_part = f" | {html.escape(province)}" if province else ""
        badges = "".join(
            f'<span class="badge b-{html.escape(lab)}">{html.escape(pretty_label(lab))}</span>' for lab in labels
        )
        items_html.append(
            f'<span class="ticker-item">{badges}<span>{html.escape(text)}{loc_part}</span></span>'
        )
        if len(items_html) >= max_items:
            break

    if not items_html:
        return

    track_html = "".join(items_html)
    st.markdown(
        f"""
        <div class="ticker-wrap">
            <div class="ticker-track">{track_html}{track_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _severity_legend() -> str:
    return (
        '<div class="severity-legend">'
        '<span class="severity-pill"><span class="dot sev-critical" style="background:#7f0000"></span> Kritik (>=p95)</span>'
        '<span class="severity-pill"><span class="dot sev-high" style="background:#cb181d"></span> Yüksek (>=p80)</span>'
        '<span class="severity-pill"><span class="dot sev-medium" style="background:#ef3b2c"></span> Orta (>=p50)</span>'
        '<span class="severity-pill"><span class="dot sev-watch" style="background:#fd8d3c"></span> İzleme</span>'
        '</div>'
    )


def _render_qr_card(label: str, url: str) -> None:
    data_uri = _qr_data_uri(url)
    if not data_uri:
        st.markdown(
            f"""
            <div class="qr-card">
                <div>
                    <div class="qr-label">{html.escape(label)}</div>
                    <div class="qr-link">{html.escape(url)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="qr-card">
            <img src="{data_uri}" width="78" height="78" />
            <div>
                <div class="qr-label">{html.escape(label)}</div>
                <div class="qr-link">{html.escape(url)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_label_counts(counts_df: pd.DataFrame) -> None:
    if counts_df.empty:
        st.info("Görünen veri için pred_* label dağılımı bulunamadı.")
        return
    view = counts_df.copy()
    view["label_pretty"] = view["label"].map(pretty_label)
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            view, x="count", y="label_pretty", orientation="h",
            color="count", color_continuous_scale=["#1e293b", "#dc2626"],
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Sinyal sayısı", yaxis_title="",
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.bar_chart(view.set_index("label_pretty")["count"])


def _plot_prevalence(prevalence_df: pd.DataFrame) -> None:
    if prevalence_df.empty:
        return
    view = prevalence_df.copy()
    view["label_pretty"] = view["label"].map(pretty_label)
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_bar(name="Tüm veri (%)", x=view["label_pretty"], y=view["full_rate_pct"], marker_color="#3b82f6")
        fig.add_bar(name="Filtreli (%)", x=view["label_pretty"], y=view["filtered_rate_pct"], marker_color="#dc2626")
        fig.update_layout(
            barmode="group", template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="", yaxis_title="Yüzde",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.bar_chart(view.set_index("label_pretty")[["full_rate_pct", "filtered_rate_pct"]])


def _plot_temporal(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        st.info("date kolonu yok; zaman serisi çizilemedi.")
        return
    timeline = df.groupby("date").size().reset_index(name="count").sort_values("date")
    if timeline.empty:
        st.info("Zaman serisi için veri yok.")
        return
    if PLOTLY_AVAILABLE:
        fig = px.area(timeline, x="date", y="count", template=PLOTLY_TEMPLATE)
        fig.update_traces(line_color="#dc2626", fillcolor="rgba(220,38,38,0.3)")
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Tarih", yaxis_title="Tweet sayısı",
            height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.line_chart(timeline.set_index("date")["count"])


def _render_provenance_panel(
    csv_path: str,
    meta_path: str | None,
    metadata: dict | None,
    df_all: pd.DataFrame,
    source_kind: str,
) -> None:
    meta_row_count = metadata.get("row_count") if metadata else None
    duplicate_rows_removed = metadata.get("duplicate_rows_removed") if metadata else None
    rows_before = metadata.get("rows_before") if metadata else None
    rows_after = metadata.get("rows_after") if metadata else None

    left_col, right_col = st.columns([1.55, 1.0], gap="large")
    with left_col:
        st.subheader("Model Provenance")
        st.markdown(f"**Tahmin CSV**  \n`{_display_path(csv_path, 'Canonical final tahmin dosyası')}`")
        st.markdown(f"**Metadata JSON**  \n`{_display_path(meta_path, 'Canonical metadata dosyası') if meta_path else 'n/a'}`")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Satır (CSV)", f"{len(df_all):,}")
        metric_cols[1].metric("Satır (meta)", f"{int(meta_row_count):,}" if meta_row_count is not None else "n/a")
        metric_cols[2].metric(
            "Tekil kayıt farkı",
            f"{int(duplicate_rows_removed):,}" if duplicate_rows_removed is not None else "n/a",
        )

        if metadata:
            st.markdown(f"**Seçili experiment key**: `{metadata.get('selected_experiment_key', 'n/a')}`")
            st.markdown(f"**Seçili model dir**: `{metadata.get('model_dir', 'n/a')}`")
            threshold_source = metadata.get("threshold_source", "n/a")
            threshold_type = metadata.get("threshold_type", "n/a")
            st.markdown(f"**Eşik kaynağı / türü**: `{threshold_source}` / `{threshold_type}`")
            st.markdown(f"**Üretildi**: `{format_generated_at(metadata.get('generated_at'))}`")

            if rows_before is not None and rows_after is not None:
                st.caption(f"Dedup özeti: rows_before={rows_before:,} -> rows_after={rows_after:,}")

            if meta_row_count is not None and int(meta_row_count) != len(df_all):
                st.warning(
                    f"Metadata row_count ({int(meta_row_count):,}) ile yüklenen CSV satır sayısı ({len(df_all):,}) farklı."
                )
        else:
            st.info("Metadata bulunamadı. Dashboard CSV-only fallback modunda çalışıyor.")

        if source_kind == "historical":
            st.warning("Historical 63k preview dosyası aktif. Bu artifact canonical final output değildir.")

    with right_col:
        st.subheader("Canonical Sınırlar")
        if source_kind in {"canonical_final", "canonical_candidate"}:
            st.markdown("\n".join([f"- {item}" for item in canonical_limitations(metadata)]))
        elif source_kind == "historical":
            st.markdown(
                "- Historical 63k preview aktif; canonical experiment metadata'si veya final sınır seti doğrudan bağlı değil.\n"
                "- Bu dosya karşılaştırma için korunuyor, final production output olarak sunulmuyor."
            )
        else:
            st.markdown(
                "- Custom CSV/meta seçildi. Sınırlar seçili dosyanın gerçek provenance'ına göre yorumlanmalı.\n"
                "- Metadata sağlanırsa canonical riskler ve threshold bilgileri daha güvenli şekilde görüntülenir."
            )


def _render_schema_panel(metadata: dict | None, schema: PredictionSchema, prevalence_df: pd.DataFrame) -> None:
    with st.expander("Prediction Schema ve Metadata"):
        if metadata and metadata.get("schema_note"):
            st.caption(metadata.get("schema_note"))
        elif metadata:
            st.caption("Metadata yüklendi; column mapping metadata'dan okunuyor.")
        else:
            st.caption("Metadata yok; pred/prob kolonları CSV header'ından keşfedildi.")

        schema_rows = []
        meta_pred_positives = metadata.get("pred_positives", {}) if metadata else {}
        for label in schema.labels:
            schema_rows.append(
                {
                    "label": label,
                    "pred_column": schema.label_to_pred.get(label, ""),
                    "prob_column": schema.label_to_prob.get(label, ""),
                    "meta_pred_positives": meta_pred_positives.get(label, "n/a"),
                }
            )
        st.dataframe(pd.DataFrame(schema_rows), width="stretch", hide_index=True)

        if not prevalence_df.empty:
            st.caption("Pred prevalence (full dataset vs current filter)")
            st.dataframe(prevalence_df, width="stretch", hide_index=True)


def _advance_timeline_hour(hour_values: list[dt.datetime]) -> None:
    try:
        idx = hour_values.index(st.session_state["timeline_hour"])
    except ValueError:
        idx = 0
        st.session_state["timeline_hour"] = hour_values[0]

    if idx < (len(hour_values) - 1):
        st.session_state["timeline_hour"] = hour_values[idx + 1]
    elif bool(st.session_state.get("timeline_loop", True)):
        st.session_state["timeline_hour"] = hour_values[0]
    else:
        st.session_state["timeline_playing"] = False
        st.session_state.pop("timeline_last_tick", None)


def _default_timeline_hour(df: pd.DataFrame, hours: pd.Series, hour_values: list[dt.datetime]) -> dt.datetime:
    if not STAND_MODE or not hour_values:
        return hour_values[0]

    frame = pd.DataFrame({"_hour": hours}, index=df.index).dropna(subset=["_hour"])
    if frame.empty:
        return hour_values[0]

    if "pred_any_need" in df.columns:
        positive = pd.to_numeric(df["pred_any_need"], errors="coerce").fillna(0).astype(int) == 1
        frame = frame[positive.reindex(frame.index).fillna(False)]
        if frame.empty:
            frame = pd.DataFrame({"_hour": hours}, index=df.index).dropna(subset=["_hour"])

    counts = frame.groupby("_hour").size().sort_values(ascending=False)
    if counts.empty:
        return hour_values[0]
    return counts.index[0].to_pydatetime()


def _hourly_signal_map(df: pd.DataFrame, schema: PredictionSchema) -> None:
    if bool(st.session_state.get("timeline_playing", False)):
        interval_s = float(st.session_state.get("timeline_interval_s", 1.2))
        interval_s = max(0.8, min(interval_s, 5.0))

        @st.fragment(run_every=interval_s)
        def _autoplay_fragment() -> None:
            _render_hourly_signal_map(df, schema, auto_advance=True)

        _autoplay_fragment()
    else:
        _render_hourly_signal_map(df, schema, auto_advance=False)


def _render_hourly_signal_map(df: pd.DataFrame, schema: PredictionSchema, *, auto_advance: bool) -> None:
    st.subheader("Saatlik Yardım Sinyalleri (Harita)")
    st.markdown(_severity_legend(), unsafe_allow_html=True)

    if "created_at_local" not in df.columns or df["created_at_local"].isna().all():
        st.info("created_at bilgisi yok; saatlik harita oluşturulamadı.")
        return

    hours = pd.to_datetime(df["created_at_local"], errors="coerce").dt.floor("h")
    try:
        hours = hours.dt.tz_localize(None)
    except Exception:
        pass

    hour_values = sorted(pd.DatetimeIndex(hours.dropna().unique()).to_pydatetime().tolist())
    if not hour_values:
        st.info("Saat bilgisi bulunamadı.")
        return

    if "timeline_playing" not in st.session_state:
        st.session_state["timeline_playing"] = False
    if "timeline_interval_s" not in st.session_state:
        st.session_state["timeline_interval_s"] = 1.2
    if "timeline_loop" not in st.session_state:
        st.session_state["timeline_loop"] = True
    if "timeline_show_heatmap" not in st.session_state:
        st.session_state["timeline_show_heatmap"] = True
    if "timeline_pending_hour" in st.session_state:
        pending = st.session_state.pop("timeline_pending_hour")
        if pending in hour_values:
            st.session_state["timeline_hour"] = pending

    if "timeline_hour" not in st.session_state or st.session_state["timeline_hour"] not in hour_values:
        st.session_state["timeline_hour"] = _default_timeline_hour(df, hours, hour_values)

    interval_s = float(st.session_state.get("timeline_interval_s", 1.2))
    interval_s = max(0.8, min(interval_s, 5.0))
    if bool(st.session_state.get("timeline_playing", False)) and auto_advance and len(hour_values) > 1:
        now = time.monotonic()
        previous_tick = st.session_state.get("timeline_last_tick")
        if previous_tick is None:
            st.session_state["timeline_last_tick"] = now
        elif (now - float(previous_tick)) >= interval_s:
            _advance_timeline_hour(hour_values)
            st.session_state["timeline_last_tick"] = now
    elif not bool(st.session_state.get("timeline_playing", False)):
        st.session_state.pop("timeline_last_tick", None)

    c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.7, 1.2, 1.5])
    with c1:
        loc_level = st.selectbox("Konum seviyesi", options=["neighborhood", "district", "province"], index=0)
    with c2:
        signal_mode = st.selectbox(
            "Sinyal gücü",
            options=["count_rows", "count_any_need", "sum_urgency"],
            index=1 if "pred_any_need" in df.columns else 0,
            format_func={
                "count_rows": "Tweet sayısı",
                "count_any_need": "İhtiyaç sinyali sayısı (pred_any_need=1)",
                "sum_urgency": "Urgency toplam (urgency_score)",
            }.get,
        )
    with c3:
        st.slider(
            "Oynatma hızı (sn)",
            min_value=0.8,
            max_value=5.0,
            step=0.1,
            key="timeline_interval_s",
        )
    with c4:
        st.checkbox("Döngü", key="timeline_loop")
    with c5:
        st.checkbox("Isı haritası", key="timeline_show_heatmap")

    if loc_level == "province":
        group_cols = ["province"]
    elif loc_level == "district":
        group_cols = ["province", "district"]
    else:
        group_cols = ["province", "district", "neighborhood"]

    missing_group = [column for column in group_cols if column not in df.columns]
    if missing_group:
        st.info(f"Konum alanlari eksik: {', '.join(missing_group)}")
        return

    df_hourly = df.copy()
    df_hourly["_hour"] = hours

    def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=group_cols + ["signal"])

        if signal_mode == "count_any_need" and "pred_any_need" in frame.columns:
            frame = frame[pd.to_numeric(frame["pred_any_need"], errors="coerce").fillna(0).astype(int) == 1]
        if frame.empty:
            return pd.DataFrame(columns=group_cols + ["signal"])

        if signal_mode == "sum_urgency" and "urgency_score" in frame.columns:
            return frame.groupby(group_cols, dropna=False)["urgency_score"].sum().reset_index(name="signal")
        return frame.groupby(group_cols, dropna=False).size().reset_index(name="signal")

    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    try:
        cur_idx = hour_values.index(st.session_state["timeline_hour"])
    except ValueError:
        cur_idx = 0
        st.session_state["timeline_hour"] = hour_values[0]

    with b1:
        if st.button("Önceki", width="stretch"):
            st.session_state["timeline_playing"] = False
            st.session_state.pop("timeline_last_tick", None)
            st.session_state["timeline_hour"] = hour_values[max(0, cur_idx - 1)]
            st.rerun()
    with b2:
        button_label = "Duraklat" if st.session_state["timeline_playing"] else "Oynat"
        if st.button(button_label, width="stretch"):
            next_playing = not bool(st.session_state["timeline_playing"])
            st.session_state["timeline_playing"] = next_playing
            if next_playing:
                st.session_state["timeline_last_tick"] = time.monotonic()
            else:
                st.session_state.pop("timeline_last_tick", None)
            st.rerun()
    with b3:
        if st.button("Sonraki", width="stretch"):
            st.session_state["timeline_playing"] = False
            st.session_state.pop("timeline_last_tick", None)
            st.session_state["timeline_hour"] = hour_values[min(len(hour_values) - 1, cur_idx + 1)]
            st.rerun()
    with b4:
        st.caption(f"Saat sayısı: {len(hour_values)}")

    st.select_slider(
        "Saat seç",
        options=hour_values,
        key="timeline_hour",
        format_func=lambda value: value.strftime("%Y-%m-%d %H:00"),
    )
    selected_hour = st.session_state["timeline_hour"]

    df_selected = df_hourly[df_hourly["_hour"] == selected_hour]
    if df_selected.empty:
        st.info("Bu saatte filtreye uyan veri yok.")
        return

    aggregated = _aggregate(df_selected)
    if aggregated.empty:
        if signal_mode == "count_any_need":
            st.info("Bu saatte (pred_any_need=1) sinyal yok.")
        else:
            st.info("Bu saatte sinyal yok.")
        return

    neigh_ix, dist_ix, prov_ix = load_location_index()

    if loc_level == "neighborhood":
        aggregated["neighborhood_clean"] = (
            aggregated["neighborhood"].astype("string").fillna("").str.strip().str.lower()
        )
        aggregated = aggregated[aggregated["neighborhood_clean"] != ""].reset_index(drop=True)

        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        aggregated = aggregated.merge(neigh_ix, on=["province", "district", "neighborhood_clean"], how="left")
        aggregated = aggregated.merge(dist_ix, on=["province", "district"], how="left")
        aggregated = aggregated.merge(prov_ix, on=["province"], how="left")
        aggregated["lat"] = aggregated["lat"].fillna(aggregated.get("lat_dist")).fillna(aggregated.get("lat_prov"))
        aggregated["lon"] = aggregated["lon"].fillna(aggregated.get("lon_dist")).fillna(aggregated.get("lon_prov"))
        aggregated = aggregated.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")
    elif loc_level == "district":
        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        aggregated = aggregated.merge(dist_ix, on=["province", "district"], how="left")
        aggregated = aggregated.merge(prov_ix, on=["province"], how="left")
        aggregated["lat"] = aggregated["lat_dist"].fillna(aggregated.get("lat_prov"))
        aggregated["lon"] = aggregated["lon_dist"].fillna(aggregated.get("lon_prov"))
        aggregated = aggregated.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")
    else:
        aggregated = aggregated.merge(prov_ix, on=["province"], how="left")

    if "province" in aggregated.columns:
        missing_geo = aggregated["lat"].isna() | aggregated["lon"].isna()
        if bool(missing_geo.any()):
            for index, row in aggregated.loc[missing_geo].iterrows():
                province = str(row.get("province", ""))
                if province in PROVINCE_CENTROID:
                    lat, lon = PROVINCE_CENTROID[province]
                    aggregated.at[index, "lat"] = lat
                    aggregated.at[index, "lon"] = lon

    total_hotspots_before_geo = len(aggregated)
    aggregated = aggregated.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if aggregated.empty:
        st.info("Harita için koordinat eşleşmesi bulunamadı (gazetteer yok veya eşleşme düşük).")
        return

    aggregated["signal"] = pd.to_numeric(aggregated["signal"], errors="coerce").fillna(0.0)
    aggregated = aggregated[aggregated["signal"] > 0].sort_values("signal", ascending=False).reset_index(drop=True)
    if aggregated.empty:
        st.info("Haritada gösterilecek pozitif sinyal bulunamadı.")
        return

    total_signal = float(aggregated["signal"].sum())
    aggregated["rank"] = np.arange(1, len(aggregated) + 1)
    aggregated["share_pct"] = np.where(total_signal > 0, (aggregated["signal"] / total_signal) * 100.0, 0.0)
    q50 = float(aggregated["signal"].quantile(0.50))
    q80 = float(aggregated["signal"].quantile(0.80))
    q95 = float(aggregated["signal"].quantile(0.95))

    def _severity(value: float) -> str:
        if value >= q95:
            return "Kritik"
        if value >= q80:
            return "Yüksek"
        if value >= q50:
            return "Orta"
        return "İzleme"

    def _signal_color(value: float) -> list[int]:
        if value >= q95:
            return [127, 0, 0, 240]
        if value >= q80:
            return [203, 24, 29, 232]
        if value >= q50:
            return [239, 59, 44, 224]
        return [253, 141, 60, 216]

    aggregated["severity"] = aggregated["signal"].apply(_severity)
    aggregated["fill_color"] = aggregated["signal"].apply(_signal_color)

    if loc_level == "province":
        aggregated["location_label"] = aggregated["province"].astype("string").fillna("").str.strip()
    elif loc_level == "district":
        aggregated["location_label"] = (
            aggregated["province"].astype("string").fillna("").str.strip()
            + " / "
            + aggregated["district"].astype("string").fillna("").str.strip()
        )
    else:
        aggregated["location_label"] = (
            aggregated["province"].astype("string").fillna("").str.strip()
            + " / "
            + aggregated["district"].astype("string").fillna("").str.strip()
            + " / "
            + aggregated["neighborhood"].astype("string").fillna("").str.strip()
        )
    aggregated["share_pct_rounded"] = aggregated["share_pct"].round(1)

    prev_total_signal = None
    prev_hotspots = None
    if cur_idx > 0:
        prev_hour = hour_values[cur_idx - 1]
        prev_agg = _aggregate(df_hourly[df_hourly["_hour"] == prev_hour])
        if not prev_agg.empty:
            prev_agg["signal"] = pd.to_numeric(prev_agg["signal"], errors="coerce").fillna(0.0)
            prev_total_signal = float(prev_agg["signal"].sum())
            prev_hotspots = int((prev_agg["signal"] > 0).sum())

    def _delta_text(current: float, previous: float | None, decimals: int = 0) -> str | None:
        if previous is None:
            return None
        diff = current - previous
        if previous == 0:
            return f"{diff:+.{decimals}f} (önceki saat 0)"
        pct = (diff / previous) * 100.0
        return f"{diff:+.{decimals}f} ({pct:+.1f}%)"

    top_location = str(aggregated.iloc[0]["location_label"])
    top_signal = float(aggregated.iloc[0]["signal"])
    st.markdown(
        f"""
        <div class="signal-hero">
            <div class="signal-hero-title">Saatlik İhtiyaç Sinyalleri Analizi</div>
            <div class="signal-hero-sub">
                Saat: {selected_hour.strftime("%Y-%m-%d %H:00")} | En kritik nokta: {html.escape(top_location)} (sinyal: {top_signal:.0f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam sinyal", f"{int(total_signal):,}", delta=_delta_text(total_signal, prev_total_signal, 0))
    m2.metric(
        "Sıcak nokta",
        f"{int(len(aggregated)):,}",
        delta=_delta_text(float(len(aggregated)), float(prev_hotspots) if prev_hotspots is not None else None, 0),
    )
    m3.metric("Tepe sinyal", f"{top_signal:.0f}")
    geo_match = (len(aggregated) / total_hotspots_before_geo) * 100.0 if total_hotspots_before_geo else 0.0
    m4.metric("Koordinat kapsamı", f"{geo_match:.1f}%")

    map_col, table_col = st.columns([1.7, 1.0], gap="large")
    with map_col:
        radius_scale = st.slider("Nokta boyutu çarpanı", min_value=500, max_value=22000, value=6500, step=500)
        aggregated["radius"] = (np.sqrt(aggregated["signal"].clip(lower=0.0)) * float(radius_scale)).clip(lower=1000.0)

        center_lat = float(aggregated["lat"].mean())
        center_lon = float(aggregated["lon"].mean())
        tooltip = {
            "html": (
                "<b>{location_label}</b><br/>"
                + f"Saat: {selected_hour.strftime('%Y-%m-%d %H:00')}<br/>"
                + "Sinyal: {signal}<br/>Pay: {share_pct_rounded}%<br/>Seviye: {severity}"
            )
        }

        layers: list[pdk.Layer] = []
        if bool(st.session_state.get("timeline_show_heatmap", False)):
            layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=aggregated,
                    get_position="[lon, lat]",
                    get_weight="signal",
                    radius_pixels=75,
                    intensity=1.2,
                    threshold=0.03,
                    opacity=0.42,
                )
            )
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=aggregated,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="fill_color",
                get_line_color=[35, 10, 10, 235],
                line_width_min_pixels=1,
                radius_min_pixels=4,
                pickable=True,
                stroked=True,
            )
        )

        zoom_map = 6.4 if loc_level == "neighborhood" else (6.0 if loc_level == "district" else 5.6)
        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom_map, pitch=22),
            layers=layers,
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, width="stretch")
        st.caption(
            "Gösterim; seçili saatteki sinyal yoğunluğunu (renk), büyüklüğünü (çap) ve hotspot öncelik seviyesini birlikte sunar."
        )

    with table_col:
        top_n_max = int(min(30, len(aggregated)))
        top_n_default = int(min(10, len(aggregated)))
        if top_n_max <= 1:
            top_n = top_n_max if top_n_max >= 1 else 0
            if top_n >= 1:
                st.caption(f"Sıcak nokta listesi: {top_n} kayıt")
        else:
            top_n = st.slider(
                "Sıcak nokta listesi",
                min_value=1,
                max_value=top_n_max,
                value=max(1, min(top_n_default, top_n_max)),
                step=1,
            )
        top_df = aggregated.head(top_n).copy()
        top_df["signal"] = top_df["signal"].round(0).astype(int)
        top_df["pay"] = top_df["share_pct"].map(lambda value: f"{value:.1f}%")
        st.dataframe(
            top_df[["rank", "location_label", "signal", "pay", "severity"]].rename(
                columns={
                    "rank": "Sıra",
                    "location_label": "Konum",
                    "signal": "Sinyal",
                    "pay": "Pay",
                    "severity": "Seviye",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        if PLOTLY_AVAILABLE:
            mini = top_df.head(10).iloc[::-1]
            fig = px.bar(
                mini, x="signal", y="location_label", orientation="h",
                color="signal", color_continuous_scale=["#fed976", "#dc2626"],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Sinyal", yaxis_title="",
                coloraxis_showscale=False,
                height=320,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.bar_chart(top_df[["location_label", "signal"]].set_index("location_label"))

        label_rows = []
        for label in schema.labels:
            pred_column = schema.label_to_pred.get(label)
            if pred_column and pred_column in df_selected.columns:
                count = int(pd.to_numeric(df_selected[pred_column], errors="coerce").fillna(0).astype(int).sum())
                if count > 0:
                    label_rows.append({"label": pretty_label(label), "count": count})
        if label_rows:
            label_df = pd.DataFrame(label_rows).sort_values("count", ascending=False).head(6)
            st.caption("Saatlik ihtiyaç etiketleri")
            st.dataframe(label_df, width="stretch", hide_index=True)

    with st.expander("Saatlik sinyal tablosu (detaylı)"):
        view_cols = ["rank", "location_label", "signal", "share_pct", "severity", "lat", "lon"]
        show_df = aggregated[view_cols].copy()
        show_df["share_pct"] = show_df["share_pct"].map(lambda value: round(float(value), 2))
        st.dataframe(show_df, width="stretch", hide_index=True)


# ---------- TWEET TEST TAB ----------

EXAMPLE_TWEETS: list[tuple[str, str]] = [
    (
        "Arama-kurtarma",
        "Hatay Antakya Sümerler mahallesi enkaz altında kaldık lütfen yardım",
    ),
    (
        "Gıda & su",
        "Kahramanmaraş Dulkadiroğlu Yörükselim mah. su ve yiyecek kalmadı çok ihtiyacımız var",
    ),
    (
        "Barınma",
        "Adıyaman merkez evimiz hasarlı çadıra ihtiyaç var ailecek dışarıdayız",
    ),
    (
        "Sağlık",
        "Gaziantep Şehitkamil yaralı var insülin lazım acil sağlık ekibi çağırıyoruz",
    ),
    (
        "Bilgi paylaşımı",
        "Malatya Yeşilyurt'tan haber alamıyoruz iletişim yok lütfen ulaşabilen olursa paylaşsın",
    ),
]


@st.cache_resource(show_spinner=False)
def _cached_load_bundle(model_ref: str, labels_path: str, thresholds_path: str, max_length: int, prefer_cpu: bool) -> ModelBundle:
    location = make_user_location(model_ref, labels_path, thresholds_path)
    return load_bundle(location, max_length=max_length, prefer_cpu=prefer_cpu)


def _render_prediction_chart(result: PredictionResult) -> None:
    df = pd.DataFrame(
        {
            "label": result.labels,
            "prob": result.probs.astype(float),
            "threshold": result.thresholds.astype(float),
        }
    )
    df["label_pretty"] = df["label"].map(pretty_label)
    df["predicted"] = df["prob"] >= df["threshold"]
    df = df.sort_values("prob", ascending=True).reset_index(drop=True)

    if PLOTLY_AVAILABLE:
        bar_colors = [("#dc2626" if pred else "#475569") for pred in df["predicted"]]
        fig = go.Figure()
        fig.add_bar(
            x=df["prob"],
            y=df["label_pretty"],
            orientation="h",
            marker_color=bar_colors,
            name="Olasılık",
            hovertemplate="<b>%{y}</b><br>prob=%{x:.3f}<extra></extra>",
        )
        fig.add_trace(
            go.Scatter(
                x=df["threshold"],
                y=df["label_pretty"],
                mode="markers",
                marker=dict(symbol="line-ns", color="#fbbf24", size=18, line=dict(width=2, color="#fbbf24")),
                name="Eşik (CV)",
                hovertemplate="<b>%{y}</b><br>thr=%{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Olasılık (sigmoid)", range=[0, 1]),
            yaxis=dict(title=""),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.bar_chart(df.set_index("label_pretty")["prob"])


def _render_tweet_test_tab() -> None:
    st.subheader("Tweet Test - Canlı Etiket Tahmini")
    st.caption(
        "Bir tweet ya da kısa metin yazın; canonical leak-free model "
        "(`exp3_silver_then_gold_v3_exgold`) 9 ihtiyaç etiketi için olasılık ve "
        "CV-tuned eşiklere göre tahmin üretsin."
    )

    auto_loc = discover_model_location()
    candidates = describe_candidates()

    default_ref = (
        auto_loc.model_ref
        if auto_loc is not None
        else (candidates[0]["model_dir"].removeprefix("hf:") if candidates else "")
    )
    default_labels = str(auto_loc.labels_path) if auto_loc else (candidates[0]["labels"] if candidates else "")
    default_thresholds = str(auto_loc.thresholds_path) if auto_loc else (candidates[0]["thresholds"] if candidates else "")

    if STAND_MODE:
        model_ref_input = default_ref
        labels_input = default_labels
        thresholds_input = default_thresholds
        max_length = 192
        prefer_cpu = False
        apply_clean = True
        if auto_loc is None:
            st.warning("Canlı tahmin modeli bulunamadı. Stand sunumu öncesi model yolu veya HF repo ayarlanmalı.")
            return
    else:
        with st.expander("Model kaynağı ayarları", expanded=(auto_loc is None)):
            st.caption(
                "Model checkpoint'i ~440 MB olduğu için repo içinde tutulmuyor. İki seçenek var: "
                "**(a)** lokal disk yolu, **(b)** HuggingFace Hub repo id'si "
                "(`kullanici/repo`). HF Hub modu Streamlit Cloud'da önerilen yöntemdir."
            )
            for cand in candidates:
                badge = "[OK]" if cand["exists"] else "[--]"
                kind = "HF" if cand.get("is_hf") else "PATH"
                st.markdown(f"- {badge} `{kind}` **{cand['source']}** -> `{cand['model_dir']}`")
            st.caption(
                "Ortam değişkenleri / Streamlit secrets: "
                f"`{ENV_HF_REPO}` (HF repo), `AFETYONETIMI_MODEL_DIR` (lokal yol), "
                "`AFETYONETIMI_HF_TOKEN` (private repo için)."
            )

        cfg_col_a, cfg_col_b = st.columns([1.6, 1.0], gap="medium")
        with cfg_col_a:
            model_ref_input = st.text_input(
                "Model kaynağı (HF repo id veya lokal yol)",
                value=default_ref,
                key="tt_model_ref",
                help="Örnek HF: `ahmedberatAI/afet-need-classifier` -- örnek lokal: `C:/.../models/.../final`",
            )
            labels_input = st.text_input(
                "label_columns.json",
                value=default_labels,
                key="tt_labels",
                help="Boş bırakırsan repo içindeki bundle (`data/model_meta/label_columns.json`) kullanılır.",
            )
            thresholds_input = st.text_input(
                "thresholds_cv.json",
                value=default_thresholds,
                key="tt_thr",
                help="Boş bırakırsan repo içindeki bundle (`data/model_meta/thresholds_cv.json`) kullanılır.",
            )
        with cfg_col_b:
            max_length = st.slider("Tokenizer max_length", min_value=64, max_value=384, value=192, step=16)
            prefer_cpu = st.checkbox("CPU kullan (GPU varsa bile)", value=False)
            apply_clean = st.checkbox("Metni ön-temizle (NFC + whitespace)", value=True)
            st.caption("Metin temizliği `preprocess_emergency_data.clean_text` ile aynı: NFC normalize + whitespace.")

    if not (model_ref_input or "").strip():
        st.warning(
            "Model kaynağı boş. HF Hub'a model yükledikten sonra `kullanici/repo` formatında "
            "buraya yaz veya `AFETYONETIMI_MODEL_HF_REPO` Streamlit secret'i tanımla."
        )
        return

    is_hf_ref = looks_like_hf_repo_id(model_ref_input.strip())
    if not is_hf_ref:
        local_path = Path(model_ref_input).expanduser()
        if not local_path.exists():
            st.error(
                f"Lokal yol bulunamadı: `{local_path}`\n\n"
                "Streamlit Cloud'da yerel disk yok; HF Hub'a model yükleyip "
                f"`{ENV_HF_REPO}` (veya bu alan) içine `kullanici/repo` formatında yaz. "
                "HF Hub yükleme adımlarını README'de görebilirsin."
            )
            return

    # Labels / thresholds: boş = bundle. Yoksa lokal varlığı şart.
    labels_resolved = (Path(labels_input).expanduser() if labels_input else None)
    thresholds_resolved = (Path(thresholds_input).expanduser() if thresholds_input else None)
    missing_meta = []
    if labels_resolved is not None and not labels_resolved.exists():
        missing_meta.append(str(labels_resolved))
    if thresholds_resolved is not None and not thresholds_resolved.exists():
        missing_meta.append(str(thresholds_resolved))
    if missing_meta:
        st.error(
            "Aşağıdaki etiket/eşik dosyaları bulunamadı:\n\n"
            + "\n".join(f"- `{p}`" for p in missing_meta)
            + "\n\nBu alanları boş bırakırsan repo içindeki bundle (`data/model_meta/`) kullanılır."
        )
        return

    if STAND_MODE and PRELOAD_MODEL:
        prewarm_key = "|".join(
            [
                model_ref_input.strip(),
                str(labels_resolved) if labels_resolved else "",
                str(thresholds_resolved) if thresholds_resolved else "",
                str(max_length),
                str(prefer_cpu),
            ]
        )
        if st.session_state.get("tt_prewarm_key") != prewarm_key:
            try:
                with st.spinner("Canlı tahmin modeli hazırlanıyor..."):
                    _cached_load_bundle(
                        model_ref=(model_ref_input.strip() if is_hf_ref else str(Path(model_ref_input).expanduser())),
                        labels_path=(str(labels_resolved) if labels_resolved else ""),
                        thresholds_path=(str(thresholds_resolved) if thresholds_resolved else ""),
                        max_length=int(max_length),
                        prefer_cpu=bool(prefer_cpu),
                    )
                st.session_state["tt_prewarm_key"] = prewarm_key
                st.session_state.pop("tt_prewarm_error", None)
            except Exception as e:  # noqa: BLE001
                st.session_state["tt_prewarm_error"] = str(e)
        if st.session_state.get("tt_prewarm_error"):
            st.warning("Canlı tahmin modeli henüz hazır değil; model ayarları kontrol edilmeli.")
        elif st.session_state.get("tt_prewarm_key") == prewarm_key:
            st.caption("Canlı tahmin modeli hazır; örnek metinlerden biriyle anında test edebilirsiniz.")

    st.markdown("---")

    if "tt_text" not in st.session_state:
        st.session_state["tt_text"] = ""

    chip_cols = st.columns(len(EXAMPLE_TWEETS))
    for col, (label, sample) in zip(chip_cols, EXAMPLE_TWEETS):
        with col:
            if st.button(label, key=f"tt_chip_{label}", width="stretch"):
                st.session_state["tt_text"] = sample

    text = st.text_area(
        "Tweet / metin",
        key="tt_text",
        height=120,
        placeholder="Örnek: Hatay Antakya'da enkaz altında kalanlar var, su ve battaniye lazım...",
    )

    run_col, info_col = st.columns([1.0, 1.6])
    with run_col:
        run_clicked = st.button("Tahmin et", type="primary", width="stretch")
    with info_col:
        st.caption(
            "Etiketler: arama_kurtarma, saglik, barinma, gida_su, altyapi, "
            "guvenlik, lojistik, psikolojik, bilgi_paylasimi"
        )

    if not run_clicked:
        return
    if not (text or "").strip():
        st.warning("Önce metni yaz.")
        return

    try:
        spinner_msg = (
            "HF Hub'dan model indiriliyor / tahmin üretiliyor..."
            if is_hf_ref
            else "Model yükleniyor / tahmin üretiliyor..."
        )
        with st.spinner(spinner_msg):
            bundle = _cached_load_bundle(
                model_ref=(
                    model_ref_input.strip()
                    if is_hf_ref
                    else str(Path(model_ref_input).expanduser())
                ),
                labels_path=(str(labels_resolved) if labels_resolved else ""),
                thresholds_path=(str(thresholds_resolved) if thresholds_resolved else ""),
                max_length=int(max_length),
                prefer_cpu=bool(prefer_cpu),
            )
            result = predict_one(bundle, text, apply_clean=bool(apply_clean))
    except RuntimeError as e:
        st.error(str(e))
        st.info(
            "Tweet Test sekmesi çalışabilmek için ağır bağımlılıklara ihtiyaç duyar. "
            "Lokal kurulum:\n\n```\npip install torch transformers\n```"
        )
        return
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Tahmin sırasında hata: {e}")
        return

    result_sub = (
        "Canonical model &middot; CV ayarlı eşikler &middot; stand modu"
        if STAND_MODE
        else (
            f"Cihaz: <b>{html.escape(bundle.device)}</b> &middot; "
            f"max_length=<b>{bundle.max_length}</b> &middot; kaynak: <b>{html.escape(bundle.location.source_label)}</b>"
        )
    )
    st.markdown(
        f"<div class='signal-hero'><div class='signal-hero-title'>Tahmin sonucu</div>"
        f"<div class='signal-hero-sub'>{result_sub}</div></div>",
        unsafe_allow_html=True,
    )

    if result.predicted:
        chips = " ".join(
            f"<span class='hero-chip' style='background:rgba(220,38,38,0.18);"
            f"border-color:rgba(220,38,38,0.55);color:#fecaca;'>"
            f"{html.escape(pretty_label(lab))}</span>"
            for lab in result.predicted
        )
        st.markdown(
            f"<div style='margin:0.4rem 0 0.7rem 0;'>Tahmin edilen etiketler: {chips}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Eşiği geçen etiket yok (model hiçbir kategoriyi yeterince güvenli bulmadı).")

    _render_prediction_chart(result)

    detail_df = pd.DataFrame(
        {
            "Etiket": [pretty_label(lab) for lab in result.labels],
            "Olasılık": [round(float(p), 4) for p in result.probs],
            "Eşik (CV)": [round(float(t), 3) for t in result.thresholds],
            "Tahmin": ["[X]" if p >= t else "" for p, t in zip(result.probs, result.thresholds)],
        }
    ).sort_values("Olasılık", ascending=False)
    st.dataframe(detail_df, width="stretch", hide_index=True)

    with st.expander("Model işlenen metin (token girişinden önce)"):
        st.code(result.text or "(boş)", language="text")


# ---------- BOOT ----------

_inject_styles()

default_source = discover_default_source()

if SHOW_TECHNICAL_DETAILS:
    st.sidebar.markdown("### Veri Kaynağı")
    st.sidebar.caption(f"Otomatik varsayılan: {default_source.label}")
    st.sidebar.caption(default_source.note)

    csv_path_input = st.sidebar.text_input("Tahmin CSV yolu", value=str(default_source.csv_path))
    auto_meta = st.sidebar.checkbox("CSV yanındaki metadata dosyasını otomatik ara", value=True)

    manual_meta_default = str(default_source.meta_path) if default_source.meta_path else ""
    manual_meta_input = ""
    if not auto_meta:
        manual_meta_input = st.sidebar.text_input("Metadata JSON yolu", value=manual_meta_default)

    resolved_meta_path = infer_meta_path(csv_path_input) if auto_meta else (Path(manual_meta_input).expanduser() if manual_meta_input else None)
    if auto_meta and resolved_meta_path is not None:
        st.sidebar.caption(f"Metadata path: {format_path(resolved_meta_path)}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtreler")
else:
    csv_path_input = str(default_source.csv_path)
    resolved_meta_path = default_source.meta_path if default_source.meta_path else infer_meta_path(csv_path_input)

df_all: pd.DataFrame | None = None
try:
    with st.spinner("Tahminler yükleniyor..."):
        df_all = load_predictions_csv(csv_path_input)
except FileNotFoundError:
    st.error(f"Dosya bulunamadı: {csv_path_input}")
    df_all = None

if df_all is None or df_all.empty:
    st.stop()

metadata = load_prediction_metadata(str(resolved_meta_path) if resolved_meta_path else None)
schema = build_prediction_schema(metadata, df_all.columns.tolist())
source_kind = classify_prediction_source(csv_path_input, metadata)
try:
    active_csv_resolved = str(Path(csv_path_input).expanduser().resolve())
    default_csv_resolved = str(default_source.csv_path.expanduser().resolve())
except OSError:
    active_csv_resolved = csv_path_input
    default_csv_resolved = str(default_source.csv_path)
banner_note = default_source.note if active_csv_resolved == default_csv_resolved else ""

df_filtered = _filter_df(df_all, schema) if SHOW_TECHNICAL_DETAILS else df_all.copy()

# --- HERO + TICKER ---
_render_hero(df_all, df_filtered, schema, metadata)
_render_ticker(df_filtered if not df_filtered.empty else df_all, schema)
_render_source_banner(source_kind, metadata, banner_note)

# --- SIDEBAR LINKS / QR ---
if SHOW_TECHNICAL_DETAILS:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Proje Linkleri")
    _render_qr_card("Dashboard repo", REPO_DASHBOARD)
    _render_qr_card("Ana model repo", REPO_MAIN)
    st.sidebar.caption("QR kodu telefonunuzla taratabilirsiniz.")

# --- TABS ---
tab_map, tab_insights, tab_tweets, tab_test, tab_about = st.tabs(
    ["Canlı Harita", "İçgörüler", "Tweet Listesi", "Tweet Test", "Hakkında"]
)

with tab_map:
    if df_filtered.empty:
        st.warning("Seçili filtrelerle eşleşen satır yok. Filtreleri sidebar'dan gevşetin.")
    else:
        _hourly_signal_map(df_filtered, schema)

        st.subheader("İl Bazlı Yoğunluk (Centroid)")
        province_map_df = _province_map_df(df_filtered)
        if province_map_df.empty:
            st.info("Harita için province -> (lat, lon) eşleşmesi bulunamadı.")
        else:
            map_col, table_col = st.columns([1.6, 1.0], gap="large")
            with map_col:
                if PLOTLY_AVAILABLE:
                    fig = px.scatter_map(
                        province_map_df, lat="lat", lon="lon", size="count", color="count",
                        hover_name="province",
                        color_continuous_scale=["#fbbf24", "#dc2626"],
                        size_max=55, zoom=5.4,
                        map_style="carto-darkmatter",
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=420,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.map(province_map_df)
            with table_col:
                st.dataframe(
                    province_map_df.sort_values("count", ascending=False)[["province", "count"]].rename(
                        columns={"province": "İl", "count": "Tweet"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

with tab_insights:
    st.subheader("Etiket Prevalansı")
    prevalence_df = _label_prevalence(df_all, df_filtered, metadata, schema)
    _plot_prevalence(prevalence_df)

    chart_col_a, chart_col_b = st.columns([1.1, 1.0], gap="large")
    with chart_col_a:
        st.subheader("Etiket Dağılımı")
        filtered_counts = _label_counts(df_filtered, schema) if not df_filtered.empty else pd.DataFrame(columns=["label", "count"])
        _plot_label_counts(filtered_counts)
    with chart_col_b:
        st.subheader("Zamansal Dağılım")
        _plot_temporal(df_filtered if not df_filtered.empty else df_all)

    if not prevalence_df.empty:
        st.caption("Tablo: full vs filtreli oranlar")
        prevalence_view = prevalence_df.copy()
        prevalence_view["label"] = prevalence_view["label"].map(pretty_label)
        st.dataframe(prevalence_view, width="stretch", hide_index=True)

with tab_tweets:
    st.subheader("Tweet Listesi")
    if df_filtered.empty:
        st.info("Seçili filtrelerle eşleşen tweet yok.")
    else:
        columns_to_show = [
            column
            for column in ["date", "time", "province", "district", "neighborhood", "urgency_score", "tweet_clean"]
            if column in df_filtered.columns
        ]
        predicted_columns = [schema.label_to_pred[label] for label in schema.labels if label in schema.label_to_pred]
        columns_to_show = columns_to_show + predicted_columns
        st.dataframe(df_filtered[columns_to_show].head(500), width="stretch", hide_index=True)

with tab_test:
    _render_tweet_test_tab()

with tab_about:
    st.subheader("Proje Hakkında")
    st.markdown(
        """
        **Afet Yönetimi - Sosyal Medya Tabanlı İhtiyaç Sinyalleri** projesi, 6 Şubat 2023 Kahramanmaraş
        depremi sırasında atılan tweet'lerden ihtiyaç kategorilerini çıkarmayı ve konum bilgisi ile
        birleştirip saatlik sıcak nokta haritası üretmeyi hedefler.

        - **9 ihtiyaç etiketi**: arama_kurtarma, saglik, barinma, gida_su, altyapi, guvenlik, lojistik, psikolojik, bilgi_paylasimi
        - **Konum çıkarımı**: il / ilçe / mahalle seviyesinde
        - **Urgency skoru**: tweet önceliğini sayısallaştıran kompozit metrik
        - **Canonical v2 final**: en güncel eğitim deneyinin tahmin çıktısı
        - **Tweet Test sekmesi**: kendi yazdığınız cümleyi modele anlık gönderip
          per-label olasılık ve CV-tuned eşik tahminlerini görebilirsiniz
          (yan repo `afetYonetimi_colab` veya `AFETYONETIMI_MODEL_DIR` ile beslenir).
        """
    )

    qr_a, qr_b = st.columns(2)
    with qr_a:
        _render_qr_card("Dashboard repo (UI)", REPO_DASHBOARD)
    with qr_b:
        _render_qr_card("Ana model repo (eğitim & analiz)", REPO_MAIN)

    st.markdown("---")
    _render_provenance_panel(
        csv_path_input,
        str(resolved_meta_path) if resolved_meta_path else None,
        metadata,
        df_all,
        source_kind,
    )
    _render_schema_panel(metadata, schema, prevalence_df if "prevalence_df" in locals() else _label_prevalence(df_all, df_filtered, metadata, schema))
