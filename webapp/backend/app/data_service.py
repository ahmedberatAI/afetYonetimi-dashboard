from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_CSV_NAME = "need_predictions_geolocated_v2_final.csv"
CANONICAL_META_NAME = "need_predictions_geolocated_v2_final.meta.json"

DEFAULT_LABELS = [
    "arama_kurtarma",
    "saglik",
    "barinma",
    "gida_su",
    "altyapi",
    "guvenlik",
    "lojistik",
    "psikolojik",
    "bilgi_paylasimi",
]

LABEL_DISPLAY = {
    "arama_kurtarma": "arama kurtarma",
    "saglik": "sağlık",
    "barinma": "barınma",
    "gida_su": "gıda su",
    "altyapi": "altyapı",
    "guvenlik": "güvenlik",
    "lojistik": "lojistik",
    "psikolojik": "psikolojik",
    "bilgi_paylasimi": "bilgi paylaşımı",
}

PROVINCE_CENTROID = {
    "Hatay": (36.2022, 36.1606),
    "Adana": (37.0000, 35.3213),
    "Adıyaman": (37.7648, 38.2786),
    "Kahramanmaraş": (37.5753, 36.9228),
    "Gaziantep": (37.0662, 37.3833),
    "Diyarbakır": (37.9144, 40.2306),
    "Malatya": (38.3552, 38.3095),
    "Osmaniye": (37.0742, 36.2478),
    "Şanlıurfa": (37.1591, 38.7969),
    "Kilis": (36.7161, 37.1150),
}

PROVINCE_FIXUPS = {
    "Adiyaman": "Adıyaman",
    "Adıyaman": "Adıyaman",
    "Kahramanmaras": "Kahramanmaraş",
    "Kahramanmaraş": "Kahramanmaraş",
    "Diyarbakir": "Diyarbakır",
    "Diyarbakır": "Diyarbakır",
    "Sanliurfa": "Şanlıurfa",
    "Şanlıurfa": "Şanlıurfa",
}

UNKNOWN_LOCATION_VALUES = {"", "na", "n/a", "nan", "none", "null", "unknown", "<na>"}
MOJIBAKE_TOKENS = ("\u00c3", "\u00c4", "\u00c5", "\u00e2", "\ufffd")
INFO_POSTPROCESS_MIN_PROB = 0.20
INFO_MISSING_RE = re.compile(
    r"(haber\s+alam|haber\s+al[ıi]nam|ula[şs]am[ıi]yor|ula[şs][ıi]lam[ıi]yor)",
    flags=re.IGNORECASE,
)
INFO_REQUEST_RE = re.compile(
    r"(g[oö]ren|duyan|bilen|bilgisi\s+olan|bilgi\s+alan|haber\s+alan|ula[şs]s[ıi]n|yazs[ıi]n|bildirsin)",
    flags=re.IGNORECASE,
)
INFO_CONTACT_RE = re.compile(r"(ileti[şs]im|irtibat|telefon|numara|0\d{10}|05\d{9})", flags=re.IGNORECASE)
INFO_ANNOUNCEMENT_RE = re.compile(
    r"(duyuru|canl[ıi]\s+yay[ıi]n|transfer|da[ğg][ıi]t[ıi]m|ula[şs]t[ıi]r[ıi]ld[ıi]|bildirilsin)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceDescriptor:
    csv_path: Path
    meta_path: Path | None
    default_kind: str
    label: str
    note: str


@dataclass(frozen=True)
class PredictionSchema:
    labels: list[str]
    label_to_pred: dict[str, str]
    label_to_prob: dict[str, str]
    prediction_columns: list[str]
    probability_columns: list[str]


@dataclass(frozen=True)
class DatasetBundle:
    df: pd.DataFrame
    metadata: dict[str, Any] | None
    schema: PredictionSchema
    source: SourceDescriptor
    source_kind: str


@dataclass(frozen=True)
class Filters:
    start_date: date | None = None
    end_date: date | None = None
    provinces: tuple[str, ...] = ()
    districts: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    label_mode: str = "ANY"
    urgency_min: float | None = None
    search: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sibling_model_repo_root() -> Path:
    return repo_root().parent / "afetYonetimi_colab"


def infer_meta_path(csv_path: str | Path | None) -> Path | None:
    if not csv_path:
        return None
    path = Path(csv_path).expanduser()
    if path.suffix.lower() == ".csv":
        return path.with_name(f"{path.stem}.meta.json")
    return path.with_name(f"{path.name}.meta.json")


def discover_default_source() -> SourceDescriptor:
    root = repo_root()
    local_canonical_csv = root / "data" / "predictions" / CANONICAL_CSV_NAME
    sibling_canonical_csv = sibling_model_repo_root() / "data" / "predictions" / CANONICAL_CSV_NAME

    env_csv = os.getenv("AFETYONETIMI_CANONICAL_PREDICTIONS_CSV") or os.getenv("AFETYONETIMI_PREDICTIONS_CSV")
    env_meta = os.getenv("AFETYONETIMI_CANONICAL_PREDICTIONS_META") or os.getenv("AFETYONETIMI_PREDICTIONS_META")

    candidates: list[SourceDescriptor] = []
    if env_csv:
        env_csv_path = Path(env_csv).expanduser()
        candidates.append(
            SourceDescriptor(
                csv_path=env_csv_path,
                meta_path=(Path(env_meta).expanduser() if env_meta else infer_meta_path(env_csv_path)),
                default_kind="custom",
                label="Environment override",
                note="AFETYONETIMI_* environment variables ile çözüldü.",
            )
        )

    candidates.extend(
        [
            SourceDescriptor(
                csv_path=local_canonical_csv,
                meta_path=infer_meta_path(local_canonical_csv),
                default_kind="canonical_final",
                label="Canonical final (dashboard-local copy)",
                note="Repo-local canonical CSV/meta çifti.",
            ),
            SourceDescriptor(
                csv_path=sibling_canonical_csv,
                meta_path=infer_meta_path(sibling_canonical_csv),
                default_kind="canonical_final",
                label="Canonical final (sibling project repo)",
                note="Yan modelleme reposundan otomatik algılandı.",
            ),
        ]
    )

    for candidate in candidates:
        if candidate.csv_path.exists():
            return candidate
    return candidates[0]


def maybe_fix_mojibake(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = PROVINCE_FIXUPS.get(text, text)
    if any(token in text for token in MOJIBAKE_TOKENS):
        try:
            repaired = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            repaired = text
        text = PROVINCE_FIXUPS.get(repaired, repaired)
    else:
        text = PROVINCE_FIXUPS.get(text, text)
    return text


def normalize_location_value(value: Any) -> str:
    text = maybe_fix_mojibake(value)
    if not text:
        return ""
    if text.casefold() in UNKNOWN_LOCATION_VALUES:
        return ""
    return text


def pretty_label(label: str) -> str:
    return LABEL_DISPLAY.get(label, label.replace("_", " "))


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str)


def _info_postprocess_enabled() -> bool:
    raw = os.getenv("AFETYONETIMI_DISABLE_INFO_POSTPROCESS", "")
    return raw.strip().casefold() not in {"1", "true", "yes", "on"}


def _has_info_postprocess_signal(text: Any) -> bool:
    t = " ".join(str(text or "").casefold().split())
    missing = bool(INFO_MISSING_RE.search(t))
    request = bool(INFO_REQUEST_RE.search(t))
    contact = bool(INFO_CONTACT_RE.search(t))
    announcement = bool(INFO_ANNOUNCEMENT_RE.search(t))
    return (missing and request) or (missing and contact) or (request and contact) or announcement


def _apply_info_v1_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if not _info_postprocess_enabled():
        return df
    required = {"tweet_clean", "prob_bilgi_paylasimi", "pred_bilgi_paylasimi"}
    if not required.issubset(df.columns):
        return df

    prob = pd.to_numeric(df["prob_bilgi_paylasimi"], errors="coerce").fillna(0.0)
    pred = pd.to_numeric(df["pred_bilgi_paylasimi"], errors="coerce").fillna(0).astype(int)
    signal = df["tweet_clean"].map(_has_info_postprocess_signal)
    added = (pred == 0) & (prob >= INFO_POSTPROCESS_MIN_PROB) & signal
    if bool(added.any()):
        df.loc[added, "pred_bilgi_paylasimi"] = 1
    df["postprocess_info_v1_added"] = added.astype(int)
    return df


def load_predictions_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = _read_csv_with_fallbacks(path)

    for column in ["province", "district", "neighborhood"]:
        if column in df.columns:
            df[column] = df[column].map(normalize_location_value)

    if "created_at" in df.columns:
        created_at = df["created_at"].astype("string").fillna("").map(maybe_fix_mojibake)
        ts_utc = pd.to_datetime(created_at, errors="coerce", utc=True)
        try:
            df["created_at_local"] = ts_utc.dt.tz_convert("Europe/Istanbul")
        except Exception:
            df["created_at_local"] = ts_utc
        df["created_at_parsed"] = ts_utc
    else:
        df["created_at_local"] = pd.NaT
        df["created_at_parsed"] = pd.NaT

    if "date" not in df.columns and "created_at_local" in df.columns:
        df["date"] = pd.to_datetime(df["created_at_local"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "time" not in df.columns and "created_at_local" in df.columns:
        df["time"] = pd.to_datetime(df["created_at_local"], errors="coerce").dt.strftime("%H:%M:%S")

    for column in ["date", "time"]:
        if column in df.columns:
            df[column] = df[column].astype("string").fillna("").map(maybe_fix_mojibake).str.strip()

    if "urgency_score" in df.columns:
        df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0.0)

    pred_columns = []
    for column in df.columns:
        if column.startswith("pred_") and column not in {"pred_any_need", "pred_label_count"}:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
            pred_columns.append(column)
        elif column.startswith("prob_"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = _apply_info_v1_postprocess(df)

    if pred_columns:
        df["pred_label_count"] = df[pred_columns].sum(axis=1).astype(int)
    elif "pred_label_count" in df.columns:
        df["pred_label_count"] = pd.to_numeric(df["pred_label_count"], errors="coerce").fillna(0).astype(int)

    if "pred_label_count" in df.columns:
        df["pred_any_need"] = (df["pred_label_count"] > 0).astype(int)
    elif "pred_any_need" in df.columns:
        df["pred_any_need"] = pd.to_numeric(df["pred_any_need"], errors="coerce").fillna(0).astype(int)

    return df


def load_prediction_metadata(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text())


def build_prediction_schema(metadata: dict[str, Any] | None, columns: list[str]) -> PredictionSchema:
    column_set = set(columns)
    meta_labels = [
        str(item)
        for item in (metadata or {}).get("labels", [])
        if isinstance(item, str) and item.strip()
    ]

    labels: list[str] = []
    for label in meta_labels + DEFAULT_LABELS:
        if label not in labels:
            labels.append(label)

    for column in columns:
        if column.startswith("pred_") and column not in {"pred_any_need", "pred_label_count"}:
            label = column.removeprefix("pred_")
            if label not in labels:
                labels.append(label)

    raw_pred = (metadata or {}).get("label_to_pred_column", {})
    raw_prob = (metadata or {}).get("label_to_prob_column", {})
    label_to_pred: dict[str, str] = {}
    label_to_prob: dict[str, str] = {}

    for label in labels:
        pred_column = raw_pred.get(label) if isinstance(raw_pred, dict) else None
        prob_column = raw_prob.get(label) if isinstance(raw_prob, dict) else None

        if isinstance(pred_column, str) and pred_column in column_set:
            label_to_pred[label] = pred_column
        elif f"pred_{label}" in column_set:
            label_to_pred[label] = f"pred_{label}"

        if isinstance(prob_column, str) and prob_column in column_set:
            label_to_prob[label] = prob_column
        elif f"prob_{label}" in column_set:
            label_to_prob[label] = f"prob_{label}"

    labels = [label for label in labels if label in label_to_pred or label in label_to_prob]
    return PredictionSchema(
        labels=labels,
        label_to_pred=label_to_pred,
        label_to_prob=label_to_prob,
        prediction_columns=[label_to_pred[label] for label in labels if label in label_to_pred],
        probability_columns=[label_to_prob[label] for label in labels if label in label_to_prob],
    )


def classify_prediction_source(csv_path: Path, metadata: dict[str, Any] | None) -> str:
    name = csv_path.name
    if metadata and bool(metadata.get("canonical")):
        return "canonical_final"
    if metadata:
        supersedes = {
            Path(item).name
            for item in metadata.get("supersedes", [])
            if isinstance(item, str) and item.strip()
        }
        if name in supersedes:
            return "historical"
    if name == CANONICAL_CSV_NAME:
        return "canonical_candidate"
    return "custom"


def source_kind_label(source_kind: str) -> str:
    if source_kind == "canonical_final":
        return "Canonical final çıktı"
    if source_kind == "canonical_candidate":
        return "Canonical adlı dosya"
    if source_kind == "historical":
        return "Historical preview"
    return "Özel tahmin dosyası"


def source_kind_note(source_kind: str) -> str:
    if source_kind == "canonical_final":
        return "Canonical leak-free final tahmin çıktısı aktif."
    if source_kind == "canonical_candidate":
        return "Dosya adı canonical artifact ile eşleşiyor; metadata eksik olabilir."
    if source_kind == "historical":
        return "Historical preview aktif; canonical final çıktı olarak sunulmaz."
    return "Özel CSV/meta çifti aktif."


def format_generated_at(value: Any) -> str | None:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return str(value)
    try:
        ts = ts.tz_convert("Europe/Istanbul")
    except Exception:
        pass
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


@lru_cache(maxsize=1)
def get_dataset() -> DatasetBundle:
    source = discover_default_source()
    df = load_predictions_csv(source.csv_path)
    metadata = load_prediction_metadata(source.meta_path)
    schema = build_prediction_schema(metadata, df.columns.tolist())
    source_kind = classify_prediction_source(source.csv_path, metadata)
    return DatasetBundle(df=df, metadata=metadata, schema=schema, source=source, source_kind=source_kind)


def clear_caches() -> None:
    get_dataset.cache_clear()
    load_location_index.cache_clear()


@lru_cache(maxsize=1)
def load_location_index() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = repo_root() / "data" / "gazetteer" / "earthquake_region_neighborhoods.csv"
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


def parse_csv_param(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def date_series(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce").dt.date
    return pd.to_datetime(df["created_at_local"], errors="coerce").dt.date


def hour_series(df: pd.DataFrame) -> pd.Series:
    if "created_at_local" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    hours = pd.to_datetime(df["created_at_local"], errors="coerce").dt.floor("h")
    try:
        hours = hours.dt.tz_localize(None)
    except Exception:
        pass
    return hours


def apply_filters(df: pd.DataFrame, schema: PredictionSchema, filters: Filters) -> pd.DataFrame:
    out = df
    dates = date_series(out)
    if filters.start_date:
        out = out[dates >= filters.start_date]
        dates = dates.reindex(out.index)
    if filters.end_date:
        out = out[dates <= filters.end_date]

    if filters.provinces and "province" in out.columns:
        out = out[out["province"].isin(filters.provinces)]
    if filters.districts and "district" in out.columns:
        out = out[out["district"].isin(filters.districts)]

    if filters.urgency_min is not None and "urgency_score" in out.columns:
        urgency = pd.to_numeric(out["urgency_score"], errors="coerce").fillna(0.0)
        out = out[urgency >= float(filters.urgency_min)]

    selected_columns = [
        schema.label_to_pred[label]
        for label in filters.labels
        if label in schema.label_to_pred and schema.label_to_pred[label] in out.columns
    ]
    if selected_columns:
        sums = out[selected_columns].sum(axis=1)
        if filters.label_mode.upper() == "ALL":
            out = out[sums == len(selected_columns)]
        else:
            out = out[sums > 0]

    if filters.search:
        needle = filters.search.strip().casefold()
        if needle:
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
            mask = tweet_clean.str.casefold().str.contains(needle, na=False, regex=False) | tweet_raw.str.casefold().str.contains(
                needle,
                na=False,
                regex=False,
            )
            out = out[mask]

    return out.copy()


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, digits: int = 2) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _non_empty_unique(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    values = df[column].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def build_options(bundle: DatasetBundle) -> dict[str, Any]:
    df = bundle.df
    dates = date_series(df).dropna()
    urgency = pd.to_numeric(df["urgency_score"], errors="coerce").dropna() if "urgency_score" in df.columns else pd.Series(dtype=float)
    source = bundle.source
    metadata = bundle.metadata or {}
    return {
        "source": {
            "label": source.label,
            "kind": bundle.source_kind,
            "kindLabel": source_kind_label(bundle.source_kind),
            "note": source_kind_note(bundle.source_kind),
            "path": str(source.csv_path),
            "metaPath": str(source.meta_path) if source.meta_path else None,
            "generatedAt": format_generated_at(metadata.get("generated_at")),
            "experiment": metadata.get("selected_experiment_key"),
            "thresholdSource": metadata.get("threshold_source"),
            "thresholdType": metadata.get("threshold_type"),
        },
        "dateRange": {
            "min": dates.min().isoformat() if not dates.empty else None,
            "max": dates.max().isoformat() if not dates.empty else None,
        },
        "urgencyRange": {
            "min": _safe_float(urgency.min(), 0) if not urgency.empty else 0,
            "max": _safe_float(urgency.max(), 0) if not urgency.empty else 10,
        },
        "labels": [{"id": label, "name": pretty_label(label)} for label in bundle.schema.labels],
        "provinces": _non_empty_unique(df, "province"),
        "districts": _non_empty_unique(df, "district"),
        "rowCount": int(len(df)),
        "metadata": {
            "rowCount": metadata.get("row_count"),
            "rowsBefore": metadata.get("rows_before"),
            "rowsAfter": metadata.get("rows_after"),
            "duplicateRowsRemoved": metadata.get("duplicate_rows_removed"),
            "canonical": bool(metadata.get("canonical", False)),
            "contentOverlapNote": (metadata.get("content_overlap_audit_artifact") or {}).get("note")
            if isinstance(metadata.get("content_overlap_audit_artifact"), dict)
            else None,
        },
    }


def label_counts(df: pd.DataFrame, schema: PredictionSchema) -> list[dict[str, Any]]:
    rows = []
    for label in schema.labels:
        pred_column = schema.label_to_pred.get(label)
        if pred_column and pred_column in df.columns:
            rows.append({"label": label, "name": pretty_label(label), "count": int(df[pred_column].sum())})
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def prevalence(bundle: DatasetBundle, filtered: pd.DataFrame) -> list[dict[str, Any]]:
    df_all = bundle.df
    metadata = bundle.metadata or {}
    meta_row_count = metadata.get("row_count")
    try:
        full_row_count = int(meta_row_count) if meta_row_count is not None else int(len(df_all))
    except (TypeError, ValueError):
        full_row_count = int(len(df_all))

    meta_pred_positives = metadata.get("pred_positives", {}) if isinstance(metadata.get("pred_positives"), dict) else {}
    rows = []
    for label in bundle.schema.labels:
        pred_column = bundle.schema.label_to_pred.get(label)
        if not pred_column or pred_column not in df_all.columns:
            continue
        full_positive = meta_pred_positives.get(label)
        if full_positive is None:
            full_positive = int(df_all[pred_column].sum())
        filtered_positive = int(filtered[pred_column].sum()) if pred_column in filtered.columns else 0
        rows.append(
            {
                "label": label,
                "name": pretty_label(label),
                "fullPositive": int(full_positive),
                "fullRatePct": _safe_float((float(full_positive) / full_row_count * 100.0) if full_row_count else 0.0),
                "filteredPositive": filtered_positive,
                "filteredRatePct": _safe_float((filtered_positive / len(filtered) * 100.0) if len(filtered) else 0.0),
            }
        )
    return sorted(rows, key=lambda item: item["fullPositive"], reverse=True)


def province_map(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "province" not in df.columns:
        return []
    non_empty = df[df["province"].astype("string").fillna("").str.strip() != ""]
    grouped = non_empty.groupby("province", dropna=False).size().reset_index(name="count")
    rows = []
    for _, row in grouped.iterrows():
        province = str(row["province"])
        if province in PROVINCE_CENTROID:
            lat, lon = PROVINCE_CENTROID[province]
            rows.append({"province": province, "count": int(row["count"]), "lat": lat, "lon": lon})
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def temporal_series(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    dates = date_series(df)
    frame = pd.DataFrame({"date": dates}, index=df.index).dropna()
    if frame.empty:
        return []
    frame["rows"] = 1
    if "pred_any_need" in df.columns:
        frame["needs"] = pd.to_numeric(df["pred_any_need"], errors="coerce").fillna(0).astype(int).reindex(frame.index)
    else:
        frame["needs"] = 0
    if "urgency_score" in df.columns:
        frame["urgency"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0.0).reindex(frame.index)
    else:
        frame["urgency"] = 0.0
    grouped = frame.groupby("date").agg(rows=("rows", "sum"), needs=("needs", "sum"), urgency=("urgency", "mean")).reset_index()
    return [
        {
            "date": item["date"].isoformat() if hasattr(item["date"], "isoformat") else str(item["date"]),
            "rows": int(item["rows"]),
            "needs": int(item["needs"]),
            "urgency": _safe_float(item["urgency"]),
        }
        for _, item in grouped.iterrows()
    ]


def hour_options(df: pd.DataFrame) -> list[str]:
    hours = hour_series(df).dropna()
    if hours.empty:
        return []
    return [pd.Timestamp(value).strftime("%Y-%m-%dT%H:00:00") for value in sorted(pd.DatetimeIndex(hours.unique()))]


def default_hour(df: pd.DataFrame) -> str | None:
    hours = hour_series(df)
    frame = pd.DataFrame({"hour": hours}, index=df.index).dropna()
    if frame.empty:
        return None
    if "pred_any_need" in df.columns:
        positive = pd.to_numeric(df["pred_any_need"], errors="coerce").fillna(0).astype(int) == 1
        pos_frame = frame[positive.reindex(frame.index).fillna(False)]
        if not pos_frame.empty:
            frame = pos_frame
    counts = frame.groupby("hour").size().sort_values(ascending=False)
    if counts.empty:
        return None
    return pd.Timestamp(counts.index[0]).strftime("%Y-%m-%dT%H:00:00")


def row_labels(row: pd.Series, schema: PredictionSchema, max_labels: int | None = None) -> list[str]:
    labels = []
    for label in schema.labels:
        pred_column = schema.label_to_pred.get(label)
        if pred_column and pred_column in row.index:
            try:
                if int(row[pred_column]) == 1:
                    labels.append(label)
            except (TypeError, ValueError):
                continue
        if max_labels is not None and len(labels) >= max_labels:
            break
    return labels


def top_tweets(df: pd.DataFrame, schema: PredictionSchema, limit: int = 60) -> list[dict[str, Any]]:
    if df.empty:
        return []
    samples = df.copy()
    if "urgency_score" in samples.columns:
        samples["__u"] = pd.to_numeric(samples["urgency_score"], errors="coerce").fillna(0.0)
        samples = samples.sort_values(["__u", "pred_label_count"], ascending=[False, False])
    samples = samples.head(max(1, min(limit, 200)))
    rows = []
    for _, row in samples.iterrows():
        labels = row_labels(row, schema)
        rows.append(
            {
                "id": _clean_text(row.get("id")),
                "date": _clean_text(row.get("date")),
                "time": _clean_text(row.get("time")),
                "province": _clean_text(row.get("province")),
                "district": _clean_text(row.get("district")),
                "neighborhood": _clean_text(row.get("neighborhood")),
                "text": _clean_text(row.get("tweet_clean") or row.get("tweet")),
                "urgency": _safe_float(row.get("urgency_score")),
                "labels": [{"id": label, "name": pretty_label(label)} for label in labels],
            }
        )
    return rows


def summary(bundle: DatasetBundle, filtered: pd.DataFrame) -> dict[str, Any]:
    counts = label_counts(filtered, bundle.schema)
    top = counts[0] if counts else {"label": None, "name": "n/a", "count": 0}
    any_need = int(pd.to_numeric(filtered["pred_any_need"], errors="coerce").fillna(0).sum()) if "pred_any_need" in filtered.columns else 0
    urgency = (
        pd.to_numeric(filtered["urgency_score"], errors="coerce").fillna(0.0)
        if "urgency_score" in filtered.columns and len(filtered)
        else pd.Series(dtype=float)
    )
    provinces = _non_empty_unique(filtered, "province")
    return {
        "totalRows": int(len(bundle.df)),
        "filteredRows": int(len(filtered)),
        "needSignals": any_need,
        "needSignalRatePct": _safe_float((any_need / len(filtered) * 100.0) if len(filtered) else 0.0),
        "provinceCount": len(provinces),
        "avgUrgency": _safe_float(urgency.mean()) if not urgency.empty else 0.0,
        "maxUrgency": _safe_float(urgency.max()) if not urgency.empty else 0.0,
        "topLabel": top,
    }


def build_overview(bundle: DatasetBundle, filters: Filters) -> dict[str, Any]:
    filtered = apply_filters(bundle.df, bundle.schema, filters)
    hours = hour_options(filtered)
    return {
        "summary": summary(bundle, filtered),
        "labelCounts": label_counts(filtered, bundle.schema),
        "prevalence": prevalence(bundle, filtered),
        "temporal": temporal_series(filtered),
        "provinceMap": province_map(filtered),
        "tweets": top_tweets(filtered, bundle.schema),
        "hours": hours,
        "defaultHour": default_hour(filtered),
        "source": build_options(bundle)["source"],
    }


def _aggregate_hotspots(df: pd.DataFrame, group_cols: list[str], signal_mode: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["signal"])

    frame = df
    if signal_mode == "count_any_need" and "pred_any_need" in frame.columns:
        frame = frame[pd.to_numeric(frame["pred_any_need"], errors="coerce").fillna(0).astype(int) == 1]
    if frame.empty:
        return pd.DataFrame(columns=group_cols + ["signal"])

    if signal_mode == "sum_urgency" and "urgency_score" in frame.columns:
        return frame.groupby(group_cols, dropna=False)["urgency_score"].sum().reset_index(name="signal")
    return frame.groupby(group_cols, dropna=False).size().reset_index(name="signal")


def _attach_coordinates(aggregated: pd.DataFrame, level: str) -> pd.DataFrame:
    neigh_ix, dist_ix, prov_ix = load_location_index()
    out = aggregated.copy()

    if level == "neighborhood":
        out["neighborhood_clean"] = out["neighborhood"].astype("string").fillna("").str.strip().str.lower()
        out = out[out["neighborhood_clean"] != ""].reset_index(drop=True)
        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        out = out.merge(neigh_ix, on=["province", "district", "neighborhood_clean"], how="left")
        out = out.merge(dist_ix, on=["province", "district"], how="left")
        out = out.merge(prov_ix, on=["province"], how="left")
        out["lat"] = out["lat"].fillna(out.get("lat_dist")).fillna(out.get("lat_prov"))
        out["lon"] = out["lon"].fillna(out.get("lon_dist")).fillna(out.get("lon_prov"))
        out = out.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")
    elif level == "district":
        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        out = out.merge(dist_ix, on=["province", "district"], how="left")
        out = out.merge(prov_ix, on=["province"], how="left")
        out["lat"] = out["lat_dist"].fillna(out.get("lat_prov"))
        out["lon"] = out["lon_dist"].fillna(out.get("lon_prov"))
        out = out.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")
    else:
        out = out.merge(prov_ix, on=["province"], how="left")

    if "province" in out.columns:
        missing_geo = out["lat"].isna() | out["lon"].isna()
        for index, row in out.loc[missing_geo].iterrows():
            province = str(row.get("province", ""))
            if province in PROVINCE_CENTROID:
                lat, lon = PROVINCE_CENTROID[province]
                out.at[index, "lat"] = lat
                out.at[index, "lon"] = lon
    return out


def build_hotspots(bundle: DatasetBundle, filters: Filters, hour: str | None, level: str, signal_mode: str) -> dict[str, Any]:
    filtered = apply_filters(bundle.df, bundle.schema, filters)
    hours = hour_options(filtered)
    selected_hour = hour if hour in hours else (default_hour(filtered) or (hours[0] if hours else None))
    if not selected_hour:
        return {"hour": None, "hours": [], "points": [], "stats": {"totalSignal": 0, "hotspots": 0, "topSignal": 0, "geoCoveragePct": 0}}

    level = level if level in {"province", "district", "neighborhood"} else "province"
    signal_mode = signal_mode if signal_mode in {"count_rows", "count_any_need", "sum_urgency"} else "count_any_need"
    if level == "province":
        group_cols = ["province"]
    elif level == "district":
        group_cols = ["province", "district"]
    else:
        group_cols = ["province", "district", "neighborhood"]

    missing_group = [column for column in group_cols if column not in filtered.columns]
    if missing_group:
        return {"hour": selected_hour, "hours": hours, "points": [], "stats": {"totalSignal": 0, "hotspots": 0, "topSignal": 0, "geoCoveragePct": 0}}

    df_hourly = filtered.copy()
    df_hourly["_hour"] = hour_series(filtered).dt.strftime("%Y-%m-%dT%H:00:00")
    selected = df_hourly[df_hourly["_hour"] == selected_hour]
    aggregated = _aggregate_hotspots(selected, group_cols, signal_mode)
    before_geo = len(aggregated)
    if aggregated.empty:
        return {"hour": selected_hour, "hours": hours, "points": [], "stats": {"totalSignal": 0, "hotspots": 0, "topSignal": 0, "geoCoveragePct": 0}}

    aggregated = _attach_coordinates(aggregated, level)
    aggregated["signal"] = pd.to_numeric(aggregated["signal"], errors="coerce").fillna(0.0)
    aggregated = aggregated.dropna(subset=["lat", "lon"])
    aggregated = aggregated[aggregated["signal"] > 0].sort_values("signal", ascending=False).reset_index(drop=True)
    if aggregated.empty:
        return {"hour": selected_hour, "hours": hours, "points": [], "stats": {"totalSignal": 0, "hotspots": 0, "topSignal": 0, "geoCoveragePct": 0}}

    total_signal = float(aggregated["signal"].sum())
    q50 = float(aggregated["signal"].quantile(0.50))
    q80 = float(aggregated["signal"].quantile(0.80))
    q95 = float(aggregated["signal"].quantile(0.95))

    def severity(value: float) -> str:
        if value >= q95:
            return "Kritik"
        if value >= q80:
            return "Yüksek"
        if value >= q50:
            return "Orta"
        return "İzleme"

    points = []
    for index, row in aggregated.iterrows():
        if level == "province":
            location_label = _clean_text(row.get("province"))
        elif level == "district":
            location_label = " / ".join(part for part in [_clean_text(row.get("province")), _clean_text(row.get("district"))] if part)
        else:
            location_label = " / ".join(
                part
                for part in [_clean_text(row.get("province")), _clean_text(row.get("district")), _clean_text(row.get("neighborhood"))]
                if part
            )
        signal = float(row["signal"])
        points.append(
            {
                "rank": int(index + 1),
                "location": location_label,
                "province": _clean_text(row.get("province")),
                "district": _clean_text(row.get("district")),
                "neighborhood": _clean_text(row.get("neighborhood")),
                "lat": _safe_float(row.get("lat"), 6),
                "lon": _safe_float(row.get("lon"), 6),
                "signal": _safe_float(signal, 1),
                "sharePct": _safe_float((signal / total_signal * 100.0) if total_signal else 0.0),
                "severity": severity(signal),
            }
        )

    top_signal = points[0]["signal"] if points else 0
    return {
        "hour": selected_hour,
        "hours": hours,
        "points": points,
        "stats": {
            "totalSignal": _safe_float(total_signal, 1),
            "hotspots": len(points),
            "topSignal": top_signal,
            "geoCoveragePct": _safe_float((len(aggregated) / before_geo * 100.0) if before_geo else 0.0),
        },
    }
