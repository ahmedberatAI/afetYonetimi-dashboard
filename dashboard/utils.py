from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


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

CANONICAL_CSV_NAME = "need_predictions_geolocated_v2_final.csv"
CANONICAL_META_NAME = "need_predictions_geolocated_v2_final.meta.json"
HISTORICAL_CSV_NAME = "need_predictions_geolocated_63k.csv"
UNKNOWN_LOCATION_VALUES = {"", "na", "n/a", "nan", "none", "null", "unknown", "<na>"}
MOJIBAKE_TOKENS = ("\u00c3", "\u00c4", "\u00c5", "\u00e2", "\ufffd")

PROVINCE_CENTROID = {
    "Hatay": (36.2022, 36.1606),
    "Adana": (37.0000, 35.3213),
    "Ad\u0131yaman": (37.7648, 38.2786),
    "Kahramanmara\u015f": (37.5753, 36.9228),
    "Gaziantep": (37.0662, 37.3833),
    "Diyarbak\u0131r": (37.9144, 40.2306),
    "Malatya": (38.3552, 38.3095),
    "Osmaniye": (37.0742, 36.2478),
    "\u015eanl\u0131urfa": (37.1591, 38.7969),
    "Kilis": (36.7161, 37.1150),
}

PROVINCE_FIXUPS = {
    "Adiyaman": "Ad\u0131yaman",
    "Ad\u0131yaman": "Ad\u0131yaman",
    "Kahramanmaras": "Kahramanmara\u015f",
    "Kahramanmara\u015f": "Kahramanmara\u015f",
    "Diyarbakir": "Diyarbak\u0131r",
    "Diyarbak\u0131r": "Diyarbak\u0131r",
    "Sanliurfa": "\u015eanl\u0131urfa",
    "\u015eanl\u0131urfa": "\u015eanl\u0131urfa",
}


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


def dashboard_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sibling_model_repo_root() -> Path:
    return dashboard_repo_root().parent / "afetYonetimi_colab"


def infer_meta_path(csv_path: str | Path | None) -> Path | None:
    if not csv_path:
        return None
    path = Path(csv_path).expanduser()
    if path.suffix.lower() == ".csv":
        return path.with_name(f"{path.stem}.meta.json")
    return path.with_name(f"{path.name}.meta.json")


def discover_default_source() -> SourceDescriptor:
    repo_root = dashboard_repo_root()
    local_canonical_csv = repo_root / "data" / "predictions" / CANONICAL_CSV_NAME
    sibling_canonical_csv = sibling_model_repo_root() / "data" / "predictions" / CANONICAL_CSV_NAME
    historical_csv = repo_root / "data" / "predictions" / HISTORICAL_CSV_NAME

    env_csv = os.getenv("AFETYONETIMI_CANONICAL_PREDICTIONS_CSV") or os.getenv("AFETYONETIMI_PREDICTIONS_CSV")
    env_meta = os.getenv("AFETYONETIMI_CANONICAL_PREDICTIONS_META") or os.getenv("AFETYONETIMI_PREDICTIONS_META")

    candidates: list[SourceDescriptor] = []
    if env_csv:
        env_csv_path = Path(env_csv).expanduser()
        env_meta_path = Path(env_meta).expanduser() if env_meta else infer_meta_path(env_csv_path)
        candidates.append(
            SourceDescriptor(
                csv_path=env_csv_path,
                meta_path=env_meta_path,
                default_kind="custom",
                label="Environment override",
                note="Using AFETYONETIMI_* environment variables.",
            )
        )

    candidates.extend(
        [
            SourceDescriptor(
                csv_path=local_canonical_csv,
                meta_path=infer_meta_path(local_canonical_csv),
                default_kind="canonical_final",
                label="Canonical final (dashboard-local copy)",
                note="Repo-local canonical CSV/meta pair.",
            ),
            SourceDescriptor(
                csv_path=sibling_canonical_csv,
                meta_path=infer_meta_path(sibling_canonical_csv),
                default_kind="canonical_final",
                label="Canonical final (sibling project repo)",
                note="Auto-detected from the sibling modeling repo.",
            ),
            SourceDescriptor(
                csv_path=historical_csv,
                meta_path=infer_meta_path(historical_csv),
                default_kind="historical",
                label="Historical 63k preview",
                note="Fallback only; superseded by canonical v2 final output.",
            ),
        ]
    )

    for candidate in candidates:
        if candidate.csv_path.exists():
            return candidate

    return candidates[0]


def classify_prediction_source(csv_path: str | Path, metadata: dict[str, Any] | None) -> str:
    name = Path(csv_path).name
    if metadata and bool(metadata.get("canonical")):
        return "canonical_final"
    if name == HISTORICAL_CSV_NAME:
        return "historical"
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
        return "Canonical final output"
    if source_kind == "canonical_candidate":
        return "Canonical-named file (metadata missing)"
    if source_kind == "historical":
        return "Historical preview artifact"
    return "Custom prediction file"


def source_kind_note(source_kind: str) -> str:
    if source_kind == "canonical_final":
        return "Canonical leak-free final prediction output is active."
    if source_kind == "canonical_candidate":
        return "File name matches the canonical artifact, but metadata is missing so provenance cannot be fully verified."
    if source_kind == "historical":
        return "Historical preview is active. It is preserved for comparison, not as the canonical final output."
    return "Custom CSV/meta pair is active. Provenance depends on the selected files."


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


def format_path(path: str | Path | None) -> str:
    if not path:
        return "n/a"
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path)


def format_generated_at(value: Any) -> str:
    if value in (None, ""):
        return "n/a"
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return str(value)
    try:
        ts = ts.tz_convert("Europe/Istanbul")
    except Exception:
        pass
    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")


def pretty_label(label: str) -> str:
    return label.replace("_", " ")


def canonical_limitations(metadata: dict[str, Any] | None) -> list[str]:
    overlap_note = None
    if metadata and isinstance(metadata.get("content_overlap_audit_artifact"), dict):
        overlap_note = metadata["content_overlap_audit_artifact"].get("note")

    return [
        "`guvenlik` ve `bilgi_paylasimi` canonical winner'in acik zayif etiketleri; mevcut CV threshold'lari recall'u tutucu tutuyor.",
        "Rare labels (`altyapi`, `psikolojik`, `saglik`, `guvenlik`) cok kucuk test destegiyle olculdugu icin skorlar hizla sature olabilir.",
        overlap_note
        or "Id-level leak kapatildi, ancak normalized-text overlap kaynakli residual risk hala dokumante edilmis durumda.",
    ]


def _read_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str)


@st.cache_data(show_spinner=False)
def load_predictions_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser()
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

    if "pred_label_count" in df.columns:
        df["pred_label_count"] = pd.to_numeric(df["pred_label_count"], errors="coerce").fillna(0).astype(int)
    elif pred_columns:
        df["pred_label_count"] = df[pred_columns].sum(axis=1).astype(int)

    if "pred_any_need" in df.columns:
        df["pred_any_need"] = pd.to_numeric(df["pred_any_need"], errors="coerce").fillna(0).astype(int)
    elif "pred_label_count" in df.columns:
        df["pred_any_need"] = (df["pred_label_count"] > 0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_prediction_metadata(path_str: str | None) -> dict[str, Any] | None:
    if not path_str:
        return None

    path = Path(path_str).expanduser()
    if not path.exists():
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
    prediction_columns = [label_to_pred[label] for label in labels if label in label_to_pred]
    probability_columns = [label_to_prob[label] for label in labels if label in label_to_prob]

    return PredictionSchema(
        labels=labels,
        label_to_pred=label_to_pred,
        label_to_prob=label_to_prob,
        prediction_columns=prediction_columns,
        probability_columns=probability_columns,
    )
