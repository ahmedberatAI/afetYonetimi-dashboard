"""Single-tweet inference helper for the dashboard's "Tweet Test" tab.

This module loads the canonical leak-free need-classification model
(`exp3_silver_then_gold_v3_exgold`) and produces per-label probabilities and
threshold-applied predictions for a user-supplied text.

Heavy dependencies (`torch`, `transformers`) are imported lazily so the rest
of the dashboard keeps working when this tab is not used (e.g. on Streamlit
Community Cloud where the model + libs would exceed the memory budget).
"""
from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from dashboard.utils import dashboard_repo_root, sibling_model_repo_root
except ModuleNotFoundError:  # pragma: no cover - local script context
    from utils import dashboard_repo_root, sibling_model_repo_root  # type: ignore[no-redef]


CANONICAL_MODEL_REL = Path("models") / "exp3_silver_then_gold_v3_exgold" / "final"
CANONICAL_LABELS_REL = Path("models") / "exp3_silver_then_gold_v3_exgold" / "label_columns.json"
CANONICAL_THRESHOLDS_REL = Path("models") / "exp3_silver_then_gold_v3_exgold" / "thresholds_cv.json"
SELECTION_REL = Path("models") / "final" / "selection.json"

ENV_MODEL_DIR = "AFETYONETIMI_MODEL_DIR"
ENV_LABELS_JSON = "AFETYONETIMI_LABELS_JSON"
ENV_THRESHOLDS_JSON = "AFETYONETIMI_THRESHOLDS_JSON"


@dataclass(frozen=True)
class ModelLocation:
    model_dir: Path
    labels_path: Path
    thresholds_path: Path
    source_label: str
    note: str

    def all_exist(self) -> bool:
        return self.model_dir.exists() and self.labels_path.exists() and self.thresholds_path.exists()


@dataclass
class ModelBundle:
    location: ModelLocation
    labels: list[str]
    thresholds: dict[str, float]
    tokenizer: Any
    model: Any
    device: str
    max_length: int = 192


@dataclass(frozen=True)
class PredictionResult:
    text: str
    labels: list[str]
    probs: np.ndarray
    thresholds: np.ndarray
    predicted: list[str]
    model_source: str


def _candidate_locations() -> list[ModelLocation]:
    cands: list[ModelLocation] = []

    env_model = os.environ.get(ENV_MODEL_DIR)
    env_labels = os.environ.get(ENV_LABELS_JSON)
    env_thr = os.environ.get(ENV_THRESHOLDS_JSON)
    if env_model:
        model_dir = Path(env_model).expanduser()
        labels = Path(env_labels).expanduser() if env_labels else (model_dir.parent / "label_columns.json")
        thr = Path(env_thr).expanduser() if env_thr else (model_dir.parent / "thresholds_cv.json")
        cands.append(
            ModelLocation(
                model_dir=model_dir,
                labels_path=labels,
                thresholds_path=thr,
                source_label="env override",
                note=f"`{ENV_MODEL_DIR}` ortam degiskeninden cozuldu.",
            )
        )

    # Sibling modeling repo (canonical layout used during local dev).
    sibling = sibling_model_repo_root()
    cands.append(
        ModelLocation(
            model_dir=sibling / CANONICAL_MODEL_REL,
            labels_path=sibling / CANONICAL_LABELS_REL,
            thresholds_path=sibling / CANONICAL_THRESHOLDS_REL,
            source_label="sibling repo (afetYonetimi_colab)",
            note="Yan repo `afetYonetimi_colab` icindeki canonical winner artefaktlari.",
        )
    )

    # Sibling modeling repo via selection.json (more authoritative).
    selection_path = sibling / SELECTION_REL
    if selection_path.exists():
        try:
            sel = json.loads(selection_path.read_text(encoding="utf-8"))
            sel_model = sibling / sel["model_dir"]
            sel_labels = sibling / sel["label_columns_json"]
            sel_thr = sibling / sel["thresholds_json"]
            cands.append(
                ModelLocation(
                    model_dir=sel_model,
                    labels_path=sel_labels,
                    thresholds_path=sel_thr,
                    source_label="sibling repo selection.json",
                    note=f"`{selection_path.name}` -> {sel.get('selected_experiment_key', '?')}",
                )
            )
        except Exception:  # noqa: BLE001
            pass

    # Repo-local fallback (if user has copied model into dashboard repo).
    repo_root = dashboard_repo_root()
    cands.append(
        ModelLocation(
            model_dir=repo_root / CANONICAL_MODEL_REL,
            labels_path=repo_root / CANONICAL_LABELS_REL,
            thresholds_path=repo_root / CANONICAL_THRESHOLDS_REL,
            source_label="dashboard repo (local copy)",
            note="Dashboard repo icine kopyalanmis model artefaktlari.",
        )
    )
    return cands


def discover_model_location() -> ModelLocation | None:
    """First candidate whose paths all exist."""
    for cand in _candidate_locations():
        if cand.all_exist():
            return cand
    return None


def describe_candidates() -> list[dict[str, Any]]:
    return [
        {
            "source": c.source_label,
            "note": c.note,
            "model_dir": str(c.model_dir),
            "labels": str(c.labels_path),
            "thresholds": str(c.thresholds_path),
            "exists": c.all_exist(),
        }
        for c in _candidate_locations()
    ]


def clean_tweet_text(text: str) -> str:
    """Mirror of `preprocess_emergency_data.clean_text` (NFC + whitespace)."""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())
    return s.strip()


def load_bundle(location: ModelLocation, *, max_length: int = 192, prefer_cpu: bool = False) -> ModelBundle:
    if not location.all_exist():
        raise FileNotFoundError(
            "Model artefaktlari eksik: "
            f"model_dir={location.model_dir}, labels={location.labels_path}, "
            f"thresholds={location.thresholds_path}"
        )

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Tweet Test sekmesi icin `torch` ve `transformers` kurulu olmali. "
            "Lokal: `pip install torch transformers`."
        ) from e

    labels_data = json.loads(location.labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels_data, list) or not labels_data:
        raise ValueError(f"Etiket listesi bos/bozuk: {location.labels_path}")
    labels = [str(x) for x in labels_data]

    thr_data = json.loads(location.thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(thr_data, dict):
        raise ValueError(f"Thresholds JSON dict olmali: {location.thresholds_path}")
    thresholds: dict[str, float] = {}
    for lab in labels:
        if lab not in thr_data:
            raise ValueError(f"Thresholds'de eksik etiket: {lab}")
        thresholds[lab] = float(thr_data[lab])

    use_cuda = (not prefer_cpu) and bool(getattr(torch.cuda, "is_available", lambda: False)())
    device = "cuda" if use_cuda else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(location.model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(location.model_dir))
    model.eval()
    model.to(device)

    return ModelBundle(
        location=location,
        labels=labels,
        thresholds=thresholds,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=int(max_length),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def predict_one(bundle: ModelBundle, text: str, *, apply_clean: bool = True) -> PredictionResult:
    import torch  # type: ignore  # already imported by load_bundle if we got here

    cleaned = clean_tweet_text(text) if apply_clean else (text or "")
    enc = bundle.tokenizer(
        [cleaned],
        truncation=True,
        max_length=bundle.max_length,
        padding=True,
        return_tensors="pt",
    )
    enc = {k: v.to(bundle.device) for k, v in enc.items()}
    with torch.no_grad():
        outp = bundle.model(**enc)
    logits = outp.logits.detach().cpu().numpy()
    probs = _sigmoid(logits)[0].astype(np.float32)
    thr_vec = np.array([bundle.thresholds[lab] for lab in bundle.labels], dtype=np.float32)
    predicted = [lab for lab, p, t in zip(bundle.labels, probs, thr_vec) if p >= t]
    return PredictionResult(
        text=cleaned,
        labels=list(bundle.labels),
        probs=probs,
        thresholds=thr_vec,
        predicted=predicted,
        model_source=bundle.location.source_label,
    )


def predict_many(bundle: ModelBundle, texts: Iterable[str], *, apply_clean: bool = True) -> list[PredictionResult]:
    return [predict_one(bundle, t, apply_clean=apply_clean) for t in texts]
