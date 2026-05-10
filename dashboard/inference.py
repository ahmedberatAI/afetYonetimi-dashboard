"""Single-tweet inference helper for the dashboard's "Tweet Test" tab.

This module loads the canonical leak-free need-classification model
(`exp3_silver_then_gold_v3_exgold`) and produces per-label probabilities and
threshold-applied predictions for a user-supplied text.

Two delivery modes are supported so the tab works both locally and on
Streamlit Community Cloud:

1. **Local model directory** - canonical artefacts on disk
   (`models/exp3_silver_then_gold_v3_exgold/final` in the sibling repo
   `afetYonetimi_colab`, env override, or a copy inside the dashboard repo).
2. **HuggingFace Hub repo** - `from_pretrained("user/repo")` downloads on
   first use and caches on the runtime disk. This is the only option that
   works on Streamlit Cloud where the 442 MB checkpoint cannot be in git.

Heavy dependencies (`torch`, `transformers`) are imported lazily so the rest
of the dashboard keeps working when this tab is not used or when those
libraries are not installed.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
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

# Bundled meta inside the dashboard repo - tiny JSONs, always available.
BUNDLED_LABELS_REL = Path("data") / "model_meta" / "label_columns.json"
BUNDLED_THRESHOLDS_REL = Path("data") / "model_meta" / "thresholds_cv.json"

ENV_MODEL_DIR = "AFETYONETIMI_MODEL_DIR"
ENV_LABELS_JSON = "AFETYONETIMI_LABELS_JSON"
ENV_THRESHOLDS_JSON = "AFETYONETIMI_THRESHOLDS_JSON"
ENV_HF_REPO = "AFETYONETIMI_MODEL_HF_REPO"
ENV_HF_REVISION = "AFETYONETIMI_MODEL_HF_REVISION"
ENV_HF_TOKEN = "AFETYONETIMI_HF_TOKEN"  # for private HF repos

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*$")


def _streamlit_secret(name: str) -> str | None:
    """Best-effort `st.secrets[name]` lookup; never raises if streamlit/secrets missing."""
    try:
        import streamlit as st  # type: ignore

        try:
            value = st.secrets[name]  # type: ignore[index]
        except Exception:  # noqa: BLE001 - secrets file may not exist
            return None
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    except Exception:  # noqa: BLE001 - streamlit not importable in some contexts
        return None


def _resolve_setting(env_name: str) -> str | None:
    raw = os.environ.get(env_name)
    if raw and raw.strip():
        return raw.strip()
    return _streamlit_secret(env_name)


def looks_like_hf_repo_id(value: str) -> bool:
    """Heuristic: 'user/repo'-shaped without any path separator characters."""
    if not value:
        return False
    if "\\" in value or value.startswith("."):
        return False
    if Path(value).expanduser().exists():
        return False
    return bool(_HF_REPO_RE.match(value.strip()))


@dataclass(frozen=True)
class ModelLocation:
    model_ref: str  # local path string OR HF repo id (e.g. "user/repo")
    is_hf_repo: bool
    labels_path: Path
    thresholds_path: Path
    source_label: str
    note: str
    revision: str | None = None

    def model_dir_display(self) -> str:
        return f"hf:{self.model_ref}" if self.is_hf_repo else self.model_ref

    def all_exist(self) -> bool:
        if not (self.labels_path.exists() and self.thresholds_path.exists()):
            return False
        if self.is_hf_repo:
            return True  # remote; assumed reachable, validated at load time
        return Path(self.model_ref).exists()


@dataclass
class ModelBundle:
    location: ModelLocation
    labels: list[str]
    thresholds: dict[str, float]
    tokenizer: Any
    model: Any
    device: str
    max_length: int = 192
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    text: str
    labels: list[str]
    probs: np.ndarray
    thresholds: np.ndarray
    predicted: list[str]
    model_source: str


def _bundled_meta() -> tuple[Path, Path]:
    repo = dashboard_repo_root()
    return repo / BUNDLED_LABELS_REL, repo / BUNDLED_THRESHOLDS_REL


def _candidate_locations() -> list[ModelLocation]:
    cands: list[ModelLocation] = []
    bundled_labels, bundled_thr = _bundled_meta()

    # 1) HF Hub repo via env / Streamlit secrets - works on Cloud.
    hf_repo = _resolve_setting(ENV_HF_REPO)
    if hf_repo and looks_like_hf_repo_id(hf_repo):
        cands.append(
            ModelLocation(
                model_ref=hf_repo,
                is_hf_repo=True,
                labels_path=bundled_labels,
                thresholds_path=bundled_thr,
                source_label=f"HuggingFace Hub ({hf_repo})",
                note=f"`{ENV_HF_REPO}` ile çözülen HF repo. Etiket+eşik repo içindeki bundle'dan.",
                revision=_resolve_setting(ENV_HF_REVISION),
            )
        )

    # 2) Local env override.
    env_model = _resolve_setting(ENV_MODEL_DIR)
    if env_model:
        if looks_like_hf_repo_id(env_model):
            cands.append(
                ModelLocation(
                    model_ref=env_model,
                    is_hf_repo=True,
                    labels_path=bundled_labels,
                    thresholds_path=bundled_thr,
                    source_label=f"HuggingFace Hub ({env_model})",
                    note=f"`{ENV_MODEL_DIR}` HF repo formatında algılandı.",
                    revision=_resolve_setting(ENV_HF_REVISION),
                )
            )
        else:
            model_dir = Path(env_model).expanduser()
            env_labels = _resolve_setting(ENV_LABELS_JSON)
            env_thr = _resolve_setting(ENV_THRESHOLDS_JSON)
            labels = (
                Path(env_labels).expanduser()
                if env_labels
                else (model_dir.parent / "label_columns.json")
            )
            thr = (
                Path(env_thr).expanduser()
                if env_thr
                else (model_dir.parent / "thresholds_cv.json")
            )
            cands.append(
                ModelLocation(
                    model_ref=str(model_dir),
                    is_hf_repo=False,
                    labels_path=labels,
                    thresholds_path=thr,
                    source_label="env override",
                    note=f"`{ENV_MODEL_DIR}` ortam değişkeninden çözüldü.",
                )
            )

    # 3) Sibling modeling repo (canonical layout used during local dev).
    sibling = sibling_model_repo_root()
    cands.append(
        ModelLocation(
            model_ref=str(sibling / CANONICAL_MODEL_REL),
            is_hf_repo=False,
            labels_path=sibling / CANONICAL_LABELS_REL,
            thresholds_path=sibling / CANONICAL_THRESHOLDS_REL,
            source_label="sibling repo (afetYonetimi_colab)",
            note="Yan repo `afetYonetimi_colab` içindeki canonical winner artefaktları.",
        )
    )

    # 4) Sibling modeling repo via selection.json (more authoritative).
    selection_path = sibling / SELECTION_REL
    if selection_path.exists():
        try:
            sel = json.loads(selection_path.read_text(encoding="utf-8"))
            sel_model = sibling / sel["model_dir"]
            sel_labels = sibling / sel["label_columns_json"]
            sel_thr = sibling / sel["thresholds_json"]
            cands.append(
                ModelLocation(
                    model_ref=str(sel_model),
                    is_hf_repo=False,
                    labels_path=sel_labels,
                    thresholds_path=sel_thr,
                    source_label="sibling repo selection.json",
                    note=f"`{selection_path.name}` -> {sel.get('selected_experiment_key', '?')}",
                )
            )
        except Exception:  # noqa: BLE001
            pass

    # 5) Repo-local fallback (model copied into dashboard repo).
    repo_root = dashboard_repo_root()
    cands.append(
        ModelLocation(
            model_ref=str(repo_root / CANONICAL_MODEL_REL),
            is_hf_repo=False,
            labels_path=repo_root / CANONICAL_LABELS_REL,
            thresholds_path=repo_root / CANONICAL_THRESHOLDS_REL,
            source_label="dashboard repo (local copy)",
            note="Dashboard repo içine kopyalanmış model artefaktları.",
        )
    )
    return cands


def discover_model_location() -> ModelLocation | None:
    """First candidate whose paths all exist (or HF Hub configured)."""
    for cand in _candidate_locations():
        if cand.all_exist():
            return cand
    return None


def describe_candidates() -> list[dict[str, Any]]:
    return [
        {
            "source": c.source_label,
            "note": c.note,
            "model_dir": c.model_dir_display(),
            "labels": str(c.labels_path),
            "thresholds": str(c.thresholds_path),
            "exists": c.all_exist(),
            "is_hf": c.is_hf_repo,
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


def make_user_location(
    model_ref: str,
    labels_path: str | Path | None,
    thresholds_path: str | Path | None,
) -> ModelLocation:
    """Build a `ModelLocation` from a UI-supplied path or HF repo id.

    Falls back to the bundled labels/thresholds JSONs when the user does not
    override them (typical when pointing at an HF repo that ships only the
    model + tokenizer files).
    """
    bundled_labels, bundled_thr = _bundled_meta()
    labels_resolved = Path(labels_path).expanduser() if labels_path else bundled_labels
    thr_resolved = Path(thresholds_path).expanduser() if thresholds_path else bundled_thr

    is_hf = looks_like_hf_repo_id(model_ref)
    if is_hf:
        return ModelLocation(
            model_ref=model_ref.strip(),
            is_hf_repo=True,
            labels_path=labels_resolved,
            thresholds_path=thr_resolved,
            source_label=f"HuggingFace Hub ({model_ref.strip()})",
            note="UI üzerinden girilmiş HF repo id.",
            revision=_resolve_setting(ENV_HF_REVISION),
        )
    return ModelLocation(
        model_ref=str(Path(model_ref).expanduser()),
        is_hf_repo=False,
        labels_path=labels_resolved,
        thresholds_path=thr_resolved,
        source_label="user-supplied path",
        note="UI üzerinden girilmiş lokal yol.",
    )


def load_bundle(location: ModelLocation, *, max_length: int = 192, prefer_cpu: bool = False) -> ModelBundle:
    if not (location.labels_path.exists() and location.thresholds_path.exists()):
        raise FileNotFoundError(
            "Etiket / eşik dosyaları eksik: "
            f"labels={location.labels_path}, thresholds={location.thresholds_path}"
        )
    if not location.is_hf_repo and not Path(location.model_ref).exists():
        raise FileNotFoundError(
            "Model dizini bulunamadı: "
            f"{location.model_ref}\n"
            f"Lokal yol yerine HF Hub kullanmak için '{ENV_HF_REPO}' ortam değişkenini "
            "veya UI'daki model_ref alanını 'kullanici/repo' formatında girin."
        )

    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Tweet Test sekmesi için `torch` ve `transformers` kurulu olmalı. "
            "Lokal: `pip install torch transformers`. Streamlit Cloud için "
            "`requirements.txt`'ye eklendiğinden emin ol."
        ) from e

    labels_data = json.loads(location.labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels_data, list) or not labels_data:
        raise ValueError(f"Etiket listesi boş/bozuk: {location.labels_path}")
    labels = [str(x) for x in labels_data]

    thr_data = json.loads(location.thresholds_path.read_text(encoding="utf-8"))
    if not isinstance(thr_data, dict):
        raise ValueError(f"Thresholds JSON dict olmalı: {location.thresholds_path}")
    thresholds: dict[str, float] = {}
    for lab in labels:
        if lab not in thr_data:
            raise ValueError(f"Thresholds'de eksik etiket: {lab}")
        thresholds[lab] = float(thr_data[lab])

    use_cuda = (not prefer_cpu) and bool(getattr(torch.cuda, "is_available", lambda: False)())
    device = "cuda" if use_cuda else "cpu"

    from_pretrained_kwargs: dict[str, Any] = {}
    if location.is_hf_repo and location.revision:
        from_pretrained_kwargs["revision"] = location.revision
    hf_token = _resolve_setting(ENV_HF_TOKEN) if location.is_hf_repo else None
    if hf_token:
        from_pretrained_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(location.model_ref, use_fast=True, **from_pretrained_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(location.model_ref, **from_pretrained_kwargs)
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
        extras={"hf_token_used": bool(hf_token)},
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
