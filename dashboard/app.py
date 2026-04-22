import datetime as dt
import html
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

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


st.set_page_config(page_title="AfetYonetimi | Need Dashboard", layout="wide")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.75rem;
        }
        .source-banner {
            border-radius: 18px;
            padding: 1.05rem 1.2rem 1.15rem 1.2rem;
            margin: 0.35rem 0 1.15rem 0;
            border: 1px solid rgba(148, 163, 184, 0.35);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.14);
        }
        .source-banner .eyebrow {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.82;
        }
        .source-banner .title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.18rem;
        }
        .source-banner .body {
            margin-top: 0.4rem;
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .source-banner .meta {
            margin-top: 0.5rem;
            font-size: 0.86rem;
            opacity: 0.9;
        }
        .source-banner.canonical {
            background: linear-gradient(130deg, rgba(6, 78, 59, 0.95) 0%, rgba(8, 145, 178, 0.92) 100%);
            color: #f0fdfa;
        }
        .source-banner.candidate {
            background: linear-gradient(130deg, rgba(14, 116, 144, 0.92) 0%, rgba(30, 64, 175, 0.92) 100%);
            color: #eff6ff;
        }
        .source-banner.historical {
            background: linear-gradient(130deg, rgba(146, 64, 14, 0.95) 0%, rgba(194, 65, 12, 0.92) 100%);
            color: #fff7ed;
        }
        .source-banner.custom {
            background: linear-gradient(130deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.92) 100%);
            color: #f8fafc;
        }
        .callout-panel {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, rgba(248, 250, 252, 0.95) 0%, rgba(241, 245, 249, 0.98) 100%);
        }
        .callout-panel h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1.02rem;
        }
        .callout-panel p {
            margin: 0.2rem 0;
            font-size: 0.92rem;
            line-height: 1.45;
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

    date_range = st.sidebar.date_input("Tarih araligi", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    elif isinstance(date_range, list) and len(date_range) == 2:
        start, end = date_range
    else:
        start = end = date_range
    out = out[(date_values >= start) & (date_values <= end)]

    if "province" in out.columns:
        provinces = sorted([value for value in out["province"].dropna().unique().tolist() if value])
        selected_provinces = st.sidebar.multiselect("Il (province)", options=provinces, default=provinces)
        if selected_provinces:
            out = out[out["province"].isin(selected_provinces)]

    if "district" in out.columns:
        districts = sorted([value for value in out["district"].dropna().unique().tolist() if value])
        selected_districts = st.sidebar.multiselect("Ilce (district)", options=districts, default=[])
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
        urgency_min = st.sidebar.slider("Urgency score (min)", min_value=lo, max_value=max(lo, hi), value=lo, step=1)
        out = out[urgency_numeric >= urgency_min]

    selected_labels = st.sidebar.multiselect(
        "Ihtiyac etiketleri",
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

    query = st.sidebar.text_input("Metin ara (tweet/tweet_clean)")
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
        meta_line.append(f"Thresholds: {metadata['threshold_source']} / {threshold_type}")
    if metadata and metadata.get("generated_at"):
        meta_line.append(f"Generated at: {format_generated_at(metadata.get('generated_at'))}")

    meta_text = " | ".join(meta_line)
    banner_body = source_kind_note(source_kind)
    if default_note:
        banner_body = f"{banner_body} {default_note}"

    st.markdown(
        f"""
        <div class="source-banner {banner_class}">
            <div class="eyebrow">Data Provenance</div>
            <div class="title">{html.escape(source_kind_label(source_kind))}</div>
            <div class="body">{html.escape(banner_body)}</div>
            <div class="meta">{html.escape(meta_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_cards(df_all: pd.DataFrame, df_filtered: pd.DataFrame, schema: PredictionSchema) -> None:
    label_counts = _label_counts(df_filtered, schema)
    top_label = pretty_label(label_counts.iloc[0]["label"]) if not label_counts.empty else "n/a"
    top_label_value = int(label_counts.iloc[0]["count"]) if not label_counts.empty else 0

    columns = st.columns(5)
    columns[0].metric("Toplam satir", f"{len(df_all):,}")
    columns[1].metric("Filtrelenmis satir", f"{len(df_filtered):,}")

    if "pred_any_need" in df_filtered.columns and len(df_filtered):
        any_need_total = int(pd.to_numeric(df_filtered["pred_any_need"], errors="coerce").fillna(0).sum())
        any_need_rate = (any_need_total / len(df_filtered)) * 100.0
        columns[2].metric("pred_any_need=1", f"{any_need_total:,}", delta=f"{any_need_rate:.1f}%")
    else:
        columns[2].metric("pred_any_need=1", "n/a")

    if "urgency_score" in df_filtered.columns and len(df_filtered):
        urgency_mean = pd.to_numeric(df_filtered["urgency_score"], errors="coerce").fillna(0.0).mean()
        columns[3].metric("Urgency mean", f"{urgency_mean:.2f}")
    else:
        columns[3].metric("Urgency mean", "n/a")

    columns[4].metric("Top label", top_label, delta=f"{top_label_value:,}" if top_label_value else None)


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
        st.markdown(f"**Prediction CSV**  \n`{format_path(csv_path)}`")
        st.markdown(f"**Metadata JSON**  \n`{format_path(meta_path) if meta_path else 'n/a'}`")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Rows (CSV)", f"{len(df_all):,}")
        metric_cols[1].metric("Rows (meta)", f"{int(meta_row_count):,}" if meta_row_count is not None else "n/a")
        metric_cols[2].metric(
            "Duplicate removal",
            f"{int(duplicate_rows_removed):,}" if duplicate_rows_removed is not None else "n/a",
        )

        if metadata:
            st.markdown(f"**Selected experiment key**: `{metadata.get('selected_experiment_key', 'n/a')}`")
            st.markdown(f"**Selected model dir**: `{metadata.get('model_dir', 'n/a')}`")
            threshold_source = metadata.get("threshold_source", "n/a")
            threshold_type = metadata.get("threshold_type", "n/a")
            st.markdown(f"**Threshold source / type**: `{threshold_source}` / `{threshold_type}`")
            st.markdown(f"**Generated at**: `{format_generated_at(metadata.get('generated_at'))}`")

            if rows_before is not None and rows_after is not None:
                st.caption(f"Dedup summary: rows_before={rows_before:,} -> rows_after={rows_after:,}")

            if meta_row_count is not None and int(meta_row_count) != len(df_all):
                st.warning(
                    f"Metadata row_count ({int(meta_row_count):,}) ile yuklenen CSV satir sayisi ({len(df_all):,}) farkli."
                )
        else:
            st.info("Metadata bulunamadi. Dashboard CSV-only fallback modunda calisiyor.")

        if source_kind == "historical":
            st.warning("Historical 63k preview dosyasi aktif. Bu artifact canonical final output degildir.")

    with right_col:
        st.subheader("Canonical Limitations")
        if source_kind in {"canonical_final", "canonical_candidate"}:
            st.markdown("\n".join([f"- {item}" for item in canonical_limitations(metadata)]))
        elif source_kind == "historical":
            st.markdown(
                "- Historical 63k preview aktif; canonical experiment metadata'si veya final limitation seti dogrudan bagli degil.\n"
                "- Bu dosya karsilastirma icin korunuyor, final production output olarak sunulmuyor."
            )
        else:
            st.markdown(
                "- Custom CSV/meta secildi. Limitations secili dosyanin gercek provenance'ina gore yorumlanmali.\n"
                "- Metadata saglanirsa canonical riskler ve threshold bilgileri daha guvenli sekilde goruntulenir."
            )


def _render_schema_panel(metadata: dict | None, schema: PredictionSchema, prevalence_df: pd.DataFrame) -> None:
    with st.expander("Prediction Schema ve Metadata"):
        if metadata and metadata.get("schema_note"):
            st.caption(metadata.get("schema_note"))
        elif metadata:
            st.caption("Metadata yuklendi; column mapping metadata'dan okunuyor.")
        else:
            st.caption("Metadata yok; pred/prob kolonlari CSV header'indan kesfedildi.")

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
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

        if not prevalence_df.empty:
            st.caption("Pred prevalence (full dataset vs current filter)")
            st.dataframe(prevalence_df, use_container_width=True, hide_index=True)


def _hourly_signal_map(df: pd.DataFrame, schema: PredictionSchema) -> None:
    st.subheader("Saatlik Yardim Sinyalleri (Harita)")

    st.markdown(
        """
        <style>
        .signal-hero {
            background: linear-gradient(120deg, rgba(12,74,110,0.92) 0%, rgba(30,64,175,0.88) 45%, rgba(15,23,42,0.94) 100%);
            border: 1px solid rgba(148, 163, 184, 0.45);
            border-radius: 14px;
            padding: 14px 18px;
            margin: 0.35rem 0 0.85rem 0;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
        }
        .signal-hero-title {
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }
        .signal-hero-sub {
            color: #dbeafe;
            font-size: 0.91rem;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "created_at_local" not in df.columns or df["created_at_local"].isna().all():
        st.info("created_at bilgisi yok; saatlik harita olusturulamadi.")
        return

    hours = pd.to_datetime(df["created_at_local"], errors="coerce").dt.floor("h")
    try:
        hours = hours.dt.tz_localize(None)
    except Exception:
        pass

    hour_values = sorted(pd.DatetimeIndex(hours.dropna().unique()).to_pydatetime().tolist())
    if not hour_values:
        st.info("Saat bilgisi bulunamadi.")
        return

    if "timeline_playing" not in st.session_state:
        st.session_state["timeline_playing"] = False
    if "timeline_interval_s" not in st.session_state:
        st.session_state["timeline_interval_s"] = 0.8
    if "timeline_loop" not in st.session_state:
        st.session_state["timeline_loop"] = True
    if "timeline_show_heatmap" not in st.session_state:
        st.session_state["timeline_show_heatmap"] = False
    if "timeline_pending_hour" in st.session_state:
        pending = st.session_state.pop("timeline_pending_hour")
        if pending in hour_values:
            st.session_state["timeline_hour"] = pending

    if "timeline_hour" not in st.session_state or st.session_state["timeline_hour"] not in hour_values:
        st.session_state["timeline_hour"] = hour_values[0]

    c1, c2, c3, c4, c5 = st.columns([2.2, 2.2, 1.7, 1.2, 1.5])
    with c1:
        loc_level = st.selectbox("Konum seviyesi", options=["neighborhood", "district", "province"], index=0)
    with c2:
        signal_mode = st.selectbox(
            "Sinyal gucu",
            options=["count_rows", "count_any_need", "sum_urgency"],
            index=1 if "pred_any_need" in df.columns else 0,
            format_func={
                "count_rows": "Tweet sayisi (filtreli)",
                "count_any_need": "Any-need tweet sayisi (pred_any_need=1)",
                "sum_urgency": "Urgency toplam (urgency_score)",
            }.get,
        )
    with c3:
        st.slider(
            "Oynatma hizi (sn)",
            min_value=0.2,
            max_value=5.0,
            value=float(st.session_state["timeline_interval_s"]),
            step=0.1,
            key="timeline_interval_s",
        )
    with c4:
        st.checkbox("Loop", value=bool(st.session_state["timeline_loop"]), key="timeline_loop")
    with c5:
        st.checkbox("Heatmap", value=bool(st.session_state["timeline_show_heatmap"]), key="timeline_show_heatmap")

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
        if st.button("Prev", use_container_width=True):
            st.session_state["timeline_playing"] = False
            st.session_state["timeline_hour"] = hour_values[max(0, cur_idx - 1)]
            st.rerun()
    with b2:
        button_label = "Pause" if st.session_state["timeline_playing"] else "Play"
        if st.button(button_label, use_container_width=True):
            st.session_state["timeline_playing"] = not bool(st.session_state["timeline_playing"])
            st.rerun()
    with b3:
        if st.button("Next", use_container_width=True):
            st.session_state["timeline_playing"] = False
            st.session_state["timeline_hour"] = hour_values[min(len(hour_values) - 1, cur_idx + 1)]
            st.rerun()
    with b4:
        st.caption(f"Hours: {len(hour_values)}")

    st.select_slider(
        "Saat sec (hour-by-hour)",
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
        st.info("Harita icin koordinat eslesmesi bulunamadi (gazetteer yok veya eslesme dusuk).")
        return

    aggregated["signal"] = pd.to_numeric(aggregated["signal"], errors="coerce").fillna(0.0)
    aggregated = aggregated[aggregated["signal"] > 0].sort_values("signal", ascending=False).reset_index(drop=True)
    if aggregated.empty:
        st.info("Haritada gosterilecek pozitif sinyal bulunamadi.")
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
            return "Yuksek"
        if value >= q50:
            return "Orta"
        return "Izleme"

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
            return f"{diff:+.{decimals}f} (onceki saat 0)"
        pct = (diff / previous) * 100.0
        return f"{diff:+.{decimals}f} ({pct:+.1f}%)"

    top_location = str(aggregated.iloc[0]["location_label"])
    top_signal = float(aggregated.iloc[0]["signal"])
    st.markdown(
        f"""
        <div class="signal-hero">
            <div class="signal-hero-title">Saatlik Ihtiyac Sinyalleri Analizi</div>
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
        "Sicak nokta",
        f"{int(len(aggregated)):,}",
        delta=_delta_text(float(len(aggregated)), float(prev_hotspots) if prev_hotspots is not None else None, 0),
    )
    m3.metric("Tepe sinyal", f"{top_signal:.0f}")
    geo_match = (len(aggregated) / total_hotspots_before_geo) * 100.0 if total_hotspots_before_geo else 0.0
    m4.metric("Koordinat kapsami", f"{geo_match:.1f}%")

    map_col, table_col = st.columns([1.7, 1.0], gap="large")
    with map_col:
        radius_scale = st.slider("Nokta boyutu carpani", min_value=500, max_value=22000, value=6500, step=500)
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
                    opacity=0.38,
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
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom_map, pitch=22),
            layers=layers,
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption(
            "Gosterim; secili saatteki sinyal yogunlugunu (renk), buyuklugunu (cap) ve hotspot oncelik seviyesini birlikte sunar."
        )

    with table_col:
        top_n_max = int(min(30, len(aggregated)))
        top_n_default = int(min(10, len(aggregated)))
        top_n = st.slider("Sicak nokta listesi", min_value=1, max_value=max(1, top_n_max), value=max(1, top_n_default), step=1)
        top_df = aggregated.head(top_n).copy()
        top_df["signal"] = top_df["signal"].round(0).astype(int)
        top_df["pay"] = top_df["share_pct"].map(lambda value: f"{value:.1f}%")
        st.dataframe(
            top_df[["rank", "location_label", "signal", "pay", "severity"]].rename(
                columns={
                    "rank": "Sira",
                    "location_label": "Konum",
                    "signal": "Sinyal",
                    "pay": "Pay",
                    "severity": "Seviye",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
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
            st.caption("Saatlik ihtiyac etiketleri")
            st.dataframe(label_df, use_container_width=True, hide_index=True)

    with st.expander("Saatlik sinyal tablosu (detayli)"):
        view_cols = ["rank", "location_label", "signal", "share_pct", "severity", "lat", "lon"]
        show_df = aggregated[view_cols].copy()
        show_df["share_pct"] = show_df["share_pct"].map(lambda value: round(float(value), 2))
        st.dataframe(show_df, use_container_width=True, hide_index=True)

    if bool(st.session_state.get("timeline_playing", False)) and len(hour_values) > 1:
        interval_s = float(st.session_state.get("timeline_interval_s", 0.8))
        interval_s = max(0.05, min(interval_s, 30.0))
        time.sleep(interval_s)

        try:
            idx = hour_values.index(st.session_state["timeline_hour"])
        except ValueError:
            idx = 0
            st.session_state["timeline_hour"] = hour_values[0]

        if idx < (len(hour_values) - 1):
            st.session_state["timeline_pending_hour"] = hour_values[idx + 1]
        else:
            if bool(st.session_state.get("timeline_loop", True)):
                st.session_state["timeline_pending_hour"] = hour_values[0]
            else:
                st.session_state["timeline_playing"] = False

        st.rerun()


_inject_styles()

st.title("AfetYonetimi | Ihtiyac Siniflandirma Dashboard")
st.caption("Canonical final v2 output tercih edilir. Historical 63k artifact sadece acik fallback olarak gosterilir.")

default_source = discover_default_source()

st.sidebar.header("Veri Kaynagi")
st.sidebar.caption(f"Otomatik varsayilan: {default_source.label}")
st.sidebar.caption(default_source.note)

csv_path_input = st.sidebar.text_input("Predictions CSV yolu", value=str(default_source.csv_path))
auto_meta = st.sidebar.checkbox("CSV yanindaki metadata dosyasini otomatik ara", value=True)

manual_meta_default = str(default_source.meta_path) if default_source.meta_path else ""
manual_meta_input = ""
if not auto_meta:
    manual_meta_input = st.sidebar.text_input("Metadata JSON yolu", value=manual_meta_default)

resolved_meta_path = infer_meta_path(csv_path_input) if auto_meta else (Path(manual_meta_input).expanduser() if manual_meta_input else None)
if auto_meta and resolved_meta_path is not None:
    st.sidebar.caption(f"Metadata path: {format_path(resolved_meta_path)}")

df_all: pd.DataFrame | None = None
try:
    df_all = load_predictions_csv(csv_path_input)
except FileNotFoundError:
    st.error(f"Dosya bulunamadi: {csv_path_input}")
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

_render_source_banner(source_kind, metadata, banner_note)

df_filtered = _filter_df(df_all, schema)

_render_summary_cards(df_all, df_filtered, schema)
_render_provenance_panel(csv_path_input, str(resolved_meta_path) if resolved_meta_path else None, metadata, df_all, source_kind)

st.subheader("Label Prevalence")
prevalence_df = _label_prevalence(df_all, df_filtered, metadata, schema)
prev_col, chart_col = st.columns([1.45, 1.0], gap="large")
with prev_col:
    if prevalence_df.empty:
        st.info("Pred prevalence tablosu olusturulamadi.")
    else:
        prevalence_view = prevalence_df.copy()
        prevalence_view["label"] = prevalence_view["label"].map(pretty_label)
        st.dataframe(prevalence_view, use_container_width=True, hide_index=True)
with chart_col:
    if not prevalence_df.empty:
        prevalence_chart_df = prevalence_df.copy()
        prevalence_chart_df["label"] = prevalence_chart_df["label"].map(pretty_label)
        st.bar_chart(prevalence_chart_df.set_index("label")[["full_rate_pct", "filtered_rate_pct"]])

_render_schema_panel(metadata, schema, prevalence_df)

if df_filtered.empty:
    st.warning("Secili filtrelerle eslesen satir yok. Provenance ve schema panelleri yine de yukarida gorunur.")
    st.stop()

st.subheader("Etiket Dagilimi (Pred)")
filtered_counts = _label_counts(df_filtered, schema)
if filtered_counts.empty:
    st.info("Filtrelenmis veri icin pred_* label dagilimi bulunamadi.")
else:
    counts_view = filtered_counts.copy()
    counts_view["label"] = counts_view["label"].map(pretty_label)
    st.dataframe(counts_view, use_container_width=True, hide_index=True)
    st.bar_chart(filtered_counts.set_index("label")["count"])

st.subheader("Zamansal Dagilim")
if "date" in df_filtered.columns:
    timeline = df_filtered.groupby("date").size().reset_index(name="count").sort_values("date")
    st.line_chart(timeline.set_index("date")["count"])
else:
    st.info("date kolonu yok; zaman serisi cizilemedi.")

st.subheader("Harita (Il Centroid Prototype)")
province_map_df = _province_map_df(df_filtered)
if province_map_df.empty:
    st.info("Harita icin province -> (lat, lon) eslesmesi bulunamadi.")
else:
    st.map(province_map_df)
    st.dataframe(province_map_df.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

_hourly_signal_map(df_filtered, schema)

st.subheader("Tweet Listesi")
columns_to_show = [
    column
    for column in ["date", "time", "province", "district", "neighborhood", "urgency_score", "tweet_clean"]
    if column in df_filtered.columns
]
predicted_columns = [schema.label_to_pred[label] for label in schema.labels if label in schema.label_to_pred]
columns_to_show = columns_to_show + predicted_columns
st.dataframe(df_filtered[columns_to_show].head(500), use_container_width=True, hide_index=True)
