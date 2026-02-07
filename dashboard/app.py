import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="AfetYonetimi | Need Dashboard", layout="wide")


LABELS = [
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


# Province centroids (approx). Used only for a lightweight prototype map.
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


@st.cache_data(show_spinner=False)
def load_predictions_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)

    # Coerce core fields
    if "urgency_score" in df.columns:
        df["urgency_score"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0).astype(int)
    for c in ["province", "district", "neighborhood", "date", "time"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").str.strip()

    # Combine datetime for sorting/filtering (best-effort).
    if "created_at" in df.columns:
        # Parse and keep a "local" view for timeline playback. Input strings in this project
        # already have `+03:00`, so converting to Europe/Istanbul keeps hours intuitive.
        ts_utc = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        try:
            df["created_at_local"] = ts_utc.dt.tz_convert("Europe/Istanbul")
        except Exception:
            df["created_at_local"] = ts_utc
        df["created_at_parsed"] = ts_utc
    else:
        df["created_at_local"] = pd.NaT
        df["created_at_parsed"] = pd.NaT

    # Ensure pred_* columns are numeric 0/1
    for lab in LABELS:
        c = f"pred_{lab}"
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in ["pred_any_need", "pred_label_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_location_index(path_str: str = "data/gazetteer/earthquake_region_neighborhoods.csv"):
    path = Path(path_str)
    # Return empty tables if the gazetteer isn't available locally.
    empty_neigh = pd.DataFrame(columns=["province", "district", "neighborhood_clean", "lat", "lon"])
    empty_dist = pd.DataFrame(columns=["province", "district", "lat", "lon"])
    empty_prov = pd.DataFrame(columns=["province", "lat", "lon"])
    if not path.exists():
        return empty_neigh, empty_dist, empty_prov

    g = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    for c in ["province", "district", "neighborhood_clean", "lat", "lon"]:
        if c not in g.columns:
            g[c] = ""

    g["province"] = g["province"].astype("string").fillna("").str.strip()
    g["district"] = g["district"].astype("string").fillna("").str.strip()
    g["neighborhood_clean"] = g["neighborhood_clean"].astype("string").fillna("").str.strip().str.lower()
    g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
    g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    neigh = g[["province", "district", "neighborhood_clean", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    dist = g.groupby(["province", "district"], dropna=False)[["lat", "lon"]].mean().reset_index()
    prov = g.groupby(["province"], dropna=False)[["lat", "lon"]].mean().reset_index()
    return neigh, dist, prov


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df

    # Date filter: prefer `date` column, else fallback to created_at.
    if "date" in out.columns:
        d = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        d = out["created_at_parsed"].dt.date

    valid_dates = d.dropna()
    if not valid_dates.empty:
        min_d = valid_dates.min()
        max_d = valid_dates.max()
    else:
        min_d = dt.date(2023, 2, 6)
        max_d = dt.date(2023, 2, 13)

    date_range = st.sidebar.date_input("Tarih araligi", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        out = out[(d >= start) & (d <= end)]

    # Location filters
    if "province" in out.columns:
        provs = sorted([p for p in out["province"].dropna().unique().tolist() if p and p not in ("NA", "Unknown")])
        sel_prov = st.sidebar.multiselect("Il (province)", options=provs, default=provs)
        if sel_prov:
            out = out[out["province"].isin(sel_prov)]

    if "district" in out.columns:
        dists = sorted([x for x in out["district"].dropna().unique().tolist() if x])
        sel_dist = st.sidebar.multiselect("Ilce (district)", options=dists, default=[])
        if sel_dist:
            out = out[out["district"].isin(sel_dist)]

    if "neighborhood" in out.columns:
        neighs = sorted([x for x in out["neighborhood"].dropna().unique().tolist() if x])
        sel_neigh = st.sidebar.multiselect("Mahalle (neighborhood)", options=neighs, default=[])
        if sel_neigh:
            out = out[out["neighborhood"].isin(sel_neigh)]

    # Urgency filter
    if "urgency_score" in out.columns and not out.empty:
        lo, hi = int(out["urgency_score"].min()), int(out["urgency_score"].max())
        urg = st.sidebar.slider("Urgency score (min)", min_value=lo, max_value=hi, value=0, step=1)
        out = out[out["urgency_score"] >= urg]

    # Label filter
    sel_labels = st.sidebar.multiselect("Ihtiyac etiketleri (pred_*)", options=LABELS, default=[])
    mode = st.sidebar.radio("Etiket filtresi modu", options=["ANY", "ALL"], index=0)
    if sel_labels:
        cols = [f"pred_{x}" for x in sel_labels if f"pred_{x}" in out.columns]
        if cols:
            if mode == "ALL":
                out = out[out[cols].sum(axis=1) == len(cols)]
            else:
                out = out[out[cols].sum(axis=1) > 0]

    # Text search
    q = st.sidebar.text_input("Metin ara (tweet/tweet_clean)")
    if q:
        ql = q.lower()
        tc = (
            out["tweet_clean"].astype("string").fillna("")
            if "tweet_clean" in out.columns
            else pd.Series([""] * len(out), index=out.index)
        )
        tt = (
            out["tweet"].astype("string").fillna("")
            if "tweet" in out.columns
            else pd.Series([""] * len(out), index=out.index)
        )
        mask = tc.str.lower().str.contains(ql, na=False) | tt.str.lower().str.contains(ql, na=False)
        out = out[mask]

    return out


def _label_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab in LABELS:
        c = f"pred_{lab}"
        if c not in df.columns:
            continue
        rows.append({"label": lab, "count": int(df[c].sum())})
    if not rows:
        return pd.DataFrame(columns=["label", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _province_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if "province" not in df.columns:
        return pd.DataFrame(columns=["province", "count", "lat", "lon"])
    g = df.groupby("province", dropna=False).size().reset_index(name="count")
    g["lat"] = np.nan
    g["lon"] = np.nan
    for i, row in g.iterrows():
        prov = str(row["province"])
        if prov in PROVINCE_CENTROID:
            lat, lon = PROVINCE_CENTROID[prov]
            g.at[i, "lat"] = lat
            g.at[i, "lon"] = lon
    g = g.dropna(subset=["lat", "lon"])
    return g


def _hourly_signal_map(df: pd.DataFrame) -> None:
    st.subheader("Saatlik Yardim Sinyalleri (Harita)")

    if "created_at_local" not in df.columns or df["created_at_local"].isna().all():
        st.info("created_at bilgisi yok; saatlik harita olusturulamadi.")
        return

    # Hour list for playback.
    # NOTE: use "h" (not "H") to avoid pandas deprecation warnings.
    hours = pd.to_datetime(df["created_at_local"], errors="coerce").dt.floor("h")
    try:
        hours = hours.dt.tz_localize(None)
    except Exception:
        pass

    # Do NOT use numpy datetime64 `.tolist()` here: it becomes int nanoseconds and breaks strftime.
    hour_values = sorted(pd.DatetimeIndex(hours.dropna().unique()).to_pydatetime().tolist())
    if not hour_values:
        st.info("Saat bilgisi bulunamadi.")
        return

    # Playback state (persistent across reruns)
    if "timeline_playing" not in st.session_state:
        st.session_state["timeline_playing"] = False
    if "timeline_interval_s" not in st.session_state:
        st.session_state["timeline_interval_s"] = 0.8
    if "timeline_loop" not in st.session_state:
        st.session_state["timeline_loop"] = True
    # If playback scheduled a "pending" hour, apply it BEFORE widget instantiation.
    if "timeline_pending_hour" in st.session_state:
        pending = st.session_state.pop("timeline_pending_hour")
        if pending in hour_values:
            st.session_state["timeline_hour"] = pending

    if "timeline_hour" not in st.session_state or st.session_state["timeline_hour"] not in hour_values:
        st.session_state["timeline_hour"] = hour_values[0]

    # Controls
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
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
        label = "Pause" if st.session_state["timeline_playing"] else "Play"
        if st.button(label, use_container_width=True):
            st.session_state["timeline_playing"] = not bool(st.session_state["timeline_playing"])
            st.rerun()
    with b3:
        if st.button("Next", use_container_width=True):
            st.session_state["timeline_playing"] = False
            st.session_state["timeline_hour"] = hour_values[min(len(hour_values) - 1, cur_idx + 1)]
            st.rerun()
    with b4:
        st.caption(f"Hours: {len(hour_values)}")

    # Hour selector (synced with session_state so playback can move it)
    st.select_slider(
        "Saat sec (hour-by-hour)",
        options=hour_values,
        key="timeline_hour",
        format_func=lambda x: x.strftime("%Y-%m-%d %H:00"),
    )
    selected_hour = st.session_state["timeline_hour"]

    # Compute hour column aligned with slider.
    df2 = df.copy()
    df2["_hour"] = hours
    df2 = df2[df2["_hour"] == selected_hour]

    if df2.empty:
        st.info("Bu saatte filtreye uyan veri yok.")
        return

    if signal_mode == "count_any_need" and "pred_any_need" in df2.columns:
        df2 = df2[pd.to_numeric(df2["pred_any_need"], errors="coerce").fillna(0).astype(int) == 1]
    if df2.empty:
        st.info("Bu saatte (pred_any_need=1) sinyal yok.")
        return

    # Aggregate signals per location.
    if loc_level == "province":
        group_cols = ["province"]
    elif loc_level == "district":
        group_cols = ["province", "district"]
    else:
        group_cols = ["province", "district", "neighborhood"]

    if signal_mode == "sum_urgency" and "urgency_score" in df2.columns:
        agg = df2.groupby(group_cols, dropna=False)["urgency_score"].sum().reset_index(name="signal")
    else:
        agg = df2.groupby(group_cols, dropna=False).size().reset_index(name="signal")

    # Attach coordinates from gazetteer centroids (fallback to built-in province centroids).
    neigh_ix, dist_ix, prov_ix = load_location_index()

    # Normalize neighborhood join key (dataset has `neighborhood`, gazetteer has `neighborhood_clean`).
    if loc_level == "neighborhood":
        if "neighborhood" not in agg.columns:
            st.info("neighborhood kolonu yok; mahalle bazli harita olusturulamadi.")
            return
        agg["neighborhood_clean"] = agg["neighborhood"].astype("string").fillna("").str.strip().str.lower()
        agg = agg[agg["neighborhood_clean"] != ""].reset_index(drop=True)

        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        agg = agg.merge(neigh_ix, on=["province", "district", "neighborhood_clean"], how="left")
        agg = agg.merge(dist_ix, on=["province", "district"], how="left")
        agg = agg.merge(prov_ix, on=["province"], how="left")
        agg["lat"] = agg["lat"].fillna(agg.get("lat_dist")).fillna(agg.get("lat_prov"))
        agg["lon"] = agg["lon"].fillna(agg.get("lon_dist")).fillna(agg.get("lon_prov"))
        agg = agg.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")

    elif loc_level == "district":
        dist_ix = dist_ix.rename(columns={"lat": "lat_dist", "lon": "lon_dist"})
        prov_ix = prov_ix.rename(columns={"lat": "lat_prov", "lon": "lon_prov"})
        agg = agg.merge(dist_ix, on=["province", "district"], how="left")
        agg = agg.merge(prov_ix, on=["province"], how="left")
        agg["lat"] = agg["lat_dist"].fillna(agg.get("lat_prov"))
        agg["lon"] = agg["lon_dist"].fillna(agg.get("lon_prov"))
        agg = agg.drop(columns=["lat_dist", "lon_dist", "lat_prov", "lon_prov"], errors="ignore")

    else:
        agg = agg.merge(prov_ix, on=["province"], how="left")

    # Fall back to hardcoded province centroids (prototype) when still missing.
    if "province" in agg.columns:
        missing = agg["lat"].isna() | agg["lon"].isna()
        if bool(missing.any()):
            for i, row in agg.loc[missing].iterrows():
                prov = str(row.get("province", ""))
                if prov in PROVINCE_CENTROID:
                    lat, lon = PROVINCE_CENTROID[prov]
                    agg.at[i, "lat"] = lat
                    agg.at[i, "lon"] = lon

    agg = agg.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if agg.empty:
        st.info("Harita icin koordinat eslesmesi bulunamadi (gazetteer yok veya eslesme dusuk).")
        return

    # Visual scale: sqrt makes dense hours readable.
    radius_scale = st.slider("Nokta boyutu carpani", min_value=500, max_value=20000, value=4000, step=500)
    agg["radius"] = (np.sqrt(agg["signal"].astype(float).clip(lower=0.0)) * float(radius_scale)).clip(lower=500.0)

    loc_text = "{province}"
    if loc_level in ("district", "neighborhood"):
        loc_text += " / {district}"
    if loc_level == "neighborhood":
        loc_text += " / {neighborhood}"

    tooltip = {"text": loc_text + "\\nSaat: " + selected_hour.strftime("%Y-%m-%d %H:00") + "\\nSinyal: {signal}"}

    center_lat = float(agg["lat"].mean())
    center_lon = float(agg["lon"].mean())

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color=[220, 0, 0, 140],
        get_line_color=[120, 0, 0, 200],
        line_width_min_pixels=1,
        pickable=True,
    )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=6, pitch=0),
        layers=[layer],
        tooltip=tooltip,
    )

    st.pydeck_chart(deck, use_container_width=True)
    st.caption("Not: Bu harita, saatlik sinyalleri (filtrelenmis veri) il/ilce centroid'leri uzerinden gosterir.")

    with st.expander("Saatlik sinyal tablosu"):
        st.dataframe(agg.sort_values("signal", ascending=False), use_container_width=True, hide_index=True)

    # Auto-play: schedule the next hour (do NOT mutate the widget key after instantiation).
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


st.title("AfetYonetimi | Ihtiyac Siniflandirma Dashboard (Pseudo/Silver)")

st.sidebar.header("Veri Kaynagi")
default_path = "data/predictions/need_predictions_geolocated_63k.csv"
path = st.sidebar.text_input("Predictions CSV yolu", value=default_path)

df_all: pd.DataFrame | None = None
try:
    df_all = load_predictions_csv(path)
except FileNotFoundError:
    st.error(f"Dosya bulunamadi: {path}")
    df_all = None

if df_all is None or df_all.empty:
    st.stop()

else:
    df = _filter_df(df_all)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam satir (tum veri)", f"{len(df_all):,}")
    col2.metric("Filtrelenmis satir", f"{len(df):,}")
    if "pred_any_need" in df.columns:
        col3.metric("Any need=1", f"{int(pd.to_numeric(df['pred_any_need'], errors='coerce').fillna(0).sum()):,}")
    else:
        col3.metric("Any need=1", "n/a")
    if "urgency_score" in df.columns and len(df):
        col4.metric("Urgency mean", f"{df['urgency_score'].mean():.2f}")
    else:
        col4.metric("Urgency mean", "n/a")

    st.subheader("Etiket Dagilimi (Pred)")
    counts = _label_counts(df)
    st.dataframe(counts, use_container_width=True, hide_index=True)

    if not counts.empty:
        st.bar_chart(counts.set_index("label")["count"])

    st.subheader("Zamansal Dagilim")
    if "date" in df.columns:
        ts = df.groupby("date").size().reset_index(name="count").sort_values("date")
        st.line_chart(ts.set_index("date")["count"])
    else:
        st.info("date kolonu yok; zaman serisi cizilemedi.")

    st.subheader("Harita (Il Centroid Prototype)")
    map_df = _province_map_df(df)
    if map_df.empty:
        st.info("Map icin province->(lat,lon) eslesmesi bulunamadi.")
    else:
        st.map(map_df)
        st.dataframe(map_df.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

    _hourly_signal_map(df)

    st.subheader("Tweet Listesi")
    cols_show = [c for c in ["date", "time", "province", "district", "neighborhood", "urgency_score", "tweet_clean"] if c in df.columns]
    pred_show = [f"pred_{x}" for x in LABELS if f"pred_{x}" in df.columns]
    cols_show = cols_show + pred_show

    st.dataframe(df[cols_show].head(500), use_container_width=True, hide_index=True)
