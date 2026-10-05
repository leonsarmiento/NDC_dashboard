#!/usr/bin/env python3
"""NDC Extraction Explorer — Streamlit dashboard.

Run:
    streamlit run ndc_explorer.py

Features:
  * filter 23k+ extracted passages by country (name or ISO-3), commitment
    type, relevance, target values and free text
  * interactive world map:
      - click a country      -> add / remove it from the filter
      - click several        -> accumulates the selection (ctrl+click too)
      - drag a rectangle     -> adds every country inside the box
  * live preview of the filtered rows
  * one-click export of the exact filtered set to Excel
  * "Pledge Summaries" page: per-country LLM summaries + target aggregation
    (from `pledge_summaries/*.json`, see `summarize_pledges.py`) with a
    click-to-open country map and Excel export
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from export_extractions import apply_filters, load_rows, to_excel

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="NDC Extraction Explorer", layout="wide")

# --- data (cached until the JSON folder changes) --------------------------- #
@st.cache_data(show_spinner="Loading extraction JSONs…")
def get_full_df() -> pd.DataFrame:
    folder = APP_DIR / "all_docs_out"
    if folder.exists():
        return load_rows(folder)
    return load_rows_archive(APP_DIR / "data" / "extractions.tar.gz")


@st.cache_data(show_spinner="Loading extraction JSONs…")
def load_rows_archive(archive: Path) -> pd.DataFrame:
    """Load extraction JSONs shipped as a single compressed archive.

    The dashboard repo does not carry the 100+ MB of raw JSON folders;
    instead the extraction outputs are bundled in ``data/extractions.tar.gz``
    and unpacked to a temp directory on the first page load.
    """
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(td, filter="data")
        # archive is built as "all_docs_out/*.json"; fall back to td itself
        folder = Path(td) / "all_docs_out"
        return load_rows(folder if folder.exists() else Path(td))


@st.cache_data(show_spinner="Loading pledge summaries…")
def get_summaries() -> list[dict]:
    """Per-country LLM summaries written by summarize_pledges.py."""
    folder = APP_DIR / "pledge_summaries"
    if not folder.exists():
        return []
    out = []
    for f in sorted(folder.glob("*_pledge_summary.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


df = get_full_df()

# canonical country option: "ISO3 — Name (n docs)"
docs_per_country = df.groupby("doc_country")["source_file"].nunique()
country_pairs = (
    df[["iso3", "doc_country"]]
    .drop_duplicates()
    .assign(n=df["doc_country"].map(docs_per_country))
    .sort_values("doc_country")
)
option_to_name = {
    f"{r.iso3} — {r.doc_country}": r.doc_country for r in country_pairs.itertuples()
}
option_docs = {
    f"{r.iso3} — {r.doc_country}": int(r.n) for r in country_pairs.itertuples()
}
ALL_OPTIONS = list(option_to_name.keys())

# --- selection state (shared by sidebar list and map) ------------------------ #
if "countries_sel" not in st.session_state:
    st.session_state.countries_sel = []

# --- sidebar: logo + page switch ------------------------------------------- #
st.sidebar.image(str(APP_DIR / "logo_for_banner.png"), width=220)
PAGE = st.sidebar.radio(
    "Page",
    ["Extraction Explorer", "Pledge Summaries"],
    label_visibility="collapsed",
)

# --- page: pledge summaries ------------------------------------------------- #
if PAGE == "Pledge Summaries":
    summaries = get_summaries()
    if not summaries:
        st.title("Pledge Summaries")
        st.info(
            "No summaries found in `pledge_summaries/` yet.\n\n"
            "Generate them with:\n\n"
            "    python summarize_pledges.py --countries COL,BRA"
        )
        st.stop()

    st.title("Pledge Summaries")
    st.caption(
        "One LLM call per country: a one-paragraph summary of the ecosystem "
        "pledges (in perspective of the country's area and land cover) plus an "
        "aggregation of the numerical targets."
    )

    by_iso = {s["iso3"]: s for s in summaries}
    base = df["iso3"].dropna().unique().tolist() if len(df) else []
    # deterministic order of map rows; the map is a single trace, so the
    # clicked point_index equals the row index in this list
    order = sorted(set(base) | set(by_iso.keys()))

    if "summary_iso" not in st.session_state:
        st.session_state.summary_iso = summaries[0]["iso3"]
    cur = st.session_state.summary_iso
    if cur not in by_iso:  # folder changed under us
        st.session_state.summary_iso = cur = summaries[0]["iso3"]

    # Handle a pending map click BEFORE rendering content, so one click
    # updates the map highlight, the dropdown and the summary at once.
    sm = st.session_state.get("summary_map")
    if sm is not None and hasattr(sm, "get"):
        try:
            sig = json.dumps(dict(sm), sort_keys=True)
        except (TypeError, ValueError):
            sig = repr(sm)
        if sig != st.session_state.get("summary_map_last"):
            st.session_state.summary_map_last = sig
            try:  # debug: raw selection payload
                with open("/tmp/ndc_explorer_sel.log", "a") as fh:
                    fh.write(sig + "\n")
            except OSError:
                pass
            pts = list((sm.get("selection") or {}).get("points") or [])
            for p in pts:
                idx = p.get("point_index")
                iso = (
                    order[idx]
                    if isinstance(idx, int) and 0 <= idx < len(order)
                    else (p.get("location") or "")
                )
                iso = (iso or "").upper()
                if iso in by_iso and iso != cur:
                    st.session_state.summary_iso = cur = iso
                elif iso and iso not in by_iso:
                    st.toast(f"No summary for {iso} yet")

    s = by_iso[cur]

    mdf = pd.DataFrame(
        [(i, "current" if i == cur else "summary" if i in by_iso else "none",
          by_iso.get(i, {}).get("country", i))
         for i in order]
    )
    mdf.columns = ["iso3", "status", "name"]

    c1, c2 = st.columns((1, 1), gap="large")
    with c1:
        import plotly.express as px

        # 3 discrete colors via a continuous scale on a SINGLE trace, so the
        # map's point_index equals the row index in `order`:
        rank = mdf["status"].map(
            {"current": 2.0, "summary": 1.0, "none": 0.0}
        ).astype(float)
        fig = px.choropleth(
            mdf, locations="iso3",
            color=rank.to_numpy(),
            hover_name="name",
        )
        fig.update_traces(
            colorscale=[[0, "#d8d8d8"], [0.5, "#1f8a4c"], [1, "#e11d48"]],
            zmin=0, zmax=2,
        )
        # hover shows the country name, not the numeric rank
        fig.data[0].hovertemplate = "%{hovertext}<extra></extra>"
        fig.data[0].hovertext = mdf["name"].to_numpy()
        fig.update_layout(
            height=540, margin=dict(l=0, r=0, t=10, b=0),
            dragmode="pan", showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(
            fig, key="summary_map", width="stretch",
            on_select="rerun",
            selection_mode=["points"],
            config={"scrollZoom": True,
                    "modeBarButtonsToRemove": ["autoScale", "hoverCompare"]},
        )
        st.caption(
            "**Green** = has a summary · **red** = shown below. Click a "
            "green country to open it."
        )
    with c2:
        st.selectbox(
            "Country",
            options=[x["iso3"] for x in summaries],
            index=[x["iso3"] for x in summaries].index(cur),
            format_func=lambda i: f"{i} — {by_iso[i]['country']}",
            key="summary_iso",
        )
        gc = s.get("geo_context") or {}
        lc = gc.get("land_cover_km2") or {}
        m1, m2, m3 = st.columns(3)
        m1.metric("passages", s.get("n_passages", 0))
        m2.metric("targets", len(s.get("targets") or []))
        area = gc.get("country_area_km2")
        m3.metric(
            "country area",
            f"{area:,.0f} km²" if area else "—",
            f"forest {lc.get('Forest', 0):,.0f} km²" if lc else None,
        )
        st.subheader("Summary")
        st.markdown(s.get("summary") or "*(no summary)*")

    targets = s.get("targets") or []
    st.subheader(f"Targets ({len(targets)})")
    if targets:
        tdf = pd.DataFrame(targets)
        for col in ("commitment_type", "metric", "value", "unit",
                    "timeframe", "context"):
            if col not in tdf.columns:
                tdf[col] = None
        tdf = tdf[["commitment_type", "metric", "value", "unit",
                   "timeframe", "context"]]
        st.dataframe(
            tdf, width="stretch", height=420,
            column_config={
                "metric": st.column_config.TextColumn("Metric", width="large"),
                "context": st.column_config.TextColumn("Context", width="large"),
            },
        )
    else:
        st.caption("No numerical targets reported.")

    @st.cache_data(show_spinner="Building summary workbook…")
    def summaries_excel(payload: tuple) -> bytes:
        rows = [json.loads(x) for x in payload]
        sum_rows, tgt_rows = [], []
        for s_ in rows:
            gc_ = s_.get("geo_context") or {}
            lc_ = gc_.get("land_cover_km2") or {}
            sum_rows.append({
                "ISO3": s_.get("iso3"), "Country": s_.get("country"),
                "Model": s_.get("model"),
                "Passages": s_.get("n_passages"),
                "Targets": len(s_.get("targets") or []),
                "Country area (km2)": gc_.get("country_area_km2"),
                "Forest (km2)": lc_.get("Forest"),
                "Summary": s_.get("summary"),
            })
            for t in s_.get("targets") or []:
                tgt_rows.append({
                    "ISO3": s_.get("iso3"), "Country": s_.get("country"),
                    "Commitment type": t.get("commitment_type"),
                    "Metric": t.get("metric"),
                    "Value": t.get("value"), "Unit": t.get("unit"),
                    "Timeframe": t.get("timeframe"),
                    "Context": t.get("context"),
                })
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            pd.DataFrame(sum_rows).to_excel(w, sheet_name="summary", index=False)
            pd.DataFrame(tgt_rows).to_excel(w, sheet_name="targets", index=False)
        return buf.getvalue()

    data = summaries_excel(tuple(json.dumps(x, sort_keys=True) for x in summaries))
    st.download_button(
        "⬇️  Export all summaries to Excel",
        data=data, file_name="pledge_summaries.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    st.stop()

# --- sidebar: filters ------------------------------------------------ #
st.sidebar.title("Filters")

st.sidebar.multiselect(
    "Countries",
    options=ALL_OPTIONS,
    key="countries_sel",
    format_func=lambda o: f"{o}  ({option_docs[o]} docs)",
    placeholder="select countries…",
)
iso3_input = st.sidebar.text_input(
    "…or type ISO-3 codes (comma separated)",
    help="e.g. COL, BRA, EU — augments the country list above",
)
countries = [
    option_to_name[o] for o in st.session_state.countries_sel if o in option_to_name
] + [c.strip() for c in iso3_input.split(",") if c.strip()]

commitment_opts = ["Restoration", "Conservation", "Agriculture", "Both"]
commitments = st.sidebar.multiselect("Commitment type", commitment_opts)
relevances = st.sidebar.multiselect("Relevance confidence", ["High", "Medium", "Low"])
has_target = st.sidebar.checkbox("Only passages with a target value")
target_contains = st.sidebar.text_input("Target value contains")
search = st.sidebar.text_input("Search in text / translation / notes")
st.sidebar.caption(
    "Map below: **click** a country to add/remove it, keep clicking to build a "
    "set (ctrl+click works too), or **drag a box** to add every country inside."
)

# --- map --------------------------------------------------------------------- #
try:
    import plotly.express as px

    MAP_OK = True
except ImportError:
    MAP_OK = False

LOCATIONS: list[str] = []  # iso3 in trace order (filled below)
iso_to_options: dict[str, list[str]] = {}
for opt in ALL_OPTIONS:
    iso = opt.split(" — ", 1)[0]
    iso_to_options.setdefault(iso, []).append(opt)


def on_map_select() -> None:
    """Map selection callback (streamlit 1.64+): called with no arguments.

    The fresh selection state is read from session state under the chart's
    key ("map"); the callback fires only when the selection changes.
    """
    st_map = st.session_state.get("map")
    if not st_map:
        return
    sel = st_map.get("selection") or {}
    pts = list(sel.get("points") or [])
    if not pts:
        return
    selected = list(sel.get("point_indices") or [])
    sel_box = sel.get("box") or []
    sel_lasso = sel.get("lasso") or []

    def iso_for(p: dict) -> str:
        idx = p.get("point_index")
        if isinstance(idx, int) and 0 <= idx < len(LOCATIONS):
            return LOCATIONS[idx]
        for key in ("location", "x0", "x"):
            v = p.get(key)
            if isinstance(v, str) and v.upper() in iso_to_options:
                return v.upper()
        return ""

    is_click = not sel_box and not sel_lasso and len(selected) == 1
    state = st.session_state.countries_sel
    for i, p in enumerate(pts):
        iso = iso_for(p)
        for opt in iso_to_options.get(iso, []):
            if is_click and opt in state:
                state.remove(opt)  # single click on a selected country = remove
            elif opt not in state:
                state.append(opt)


if MAP_OK:
    # countries matching the current filter (sidebar list + ISO-3 box)
    sel_isos = {o.split(" — ", 1)[0] for o in st.session_state.countries_sel}
    sel_isos |= {
        c.strip().upper() for c in iso3_input.split(",") if c.strip()
    }
    map_df = country_pairs[country_pairs["iso3"].ne("")].copy()
    map_df["status"] = map_df["iso3"].isin(sel_isos).map(
        {True: "selected", False: "other"}
    )
    fig = px.choropleth(
        map_df,
        locations="iso3",
        color="status",
        color_discrete_map={"selected": "#1f8a4c", "other": "#d8d8d8"},
        hover_name="doc_country",
        hover_data={"n": "documents", "iso3": False, "status": False},
        labels={"documents": "documents", "doc_country": ""},
    )
    fig.update_layout(
        height=540,
        margin=dict(l=0, r=0, t=10, b=0),
        dragmode="pan",
        showlegend=False,
        coloraxis_showscale=False,
    )
    LOCATIONS = list(fig.data[0].locations)
    st.plotly_chart(
        fig,
        key="map",
        width="stretch",
        on_select=on_map_select,
        selection_mode=["points", "box", "lasso"],
        config={
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["autoScale", "hoverCompare"],
        },
    )
    st.caption(
        "\u2192 Click a country to add/remove it (keep clicking to build up "
        "a multi-country selection; click again to deselect). For region "
        "selection, use the rectangle-select tool in the map's toolbar "
        "(top-left \u25a1), drag a box over the countries, then switch back "
        "to the arrow tool."
    )

# --- main: preview + export -------------------------------------------------- #
out = apply_filters(
    df, countries, commitments, relevances,
    has_target, target_contains, search,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("rows", f"{len(out):,}", f"of {len(df):,}")
c2.metric("documents", out["source_file"].nunique() if len(out) else 0)
c3.metric("countries", out["iso3"].nunique() if len(out) else 0)
c4.metric("with target", int((~out["passage_target_value"].str.upper().isin(("NA", ""))).sum()) if len(out) else 0)

if not len(out):
    st.info("No rows match the current filters.")
    st.stop()

DISPLAY_COLS = [
    "iso3", "doc_country", "source_file",
    "passage_commitment_type", "passage_relevance_confidence",
    "passage_target_value", "passage_target_unit",
    "passage_original_text", "passage_english_translation", "passage_notes",
]
show = st.selectbox("Columns", ["default columns", "all columns"], index=0)
view = out[DISPLAY_COLS if show == "default columns" else out.columns.tolist()]

st.dataframe(
    view,
    width="stretch",
    height=560,
    column_config={
        "passage_original_text": st.column_config.TextColumn("Original text", width="large"),
        "passage_english_translation": st.column_config.TextColumn("English translation", width="large"),
    },
)
st.caption(f"Showing {len(view):,} rows (all filtered rows are exported, not just the preview).")

# export the exact filtered set
buf = io.BytesIO()
to_excel(out, buf, criteria="dashboard export")
buf.seek(0)
st.download_button(
    "⬇️  Export filtered set to Excel",
    data=buf.getvalue(),
    file_name="ndc_extractions_filtered.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
