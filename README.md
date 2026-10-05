# NDC Dashboard

Interactive Streamlit dashboard for exploring **NDC (Nationally Determined
Contribution) document extractions** and **per-country pledge summaries**.

**Live deployment:** https://iis-ndc-explorer.streamlit.app

Two pages (switchable in the sidebar):

- **Extraction Explorer** — live filtering of 23,000+ extracted passages
  (country, commitment type, relevance, target values, free-text search),
  an interactive choropleth map (click countries or drag a rectangle to
  select regions), a passage preview, and one-click Excel export.
- **Pledge Summaries** — one LLM-written paragraph per country (in the
  perspective of the country's area and land cover) plus the aggregated
  numerical targets, with a world map — **gray** = no summary, **green** =
  has one, **red** = currently shown; click a country to open it — and an
  Excel export of all summaries.

## Data

| Path | Contents |
| ---- | -------- |
| `data/extractions.tar.gz` | All extraction JSONs (one per NDC document, ~23.7k passages), bundled and compressed. Extracted to a temp folder on first page load. |
| `pledge_summaries/` | One `*_pledge_summary.json` per country. |
| `logo_for_banner.png` | Sidebar banner. |

The heavy source material (document folders, rasters, shapefiles) lives in
the companion repo [`NDC_document_processor`](https://github.com/leonsarmiento/NDC_document_processor)
and is **not** part of this dashboard.

Country names in the extraction JSONs are normalised to 3-letter codes
(the World Bank Admin-0 `ISO_A3` list, same as the geodata pipeline and the
maps) by `export_extractions.py` at load time — keep it in sync with the
companion repo when the mapping improves.

To refresh the extraction data after a new processing run, repack:

```bash
# from NDC_document_processor/
python - <<'EOF'
import tarfile
from pathlib import Path
with tarfile.open("../NDC_dashboard/data/extractions.tar.gz", "w:gz") as t:
    for f in sorted(Path("all_docs_out").glob("*.json")):
        if not f.name.startswith("._"):
            t.add(f, arcname=f"all_docs_out/{f.name}")
EOF
```

## Running locally

```bash
pip install -r requirements.txt   # or: conda env create (see requirements.txt)
streamlit run ndc_explorer.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Deploying (Streamlit Community Cloud)

1. Connect this repository to a Streamlit app in your account.
2. **Python file**: `ndc_explorer.py`
3. **Secrets**: none required.
