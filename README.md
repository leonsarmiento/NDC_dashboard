# NDC Dashboard

Interactive Streamlit dashboard for exploring **NDC (Nationally Determined
Contribution) document extractions** and **per-country pledge summaries**.

**Live deployment:** https://iis-ndc-explorer.streamlit.app

Three pages (switchable in the sidebar):

- **Extraction Explorer** — live filtering of 23,000+ extracted passages
  (country, commitment type, relevance, target values, free-text search),
  an interactive choropleth map (click countries or drag a rectangle to
  select regions), a passage preview, and one-click Excel export.
- **Pledge Summaries** — one LLM-written paragraph per country (in the
  perspective of the country's area and land cover) plus the aggregated
  numerical targets, with a world map — **gray** = no summary, **green** =
  has one, **red** = currently shown; click a country to open it — and an
  Excel export of all summaries.
- **Ask the NDCs** — natural-language (semantic) search over all
  extraction passages, powered by Google's **EmbeddingGemma 2** running
  fully locally (see below).

## Data

| Path | Contents |
| ---- | -------- |
| `data/extractions.tar.gz` | All extraction JSONs (one per NDC document, ~23.7k passages), bundled and compressed. Extracted to a temp folder on first page load. |
| `data/embeddings.npz` | EmbeddingGemma-2 vectors for every extraction row (L2-normalised, float16, 768d) + provenance metadata. Built locally with `build_embeddings.py`, committed so the deployed dashboard can search. |
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

…and then rebuild the embeddings (below), or the "Ask the NDCs" page will
report the index as out of sync.

## Natural-language search (Ask the NDCs)

Every extraction row is embedded once with
[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(740M params, 768-d vectors, Apache-2.0) via `sentence-transformers`, using
the model's own task-steered prompts (`Document` for passages,
`SearchQuery` for questions). Queries are embedded at dashboard runtime and
scored by cosine similarity against the stored matrix — a ~1.5 GB model
download on first session (cached afterwards, incl. on Streamlit Cloud) and
a sub-second search over 24k passages.

**Why `sentence-transformers` and not llama.cpp?** The `gemma-embedding2`
GGUF architecture (e.g. `unsloth/embeddinggemma-2-GGUF`, Q4_K_XL ≈ 167 MB)
was added to llama.cpp master on 2026-10-06, *after* the last released
`llama-cpp-python` wheels (0.3.36, built 2026-10-01 — PyPI and conda-forge
both). No prebuilt wheel can therefore load the model, and compiling
llama.cpp from source during a Streamlit Cloud deploy exceeds the build
window. Revisit when a newer llama-cpp-python release ships wheels built
against a llama.cpp that knows `gemma-embedding2`.

Rebuilding the index (locally, ~20 min on an Apple Silicon Mac):

```bash
python build_embeddings.py   # conda env below; writes data/embeddings.npz
```

The index row order is locked to `export_extractions.load_rows` (sorted
document JSONs, passages in document order) — the dashboard rejects the
page (with instructions) if the row count doesn't match.

## Running locally

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
streamlit run ndc_explorer.py
```

The recommended environment (`ndc_dash`, macOS/Apple Silicon):

```bash
conda create -n ndc_dash python=3.12 -c conda-forge -y
conda install -n ndc_dash -c conda-forge streamlit plotly pandas openpyxl -y
pip install sentence-transformers torchvision   # + torch with MPS
```

On macOS, conda's OpenMP (via libopenblas) and torch's bundled OpenMP
coexist through `KMP_DUPLICATE_LIB_OK=TRUE`, which `nl_search.py` sets
automatically before torch is imported (no-op on Linux/Cloud).

Then open the URL Streamlit prints (default http://localhost:8501).

## Deploying (Streamlit Community Cloud)

1. Connect this repository to a Streamlit app in your account.
2. **Python file**: `ndc_explorer.py`
3. **Secrets**: none required.

`requirements.txt` pulls CPU-only torch wheels from
`https://download.pytorch.org/whl/cpu` (same pattern as the PicoChat
deployment); the embedding model downloads from Hugging Face into the app's
persistent cache on the first "Ask the NDCs" session.
