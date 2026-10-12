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
- **Ask the NDCs** — hybrid natural-language search over all
  extraction passages: Google's **EmbeddingGemma 2** (semantic, running
  fully locally) fused with a BM25 keyword index via reciprocal rank
  fusion — so "flood control" matches actual flood-control passages, not
  generic "control" text. An optional **strict mode** scores each keyword
  separately and requires them all. Filters for "only passages with a
  target value" and by country (useful because embeddings are topic-based);
  clicking a row opens the full passage in a card below the table (original
  + translation texts, target, notes, document metadata — no table panning).

## Use of AI

**Open-weights** AI models and coding agents assisted, under the authors'
direction, with specific parts of the data-analysis pipeline described in these
repositories: the NDC document extractions, the coding of the web dashboard, and
the coding of the geoprocessing pipeline (not yet exposed in the web dashboard,
but to be added later). The analysis of the extracted results (summaries) and
the final conclusions are the authors' own work, not AI-generated. This
data-analysis pipeline is only a part of the larger project, not the project as
a whole. The authors deliberately chose open-weights models over proprietary
closed-source models to keep the pipeline reproducible and independent of any
single commercial vendor: open-weights LLMs for the extraction, and coding
agents for the coding.

- **Coding & orchestration** — implementation, debugging and experiment
  orchestration for the web dashboard and the geoprocessing pipeline were
  carried out with coding agents running on the open-weights model
  **Qwen3.8‑27B**, orchestrated through the **Pi** agent harness.
- **Text extraction** — the extractions from the NDC source documents were
  produced by open-weights language models running on local and cloud
  servers, including **Qwen3.6‑35B‑A3B**, **DeepSeek v4 Flash**,
  **DeepSeek v4.1 Flash**, **Ling 3 Flash** and **Ling 3.1 Flash**.
- **Semantic search ("Ask the NDCs")** — the dashboard's natural-language
  search embeds every passage with the open-weights **Google EmbeddingGemma
  2** model (768‑d vectors, Apache‑2.0), running fully locally, and fuses the
  result with a BM25 keyword index.

Extractions from a random **30% subset of countries** are checked by an
**expert in NDC pledges and ecosystem services**: the expert independently
identifies each country's main pledges, confirms that each is captured in the
LLM extractions, and iterates on the extraction prompt template until it
reliably captures them. The authors designed the experiments, made the
research decisions, reviewed the results, and are responsible for the full
pipeline, the revision of the results, and the final conclusions.

## Data

| Path | Contents |
| ---- | -------- |
| `data/extractions.tar.gz` | All extraction JSONs (one per NDC document, ~23.7k passages), bundled and compressed. Extracted to a temp folder on first page load. |
| `data/embeddings.npz` | EmbeddingGemma-2 vectors for every extraction row (L2-normalised, float16, 768d) + provenance metadata. Built locally with `build_embeddings.py`, committed so the deployed dashboard can search. |
| `data/bm25_index.pkl` | BM25 keyword index over the same passages (posting lists, ~12 MB), built by the same script for the hybrid search. |
| `pledge_summaries/` | One `*_pledge_summary.json` per country. |
| `logo_for_banner.png` | Sidebar banner. |

The heavy source material (document folders, rasters, shapefiles) lives in
the companion repo [`NDC_document_processor`](https://github.com/leonsarmiento/NDC_document_processor)
and is **not** part of this dashboard.

Country names in the extraction JSONs are normalised to 3-letter codes
(the World Bank Admin-0 `ISO_A3` list, same as the geodata pipeline and the
maps) by `export_extractions.py` at load time — keep it in sync with the
companion repo when the mapping improves.

The choropleth maps are drawn with the **world geography bundled in the
`plotly` package** (rendered client-side), not with any project shapefile.
A footnote under each map (`MAP_BOUNDARY_NOTE` in `ndc_explorer.py`) states
that the boundaries are for reference only and do not reflect the authors'
position on national boundaries or territories in dispute.

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

…and then rebuild the embeddings (below, which also rebuilds the BM25
keyword index), or the "Ask the NDCs" page will
report the index as out of sync.

## Natural-language search (Ask the NDCs)

The search is **hybrid**: two independent rankings over the same 23,742
passages are fused by reciprocal rank fusion (RRF, k = 60).

1. **Semantic rank** — every extraction row is embedded once with
   [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
   (740M params, 768-d vectors, Apache-2.0) via `sentence-transformers`,
   using the model's own task-steered prompts (`Document` for passages,
   `SearchQuery` for questions). Queries are embedded at dashboard runtime
   and scored by cosine similarity against the stored matrix.
2. **Keyword rank** — a precomputed BM25 index (`data/bm25_index.pkl`,
   posting lists, no model involved) scores the passage text for the query's
   exact terms.

**Why hybrid?** Dense single-vector search blends a multi-word query into
one direction in vector space, so e.g. "flood control" was being satisfied
by passages that merely mention "control" in unrelated contexts. The BM25
channel gives credit only where the words actually appear, which fixes that
class of problem without changing the model. A **strict mode** checkbox goes
further: it embeds each keyword separately and combines the per-keyword
cosines as a geometric mean, so *all* concepts must be present to rank high.

The ~1.5 GB model downloads on the first session (cached afterwards, incl.
on Streamlit Cloud); each search is one short model inference plus a
few-millisecond matrix multiply and BM25 scan.

**Why `sentence-transformers` and not llama.cpp?** The `gemma-embedding2`
GGUF architecture (e.g. `unsloth/embeddinggemma-2-GGUF`, Q4_K_XL ≈ 167 MB)
was added to llama.cpp master on 2026-10-06, *after* the last released
`llama-cpp-python` wheels (0.3.36, built 2026-10-01 — PyPI and conda-forge
both). No prebuilt wheel can therefore load the model, and compiling
llama.cpp from source during a Streamlit Cloud deploy exceeds the build
window. Revisit when a newer llama-cpp-python release ships wheels built
against a llama.cpp that knows `gemma-embedding2`.

Rebuilding both indexes (locally, ~20 min on an Apple Silicon Mac — the
embedding part dominates; BM25 takes seconds):

```bash
python build_embeddings.py   # conda env below; writes data/embeddings.npz
                             # and data/bm25_index.pkl
```

The index row order is locked to `export_extractions.load_rows` (sorted
document JSONs, passages in document order). If the embedding index doesn't
match the row count, the page stops with rebuild instructions; a missing or
out-of-sync BM25 index merely warns and falls back to pure semantic search.

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
