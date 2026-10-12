#!/usr/bin/env python3
"""Natural-language search over the NDC extractions (EmbeddingGemma 2).

Shared by:
  * ``build_embeddings.py``  -- one-off local build of the embedding index
  * ``ndc_explorer.py``      -- the "Ask the NDCs" dashboard page

Design
------
Every extraction row is embedded once (with the model's ``Document`` prompt)
and stored in ``data/embeddings.npz`` (L2-normalised, float16, same row
order as :func:`export_extractions.load_rows`).  A natural-language query is
embedded with the ``SearchQuery`` prompt and scored by plain cosine
similarity against the stored matrix (numpy matmul, a few ms for 24k rows).

``sentence-transformers`` (not llama.cpp) is used on purpose: the
``gemma-embedding2`` GGUF architecture landed in llama.cpp master on
2026-10-06, i.e. *after* the last released llama-cpp-python wheels
(0.3.36, built 2026-10-01), so no prebuilt wheel -- neither PyPI nor
conda-forge -- can load the model, and building llama.cpp from source at
deploy time exceeds Streamlit Cloud's build window.  Revisit this when a
llama-cpp-python release ships wheels with a llama.cpp that knows
``gemma-embedding2`` (the GGUF lives at ``unsloth/embeddinggemma-2-GGUF``).
"""

from __future__ import annotations

import os
from pathlib import Path

# Conda + torch on macOS ship two OpenMP runtimes (env libopenblas/libomp and
# the one bundled in the torch wheel).  Let them coexist locally; the flag is
# a no-op on Streamlit Cloud (pip-only, single runtime).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "data" / "embeddings.npz"
MODEL_ID = "google/embeddinggemma-2"

# Prompt prefixes from the model's sentence-transformers config
# (config_sentence_transformers.json): task-steered representations.
PROMPT_DOC = "Document"      # "title: none | text: "
PROMPT_QUERY = "SearchQuery"  # "task: search result | query: "

INDEX_VERSION = 1
MAX_SEQ_LEN = 512  # hard cap per passage (model context is 8k; passages are short)


# --- text assembly ---------------------------------------------------------- #
def doc_text(row) -> str:
    """Human-meaningful text for one extraction row.

    Combines the English translation, the original-language text, the
    auditor notes and the numeric target (when present).  "NA" sentinels
    are dropped.  The country name is *not* mixed in: country scoping is
    done by the structured iso3 field after the cosine search.
    """
    parts: list[str] = []
    for k in ("passage_english_translation", "passage_original_text"):
        v = str(row.get(k, "NA")).strip()
        if v and v.upper() != "NA":
            parts.append(v)
    notes = str(row.get("passage_notes", "NA")).strip()
    if notes and notes.upper() != "NA":
        parts.append(notes)
    tv = str(row.get("passage_target_value", "NA")).strip()
    if tv and tv.upper() != "NA":
        unit = str(row.get("passage_target_unit", "NA")).strip()
        if unit.upper() == "NA":
            unit = ""
        parts.append(f"Target: {tv} {unit}".strip())
    ct = str(row.get("passage_commitment_type", "NA")).strip()
    if ct and ct.upper() != "NA":
        parts.append(f"Commitment type: {ct}")
    return "\n".join(parts)


# --- model ------------------------------------------------------------------- #
def load_model():
    """Load EmbeddingGemma 2 (text-only use) via sentence-transformers.

    First call downloads ~1.5 GB from Hugging Face into the local HF cache
    (on Streamlit Cloud the cache persists, so the download happens once).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID)
    try:  # keep long passages inside one window (attribute may move across ST versions)
        model.max_seq_length = MAX_SEQ_LEN
    except Exception:  # noqa: BLE001
        pass
    return model


def embed_query(model, query: str):
    """Embed one natural-language query (L2-normalised numpy vector)."""
    import numpy as np

    v = model.encode(query, prompt_name=PROMPT_QUERY, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


# --- index ------------------------------------------------------------------- #
def build_index_texts(df) -> list[str]:
    """Per-row document strings for the whole dataframe (row order = embed order)."""
    return [doc_text(r) for r in df.to_dict("records")]


def save_index(path: Path, vectors, n_rows: int, dim: int) -> None:
    """Store the normalised matrix (float16) plus provenance metadata."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.ascontiguousarray(vectors, dtype=np.float16),
        version=np.array(INDEX_VERSION),
        n_rows=np.array(n_rows),
        dim=np.array(dim),
        model=np.array(MODEL_ID),
        prompt=np.array(PROMPT_DOC),
    )


def load_index(path: Path = INDEX_PATH):
    """Load the index -> (vectors f32, meta dict) or raise FileNotFoundError."""
    import numpy as np

    z = np.load(path, allow_pickle=False)
    meta = {
        "version": int(z["version"]),
        "n_rows": int(z["n_rows"]),
        "dim": int(z["dim"]),
        "model": str(z["model"]),
    }
    return z["vectors"].astype(np.float32), meta


def index_matches_df(df) -> bool:
    """The index and the dataframe must have the same row count (same loader, same order)."""
    import numpy as np

    if not INDEX_PATH.exists():
        return False
    try:
        z = np.load(INDEX_PATH, allow_pickle=False)
        return int(z["n_rows"]) == len(df) and int(z["version"]) == INDEX_VERSION
    except Exception:  # noqa: BLE001
        return False


# --- search ------------------------------------------------------------------ #
def search(index: np.ndarray, query_vec, top_k: int = 25,
           mask: np.ndarray | None = None) -> np.ndarray:
    """Cosine top-k.  ``mask`` optionally restricts to a subset of rows
    (boolean array aligned with the index rows); fewer candidates than
    ``top_k`` simply return all of them."""
    import numpy as np

    scores = index @ query_vec  # both L2-normalised -> cosine
    if mask is not None and (~mask).any():
        scores = np.where(mask, scores, -np.inf)
    if mask is not None:
        k = min(top_k, int(mask.sum()))
    else:
        k = min(top_k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]
