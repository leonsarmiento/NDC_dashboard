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
LEX_PATH = APP_DIR / "data" / "bm25_index.pkl"
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


# --- lexical channel (BM25) + hybrid fusion --------------------------------- #
# Dense single-vector search blends multi-word queries into one direction in
# vector space, so "flood control" can match generic "control" passages.
# A cheap BM25 channel over the same passages (no model, precomputed locally)
# fused via reciprocal-rank fusion fixes that class of problem.

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
    "is", "are", "was", "will", "shall", "may", "can", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "as", "at", "from", "into",
    "per", "via", "which", "who", "whom", "our", "their", "they", "we",
    "there", "here", "more", "most", "other", "such", "than", "then", "also",
    "only", "very", "de", "la", "le", "les", "du", "des", "el", "en", "que",
    "pour", "par", "au", "aux", "para", "por", "los", "las", "una", "un",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on non-alphanumerics (len >= 2)."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) >= 2]


def content_terms(query: str) -> list[str]:
    """Meaningful query terms (no stopwords, no pure numbers, len >= 3)."""
    return [
        t for t in tokenize(query)
        if t not in STOPWORDS and not t.isdigit() and len(t) >= 3
    ]


def build_lex_index(texts: list[str]) -> dict:
    """BM25 posting lists for the whole corpus (same row order as the index)."""
    import numpy as np

    vocab: dict[str, int] = {}
    doc_tokens: list[list[str]] = []
    for t in texts:
        toks = tokenize(t or "")
        doc_tokens.append(toks)
        for tok in set(toks):
            if tok not in vocab:
                vocab[tok] = len(vocab)

    n = len(texts)
    doc_len = np.array([len(t) for t in doc_tokens], dtype=np.int32)
    tf_by_term: list[dict[int, int]] = [{} for _ in vocab]
    for d, toks in enumerate(doc_tokens):
        for tok in toks:
            m = vocab[tok]
            tf_by_term[m][d] = tf_by_term[m].get(d, 0) + 1
    postings = []
    for m in range(len(vocab)):
        if tf_by_term[m]:
            ids = np.array(sorted(tf_by_term[m].keys()), dtype=np.int32)
            tfs = np.array([tf_by_term[m][i] for i in ids], dtype=np.float32)
            postings.append((ids, tfs))
        else:
            postings.append((np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
    return {
        "vocab": vocab,
        "postings": postings,
        "doc_len": doc_len,
        "avgdl": max(float(doc_len.mean()), 1.0),
        "n": n,
        "k1": 1.5,
        "b": 0.75,
    }


def save_lex(lex: dict, path: Path = LEX_PATH) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lex, f, protocol=4)


def load_lex(path: Path = LEX_PATH) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def lex_matches(n_rows: int) -> bool:
    import pickle

    if not LEX_PATH.exists():
        return False
    try:
        with open(LEX_PATH, "rb") as f:
            return int(pickle.load(f)["n"]) == n_rows
    except Exception:  # noqa: BLE001
        return False


def bm25_scores(lex: dict, tokens: list[str]) -> "np.ndarray":
    """Standard BM25 over the whole corpus (millisecond-scale for 24k docs)."""
    import numpy as np

    n = lex["n"]
    k1, b, avgdl = lex["k1"], lex["b"], lex["avgdl"]
    doc_len = lex["doc_len"]
    scores = np.zeros(n, dtype=np.float32)
    for tok in dict.fromkeys(tokens):  # dedupe, keep order
        m = lex["vocab"].get(tok)
        if m is None:
            continue
        ids, tfs = lex["postings"][m]
        if len(ids) == 0:
            continue
        idf = np.log(1.0 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
        tf = np.zeros(n, dtype=np.float32)
        tf[ids] = tfs
        denom = tf + k1 * (1.0 - b + b * doc_len / avgdl)
        scores += np.where(tf > 0, idf * tf * (k1 + 1.0) / denom, 0.0)
    return scores


def hybrid_search(index: np.ndarray, query_vec, lex: dict, query_tokens: list[str],
                  top_k: int = 25, mask: np.ndarray | None = None,
                  strict_vecs=None, rrf_k: int = 60):
    """Fused dense + BM25 top-k via reciprocal rank fusion.

    ``strict_vecs`` (optional): per-keyword query vectors; when given, the
    dense score becomes the geometric mean of the per-keyword cosines, so
    every keyword must be present for a passage to rank high (kills the
    "flood control" -> generic "control" dilution).  Returns
    ``(top_idx, dense_display_scores)``.
    """
    import numpy as np

    scores = index @ query_vec
    if strict_vecs:
        mat = np.stack([index @ v for v in strict_vecs], axis=1)  # (N, T)
        scores = np.exp(np.mean(np.log(np.clip(mat, 1e-3, None)), axis=1))
    if mask is not None and (~mask).any():
        scores = np.where(mask, scores, -np.inf)

    bm = bm25_scores(lex, query_tokens)
    if mask is not None:
        bm = np.where(mask, bm, 0.0)

    rank_dense = np.argsort(np.argsort(-scores, kind="stable"))
    rank_bm = np.full(len(scores), 10**9, dtype=np.int64)
    for r, d in enumerate(np.argsort(-bm, kind="stable")):
        if bm[d] > 0:
            rank_bm[d] = r

    rrf = (1.0 / (rrf_k + rank_dense.astype(np.float32))
           + 1.0 / (rrf_k + rank_bm.astype(np.float32)))
    if mask is not None:
        rrf = np.where(mask, rrf, -np.inf)

    if mask is not None:
        k = min(top_k, int(mask.sum()))
    else:
        k = min(top_k, len(rrf))
    if k <= 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    top = np.argpartition(-rrf, k - 1)[:k]
    top = top[np.argsort(-rrf[top])]
    return top, np.round(scores[top], 4)
