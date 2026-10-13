#!/usr/bin/env python3
"""Natural-language search over the NDC extractions (EmbeddingGemma 2).

Shared by:
  * ``build_embeddings.py``  -- one-off local build of the embedding index
  * ``ndc_explorer.py``      -- the "Ask the NDCs" dashboard page
  * ``eval/run_eval.py``     -- labelled-query evaluation harness

Design
------
Every extraction row is embedded once (with the model's ``Document`` prompt)
and stored in ``data/embeddings.npz`` (L2-normalised, float16, same row
order as :func:`export_extractions.load_rows`).  A natural-language query is
embedded with the ``SearchQuery`` prompt and scored by cosine similarity
(numpy matmul, a few ms for 24k rows).

The search is **hybrid**: the dense ranking is fused with a BM25 keyword
ranking via weighted reciprocal rank fusion (1-based ranks):

    score = W_DENSE/(RRF_K + rank_dense) + W_LEX/(RRF_K + rank_bm25)

followed by a small, documented relevance-confidence bonus, a per-document
cap (diversity) and an optional MMR re-rank.  See ``hybrid_search``.

The npz stores content digests (sha256 of the assembled texts and of the
row provenance); :func:`index_matches_df` checks them against the current
dataframe so a same-row-count content change cannot silently misalign the
vectors with the wrong rows.

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

# Bump whenever doc_text() or the embedding setup changes, so a rebuild is
# forced (index_matches_df compares it).
INDEX_VERSION = 2
# Per-passage token cap.  EmbeddingGemma's own ceiling is 1024 tokens, so
# anything at or above that is equivalent; the tokenizer truncates the rest.
MAX_SEQ_LEN = 1282  # hard cap per passage (>= model max; no-op above 1024)

# Character budget for the embedded text.  Measured over the shipped corpus:
# median 345 chars, p90 895, p99 1,841, max 10,557 -- ~138 rows exceed this.
# doc_text puts the highest-signal fields first, so truncation here cuts the
# lowest-signal tail (notes / tail of original text), not the target values.
TEXT_BUDGET_CHARS = 2500

# --- fusion constants (see hybrid_search) ----------------------------------- #
RRF_K = 60
W_DENSE = 0.6
W_LEX = 0.4
DEFAULT_DOC_CAP = 4          # max passages per source_file in the returned set
MMR_POOL = 200               # candidates re-scored by MMR
MMR_LAMBDA = 0.7             # 0.7 * relevance - 0.3 * max cosine to selected
# Documented, cheap relevance prior (NOT a hidden quality judgement): rows the
# extractor marked Low-confidence are demoted slightly, High slightly boosted.
CONFIDENCE_BONUS = {"High": 1.05, "Medium": 1.0, "Low": 0.9}


# --- text assembly ---------------------------------------------------------- #
def _clean(v) -> "str | None":
    s = str(v).strip()
    return s if s and s.upper() != "NA" else None


def doc_text(row) -> str:
    """Human-meaningful text for one extraction row, truncated to budget.

    See :func:`doc_text_unclipped` for the assembly order and rationale.
    """
    return doc_text_unclipped(row)[:TEXT_BUDGET_CHARS]


def doc_text_unclipped(row) -> str:
    """Assemble the per-row embedded text (no truncation).

    Order is the one from the search-improvements rec (and the eval set:
    on it this ordering gives the best hybrid MRR 0.52 and the best
    strict-mode MRR 0.59): commitment type, numeric target + unit, English
    translation, original-language text, then notes.  Highest-signal
    content first, because the tail is what gets truncated.  Truncation is
    done deliberately by the caller (:func:`doc_text`) at TEXT_BUDGET_CHARS
    instead of relying on the tokenizer.  The country name is *not* mixed
    in: country scoping is done by the structured iso3 field after the
    cosine search.
    """
    parts: list[str] = []
    ct = _clean(row.get("passage_commitment_type"))
    if ct:
        parts.append(f"Commitment type: {ct}")
    tv = _clean(row.get("passage_target_value"))
    if tv:
        unit = _clean(row.get("passage_target_unit")) or ""
        parts.append(f"Target: {tv} {unit}".strip())
    for k in ("passage_english_translation", "passage_original_text", "passage_notes"):
        v = _clean(row.get(k))
        if v:
            parts.append(v)
    return "\n".join(parts)


def build_index_texts(df):
    """Per-row (document strings, embedded char counts, raw char counts).

    Row order = embed order.  ``raw_chars`` are the lengths before the
    TEXT_BUDGET_CHARS cut, so the build log can report how many rows were
    truncated; ``embedded_chars`` is what actually went into the model.
    """
    import numpy as np

    recs = df.to_dict("records")
    texts = [doc_text(r) for r in recs]
    embedded_chars = np.array([len(t) for t in texts], dtype=np.int32)
    raw_chars = np.array([len(doc_text_unclipped(r)) for r in recs], dtype=np.int32)
    return texts, embedded_chars, raw_chars


# --- content digests (index <-> data alignment) ----------------------------- #
def corpus_digests(df):
    """(text_digest, provenance_digest) for the current dataframe.

    text_digest:      sha256 of the joined per-row embedded texts
    provenance_digest: sha256 of the ordered (source_file, page, location)
                       triples -- detects a re-ordered dataframe even when
                       the row count and the multiset of texts are unchanged.
    """
    import hashlib

    texts, _, _ = build_index_texts(df)
    text_digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    prov = (
        df["source_file"].fillna("").astype(str).to_numpy() + "\t"
        + df["passage_page"].fillna("").astype(str).to_numpy() + "\t"
        + df["passage_location_in_document"].fillna("").astype(str).to_numpy()
    )
    prov_digest = hashlib.sha256("\n".join(prov).encode("utf-8")).hexdigest()
    return text_digest, prov_digest


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
def save_index(path: Path, vectors, n_rows: int, dim: int,
               text_digest: str = "", prov_digest: str = "",
               empty_text=None, embedded_chars=None) -> None:
    """Store the normalised matrix (float16) plus provenance metadata."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        vectors=np.ascontiguousarray(vectors, dtype=np.float16),
        version=np.array(INDEX_VERSION),
        n_rows=np.array(n_rows),
        dim=np.array(dim),
        model=np.array(MODEL_ID),
        prompt=np.array(PROMPT_DOC),
        text_digest=np.array(text_digest),
        prov_digest=np.array(prov_digest),
    )
    if empty_text is not None:
        payload["empty_text"] = np.asarray(empty_text, dtype=bool)
    if embedded_chars is not None:
        payload["embedded_chars"] = np.asarray(embedded_chars, dtype=np.int32)
    np.savez_compressed(path, **payload)


def load_index(path: Path = INDEX_PATH):
    """Load the index -> (vectors float16, meta dict) or FileNotFoundError.

    The matrix is kept float16 (halves memory); the query matmul promotes to
    float32 automatically (f16 @ f32 -> f32).
    ``meta`` carries version/n_rows/dim/model plus the content digests and,
    when present, the ``empty_text`` boolean array (rows with no text at all
    -- their score is forced to -inf at search time).
    """
    import numpy as np

    z = np.load(path, allow_pickle=False)
    meta = {
        "version": int(z["version"]),
        "n_rows": int(z["n_rows"]),
        "dim": int(z["dim"]),
        "model": str(z["model"]),
        "text_digest": str(z["text_digest"]) if "text_digest" in z else "",
        "prov_digest": str(z["prov_digest"]) if "prov_digest" in z else "",
    }
    if "empty_text" in z:
        meta["empty_text"] = z["empty_text"]
    else:
        meta["empty_text"] = np.zeros(z["n_rows"], dtype=bool)
    return z["vectors"], meta


def index_matches_df(df):
    """Check the index against the dataframe's *content*, not just row count.

    Returns ``(ok, reason)``: reason is "" when ok, otherwise a sentence
    saying exactly which part changed (version, row count, text, order).
    """
    import numpy as np

    if not INDEX_PATH.exists():
        return False, "data/embeddings.npz is missing"
    try:
        z = np.load(INDEX_PATH, allow_pickle=False)
        version = int(z["version"])
        n = int(z["n_rows"])
    except Exception as e:  # noqa: BLE001
        return False, f"the index file cannot be read ({type(e).__name__})"
    if version != INDEX_VERSION:
        return False, (f"index version {version} does not match the search code "
                       f"(version {INDEX_VERSION}) -- the text assembly or the "
                       "fusion changed, so a rebuild is required")
    if n != len(df):
        return False, (f"index holds {n:,} rows but the data has {len(df):,} "
                       "-- the extraction set changed")
    text_digest, prov_digest = corpus_digests(df)
    stored_text = str(z["text_digest"]) if "text_digest" in z else None
    stored_prov = str(z["prov_digest"]) if "prov_digest" in z else None
    if stored_text is not None and stored_text != text_digest:
        return False, "the passage texts changed (same row count, different content)"
    if stored_prov is not None and stored_prov != prov_digest:
        return False, "the row order changed (same texts, different order)"
    return True, ""


# --- lexical channel (BM25) + hybrid fusion --------------------------------- #
# Dense single-vector search blends multi-word queries into one direction in
# vector space, so "flood control" can match generic "control" passages.
# A cheap BM25 channel over the same passages (no model, precomputed locally)
# fused via weighted reciprocal-rank fusion fixes that class of problem.

STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
    "is", "are", "was", "will", "shall", "may", "can", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "as", "at", "from", "into",
    "per", "via", "which", "who", "whom", "our", "their", "they", "we",
    "there", "here", "more", "most", "other", "such", "than", "then", "also",
    "only", "very", "de", "la", "le", "les", "du", "des", "el", "en", "que",
    "pour", "par", "au", "aux", "para", "por", "los", "las", "una", "un",
})

# Suffix stripping (longest first, single pass, min stem length 3).  Keeps
# "reforestation"/"reforested"/"reforest" in one equivalence class so the
# lexical channel catches morphological variants.  One pass only: enough for
# this corpus, and it avoids the classic over-stemming of real short words.
_SUFFIXES = ("sions", "ation", "tion", "ment", "ity", "ally", "ing", "ed", "es")
_MIN_STEM = 3


def _strip_accents(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on non-alphanumerics (len >= 2).

    These are the *raw* tokens (used for the "keyword hits" display).  The
    BM25 index and its query terms use :func:`stem` on top.
    """
    import re

    return [t for t in re.findall(r"[a-z0-9]+", _strip_accents(text)) if len(t) >= 2]


def stem(term: str) -> str:
    """Light, dependency-free suffix stripping (min stem length 3)."""
    for suf in _SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= _MIN_STEM:
            return term[: -len(suf)]
    return term


def normalize_tokens(text: str) -> list[str]:
    """Raw tokens with light stemming -- the term set of the BM25 index."""
    return [stem(t) for t in tokenize(text)]


def content_terms(query: str) -> list[str]:
    """Meaningful *raw* query terms (no stopwords, no pure numbers, len >= 3).

    The single source of truth for "which words does the user care about":
    used both for the BM25 query terms (stemmed on top, see
    :func:`lex_query_terms`) and for the strict-mode per-keyword embeddings.
    """
    return [
        t for t in tokenize(query)
        if t not in STOPWORDS and not t.isdigit() and len(t) >= 3
    ]


def build_lex_index(texts: list[str]) -> dict:
    """BM25 posting lists for the whole corpus (same row order as the index).

    Built on *normalised* terms (stemmed), so morphological variants of a
    query word share one posting list.
    """
    import numpy as np

    vocab: dict[str, int] = {}
    doc_tokens: list[list[str]] = []
    for t in texts:
        toks = normalize_tokens(t or "")
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


def save_lex(lex: dict, path: Path = LEX_PATH, digest: str = "") -> None:
    import pickle

    if digest:
        lex = {**lex, "digest": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(lex, f, protocol=4)


def load_lex(path: Path = LEX_PATH) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def lex_matches(lex: dict, n_rows: int, text_digest: str = "") -> "tuple[bool, str]":
    """BM25 index vs current data: same rows, and (if both sides carry a
    digest) the same text content."""
    if lex is None:
        return False, "the keyword (BM25) index is missing"
    if int(lex["n"]) != n_rows:
        return False, (f"the keyword index holds {int(lex['n']):,} rows but the "
                       f"data has {n_rows:,}")
    stored = lex.get("digest", "")
    if stored and text_digest and stored != text_digest:
        return False, "the passage texts changed; rebuild the keyword index"
    return True, ""


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


def lex_query_terms(lex: dict, query: str) -> list[str]:
    """BM25 query terms: content terms, stemmed, high-df terms dropped.

    Same content-term set as strict mode (see :func:`content_terms`), so both
    channels agree on which words the user cares about.  Terms present in
    more than 40% of the documents carry no discriminative signal and are
    dropped (they only add noise).
    """
    n = lex["n"]
    out: list[str] = []
    for t in dict.fromkeys(stem(x) for x in content_terms(query)):
        m = lex["vocab"].get(t)
        if m is None:
            continue
        ids, _ = lex["postings"][m]
        if len(ids) > 0.4 * n:
            continue
        out.append(t)
    return out


def kw_hits_for(lex: dict, query: str, idx: np.ndarray) -> "np.ndarray":
    """For the given rows: how many of the query's content terms (stemmed)
    appear in the passage.  Cheap (postings lookups) and the most intuitive
    explanation of why a row shows up."""
    import numpy as np

    hits = np.zeros(lex["n"], dtype=np.int32)
    for t in dict.fromkeys(stem(x) for x in content_terms(query)):
        m = lex["vocab"].get(t)
        if m is None:
            continue
        ids, _ = lex["postings"][m]
        if len(ids):
            hits[ids] += 1
    return hits[idx]


def row_kw_terms(lex: dict, query: str, row: int) -> list[str]:
    """The query's content terms (raw form) actually present in one row --
    for the "why did this match?" line in the result card."""
    import numpy as np

    out: list[str] = []
    for t in dict.fromkeys(content_terms(query)):
        m = lex["vocab"].get(stem(t))
        if m is None:
            continue
        ids, _ = lex["postings"][m]
        if len(ids) and bool(np.searchsorted(ids, row) < len(ids)
                             and ids[np.searchsorted(ids, row)] == row):
            out.append(t)
    return out


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


def hybrid_search(index: np.ndarray, qvec, lex: dict, query: str,
                  top_k: int = 25, mask: np.ndarray | None = None,
                  strict_vecs=None, doc_cap: int = DEFAULT_DOC_CAP,
                  mmr: bool = True, w_dense: float = W_DENSE,
                  w_lex: float = W_LEX, rrf_k: int = RRF_K,
                  confidence: "np.ndarray | None" = None,
                  doc_ids: "np.ndarray | None" = None,
                  dead_mask: "np.ndarray | None" = None):
    """Fused dense + BM25 top-k via weighted reciprocal rank fusion.

    Pipeline:
      1. dense channel: cosine of the whole query, OR (``strict_vecs``) the
         geometric mean of per-keyword cosines with rows missing any keyword
         hard-excluded (cosine below the 1st percentile of positive cosines
         -> score -inf; no silent 1e-3 near-tie floor);
      2. BM25 channel over the query's stemmed content terms;
      3. RRF fusion with 1-based ranks: ``w_d/(k+rank_dense) + w_l/(k+rank_bm)`;
      4. small documented relevance-confidence bonus (High 1.05 / Low 0.9);
      5. optional MMR re-rank of the top ~200 (diversity, unit-norm dot);
      6. per-document cap (``doc_cap``) with backfill from the next ranked
         candidate (relaxed only if the cap would return fewer than top_k).

    ``dead_mask`` rows (e.g. rows with no text at all) are forced to -inf.

    Returns ``(top_idx, fused_display, dense_display, kw_hits)`` where
    fused_display is the actual ordering score rescaled to 0..1 (max = 1)
    and kw_hits is the number of query content terms present per row.
    """
    import numpy as np

    n = len(index)
    alive = np.ones(n, dtype=bool)
    if mask is not None:
        alive &= mask
    if dead_mask is not None and dead_mask.any():
        alive &= ~dead_mask
    if not alive.any():
        return (np.array([], dtype=int), np.array([], dtype=np.float32),
                np.array([], dtype=np.float32), np.array([], dtype=np.int32))

    # 1) dense channel ------------------------------------------------------ #
    if strict_vecs:
        # per-keyword cosines; NOT the whole-query matmul (avoids the
        # redundant index @ qvec when the strict channel is active)
        mat = np.stack([index @ v for v in strict_vecs], axis=1)  # (N, T) f32
        positive = mat[mat > 0]
        thr = float(np.percentile(positive, 1)) if len(positive) else 0.0
        excluded = (mat < thr).any(axis=1)  # a missing keyword is a hard miss
        dense = np.where(excluded, 0.0,
                         np.exp(np.mean(np.log(np.where(mat > 0, mat, 1.0)), axis=1)))
    else:
        dense = index @ qvec
    dense = np.where(alive, dense, -np.inf)

    # 2) BM25 channel -------------------------------------------------------- #
    bm = bm25_scores(lex, lex_query_terms(lex, query))
    bm = np.where(alive, bm, 0.0)

    # 3) RRF fusion, 1-based ranks ------------------------------------------- #
    rank_dense = np.argsort(np.argsort(-dense, kind="stable")).astype(np.float32) + 1.0
    rank_bm = np.full(n, np.inf, dtype=np.float32)
    bm_order = np.argsort(-bm, kind="stable")
    bm_ranked = bm_order[bm[bm_order] > 0]
    if len(bm_ranked):
        rank_bm[bm_ranked] = np.arange(1, len(bm_ranked) + 1, dtype=np.float32)
    rrf = w_dense / (rrf_k + rank_dense) + w_lex / (rrf_k + rank_bm)

    # 4) documented relevance-confidence bonus -------------------------------- #
    score = rrf
    if confidence is not None:
        bonus = np.ones(n, dtype=np.float32)
        for label, b in CONFIDENCE_BONUS.items():
            bonus[confidence == label] = b
        score = score * bonus
    score = np.where(alive, score, -np.inf)

    # 5) candidate ordering + optional MMR ------------------------------------- #
    order = np.argsort(-score, kind="stable")
    order = order[score[order] > -np.inf]
    pool = order[: max(MMR_POOL, top_k)]
    if mmr and len(pool) > 1:
        f = score[pool]
        fmax = f.max()
        fn = f / fmax if fmax > 0 else np.zeros_like(f)
        S = index[pool].astype(np.float32)
        sim = S @ S.T
        mmr_order: list[int] = []
        avail = np.ones(len(pool), dtype=bool)
        picked: list[int] = []
        for _ in range(len(pool)):
            if not avail.any():
                break
            if picked:
                maxsim = sim[picked, :].max(axis=0)
            else:
                maxsim = np.zeros(len(pool), dtype=np.float32)
            cand = np.where(avail, MMR_LAMBDA * fn - (1 - MMR_LAMBDA) * maxsim,
                            -np.inf)
            i = int(np.argmax(cand))
            picked.append(i)
            avail[i] = False
            mmr_order.append(int(pool[i]))
        pool = np.array(mmr_order, dtype=int)

    # 6) per-document cap with backfill --------------------------------------- #
    k = min(top_k, int(alive.sum()))
    selected: list[int] = []
    seen: set[int] = set()
    doc_count: dict[str, int] = {}

    def _take(i: int, enforce_cap: bool) -> None:
        nonlocal doc_count
        d = str(doc_ids[i]) if doc_ids is not None else ""
        if enforce_cap and doc_cap and doc_count.get(d, 0) >= doc_cap:
            return False
        selected.append(i)
        seen.add(int(i))
        doc_count[d] = doc_count.get(d, 0) + 1
        return True

    use_cap = bool(doc_cap) and doc_ids is not None
    for i in pool:
        if len(selected) >= k:
            break
        if int(i) in seen:
            continue
        if not _take(i, enforce_cap=use_cap):
            continue
    if len(selected) < k:  # backfill from the remaining fusion-ranked rows
        for i in order:
            if len(selected) >= k:
                break
            if int(i) in seen:
                continue
            if use_cap and doc_cap and doc_count.get(str(doc_ids[i]), 0) >= doc_cap:
                continue
            _take(i, enforce_cap=False)
    if len(selected) < k and use_cap:  # not enough documents: relax the cap
        for i in order:
            if len(selected) >= k:
                break
            if int(i) in seen:
                continue
            _take(i, enforce_cap=False)

    if not selected:
        return (np.array([], dtype=int), np.array([], dtype=np.float32),
                np.array([], dtype=np.float32), np.array([], dtype=np.int32))

    # Keep selection order: the MMR pool order when MMR is on (that IS the
    # diversity re-rank), pure fusion order otherwise.  Re-sorting by score
    # here would silently discard the MMR ordering.
    top = np.array([int(i) for i in selected], dtype=int)
    s = score[top]
    fused_display = (s / s.max()) if s.max() > 0 else np.zeros_like(s)
    dense_display = np.where(np.isfinite(dense[top]), dense[top], 0.0)
    hits = kw_hits_for(lex, query, top)
    return top, np.round(fused_display, 4), np.round(dense_display, 4), hits
