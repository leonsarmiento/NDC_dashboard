#!/usr/bin/env python3
"""Build the EmbeddingGemma-2 index over the extraction rows (local, one-off).

Embeds every row of the extraction dataframe (the same rows the dashboard
shows) and stores the normalised matrix in ``data/embeddings.npz``, which is
committed to the repo so the deployed dashboard can search without
re-embedding on a CPU-only box. Also (re)builds the BM25 keyword index
(``data/bm25_index.pkl``).

Run (macOS, conda env with sentence-transformers -- uses MPS if available):

    python build_embeddings.py            # full rebuild (~20-40 min)
    python build_embeddings.py --lex-only # BM25 index only (seconds, no model)

Both index files store a sha256 digest of the passage texts (and the npz
also one of the row provenance), so the dashboard detects a content or row
change even when the row count is unchanged.

The row order of the index is defined by ``export_extractions.load_rows``
(sorted per-document JSONs, passages in document order) -- the dashboard
relies on it, so do not change the loader without rebuilding.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import numpy as np  # noqa: E402

import nl_search  # noqa: E402  (sets KMP_DUPLICATE_LIB_OK before torch)
from export_extractions import load_rows, load_rows_from_archive  # noqa: E402


def get_df():
    folder = APP_DIR / "all_docs_out"
    if folder.exists():
        return load_rows(folder)
    return load_rows_from_archive(APP_DIR / "data" / "extractions.tar.gz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lex-only", action="store_true",
                    help="rebuild only the BM25 index (no model, seconds)")
    args = ap.parse_args()

    t0 = time.time()
    print("loading extraction rows ...")
    df = get_df()
    if df.empty:
        print("no rows found", file=sys.stderr)
        return 1
    print(f"  {len(df):,} rows from {time.time()-t0:.0f}s")

    texts, embedded_chars, raw_chars = nl_search.build_index_texts(df)
    n_empty = int((embedded_chars == 0).sum())
    n_trunc = int((raw_chars > nl_search.TEXT_BUDGET_CHARS).sum())
    print(f"  embedded chars: median {int(np.median(embedded_chars))}, "
          f"p99 {int(np.percentile(embedded_chars, 99))}, max {int(embedded_chars.max())}; "
          f"pre-truncation max {int(raw_chars.max())}")
    print(f"  truncated at {nl_search.TEXT_BUDGET_CHARS} chars: {n_trunc} rows; "
          f"no text at all (forced -inf at search time): {n_empty} rows")
    text_digest, prov_digest = nl_search.corpus_digests(df)
    print(f"  text digest:      {text_digest[:16]}…")
    print(f"  provenance digest: {prov_digest[:16]}…")

    print("building lexical (BM25) index ...")
    lex = nl_search.build_lex_index(texts)
    nl_search.save_lex(lex, digest=text_digest)
    lex_mb = nl_search.LEX_PATH.stat().st_size / 1e6
    print(f"saved {nl_search.LEX_PATH}  ({lex_mb:.1f} MB, {len(lex['vocab']):,} terms)")

    if args.lex_only:
        print("done (lex-only; data/embeddings.npz untouched).")
        return 0

    print(f"loading model {nl_search.MODEL_ID} ...")
    model = nl_search.load_model()

    print(f"encoding {len(texts):,} passages (batch 32) ...")
    t1 = time.time()
    vectors = model.encode(
        texts,
        prompt_name=nl_search.PROMPT_DOC,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dt = time.time() - t1
    print(f"encoded in {dt/60:.1f} min  ({len(texts)/dt:.0f} passages/s)")

    # sanity: every non-empty text should produce a non-zero vector
    nonempty = embedded_chars > 0
    ok = (np.linalg.norm(vectors[nonempty], axis=1) > 0).all() if nonempty.any() else True
    print("all non-empty rows have non-zero vectors:", ok)
    if n_empty:
        print(f"note: {n_empty} rows have no text; they are kept (so row order is "
              "stable) but forced to -inf at search time")

    nl_search.save_index(
        nl_search.INDEX_PATH, vectors, n_rows=len(df), dim=vectors.shape[1],
        text_digest=text_digest, prov_digest=prov_digest,
        empty_text=(embedded_chars == 0), embedded_chars=embedded_chars,
    )
    size_mb = nl_search.INDEX_PATH.stat().st_size / 1e6
    print(f"saved {nl_search.INDEX_PATH}  ({size_mb:.1f} MB, {vectors.shape[1]}d float16, "
          f"version {nl_search.INDEX_VERSION})")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS, KB on Linux
    rss_mb = rss / 1048576 if sys.platform == "darwin" else rss / 1024
    print(f"peak RSS: {rss_mb:,.0f} MB")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
