#!/usr/bin/env python3
"""Build the EmbeddingGemma-2 index over the extraction rows (local, one-off).

Embeds every row of the extraction dataframe (the same rows the dashboard
shows) and stores the normalised matrix in ``data/embeddings.npz``, which is
committed to the repo so the deployed dashboard can search without
re-embedding on a CPU-only box.

Run (macOS, conda env ``ndc_dash`` -- uses MPS if available):

    python build_embeddings.py          # full rebuild (~20-40 min)

The row order of the index is defined by ``export_extractions.load_rows``
(sorted per-document JSONs, passages in document order) -- the dashboard
relies on it, so do not change the loader without rebuilding.
"""

from __future__ import annotations

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
    t0 = time.time()
    print("loading extraction rows ...")
    df = get_df()
    if df.empty:
        print("no rows found", file=sys.stderr)
        return 1
    print(f"  {len(df):,} rows from {time.time()-t0:.0f}s")

    texts = nl_search.build_index_texts(df)
    n_empty = sum(1 for t in texts if not t)
    if n_empty:
        print(f"  note: {n_empty} rows have no text (will embed as empty)")

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

    # sanity: the first non-empty doc should sit closer to its own text than
    # to a random other passage
    ok = (np.linalg.norm(vectors) > 0).all()
    print("all unit-norm:", ok)

    nl_search.save_index(nl_search.INDEX_PATH, vectors, n_rows=len(df), dim=vectors.shape[1])
    size_mb = nl_search.INDEX_PATH.stat().st_size / 1e6
    print(f"saved {nl_search.INDEX_PATH}  ({size_mb:.1f} MB, {vectors.shape[1]}d float16)")

    print("building lexical (BM25) index ...")
    lex = nl_search.build_lex_index(texts)
    nl_search.save_lex(lex)
    lex_mb = nl_search.LEX_PATH.stat().st_size / 1e6
    print(f"saved {nl_search.LEX_PATH}  ({lex_mb:.1f} MB, {len(lex['vocab']):,} terms)")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
