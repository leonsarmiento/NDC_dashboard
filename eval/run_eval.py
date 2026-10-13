#!/usr/bin/env python3
"""Evaluation harness for the "Ask the NDCs" search.

Loads the labelled query set (``eval/queries.json``) and scores three
retrieval configurations against it:

  * ``semantic``      -- dense cosine only
  * ``hybrid``        -- dense + BM25 fused (RRF)
  * ``hybrid+strict`` -- as hybrid, with the per-keyword geometric-mean dense
                         channel (the dashboard's "Strict: all keywords")

Metrics per query (expected set = hand-labelled source files):

  * precision@10  -- share of the top 10 rows whose source_file is expected
  * recall@25     -- share of expected files appearing in the top 25 rows
  * MRR           -- reciprocal rank of the first expected-file row (0 if none)

Everything except the query embedding is CPU/numpy, so once the model is
loaded the run takes seconds.

Usage:

    python eval/run_eval.py                 # all three configs, dashboard defaults
    python eval/run_eval.py --no-cap --no-mmr          # ablate post-fusion steps
    python eval/run_eval.py --confidence               # auditor-confidence prior
    python eval/run_eval.py --w-dense 1 --w-lex 1      # unweighted RRF
    python eval/run_eval.py --out results.json

Run before and after any search change (stemming, weights, truncation, ...)
so the tuning is measurable instead of guesswork.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import numpy as np  # noqa: E402

import nl_search  # noqa: E402  (sets KMP_DUPLICATE_LIB_OK before torch)
from export_extractions import load_rows, load_rows_from_archive  # noqa: E402


# --- data ------------------------------------------------------------------- #
def get_df() -> "pd.DataFrame":
    folder = APP_DIR / "all_docs_out"
    if folder.exists():
        return load_rows(folder)
    return load_rows_from_archive(APP_DIR / "data" / "extractions.tar.gz")


# --- one search call, version-agnostic --------------------------------------- #
def run_one(index, lex, qvec, strict_vecs, query, top_k, mask,
            doc_cap, use_mmr, w_dense, w_lex, use_confidence,
            doc_ids, confidence, dead_mask):
    """One search under one configuration -> (top_idx, files).

    Works against both the old hybrid_search API (positional query_tokens,
    returns (idx, dense_scores)) and the new one (query string, fusion
    options, returns (idx, fused, dense, kw_hits)).
    """
    new_api = hasattr(nl_search, "lex_query_terms")
    if new_api:
        top, _fused, _dense, _hits = nl_search.hybrid_search(
            index, qvec, lex, query, top_k=top_k, mask=mask,
            strict_vecs=strict_vecs, doc_cap=doc_cap, mmr=use_mmr,
            w_dense=w_dense, w_lex=w_lex,
            confidence=confidence if use_confidence else None,
            doc_ids=doc_ids, dead_mask=dead_mask,
        )
    else:
        top, _dense = nl_search.hybrid_search(
            index, qvec, lex, nl_search.tokenize(query),
            top_k=top_k, mask=mask, strict_vecs=strict_vecs,
        )
    return top


def run_semantic(index, qvec, top_k, mask):
    return nl_search.search(index, qvec, top_k=top_k, mask=mask)


# --- metrics ---------------------------------------------------------------- #
def metrics_for(top: np.ndarray, files: list[str], expected: set[str]) -> dict:
    files = [f for f in files]
    top10 = files[:10]
    top25 = files[:25]
    p10 = (sum(1 for f in top10 if f in expected) / len(top10)) if top10 else 0.0
    r25 = len({f for f in top25 if f in expected}) / len(expected) if expected else 0.0
    mrr = 0.0
    for i, f in enumerate(files, 1):
        if f in expected:
            mrr = 1.0 / i
            break
    return {"p10": p10, "r25": r25, "mrr": mrr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--no-cap", action="store_true",
                    help="disable the per-document cap (new API only)")
    ap.add_argument("--no-mmr", action="store_true", help="disable MMR (new API only)")
    ap.add_argument("--confidence", action="store_true",
                    help="apply the auditor relevance-confidence bonus (off by "
                         "default, matching the dashboard)")
    ap.add_argument("--w-dense", type=float,
                    default=getattr(nl_search, "W_DENSE", 0.6))
    ap.add_argument("--w-lex", type=float,
                    default=getattr(nl_search, "W_LEX", 0.4))
    ap.add_argument("--out", type=str, default=None,
                    help="also dump the per-query results to this JSON file")
    args = ap.parse_args()

    t0 = time.time()
    queries = json.loads((APP_DIR / "eval" / "queries.json").read_text(encoding="utf-8"))
    print(f"loading dataframe ...")
    df = get_df()
    files_all = list(df["source_file"])
    index, meta = nl_search.load_index()
    lex = nl_search.load_lex()
    dead_mask = np.array(meta.get("empty_text", [False] * len(df)), dtype=bool)
    doc_ids = np.array(files_all, dtype=object)
    confidence = df["passage_relevance_confidence"].fillna("NA").to_numpy()

    # sanity: every labelled file must exist in the corpus
    known = set(files_all)
    for q in queries:
        missing = [f for f in q["expected"] if f not in known]
        if missing:
            print(f"!! {q['id']}: unknown source_file(s): {missing}", file=sys.stderr)
    print(f"loaded {len(df):,} rows, {len(queries)} queries  ({time.time()-t0:.0f}s)")

    doc_cap = 0 if args.no_cap else 4
    print(f"\nconfig: doc_cap={doc_cap} mmr={not args.no_mmr} "
          f"confidence={args.confidence} "
          f"w_dense={args.w_dense} w_lex={args.w_lex}")

    model = nl_search.load_model()

    per_query = []
    config_results = {}
    for cfg in ("semantic", "hybrid", "hybrid+strict"):
        config_results[cfg] = {}
        t1 = time.time()
        for q in queries:
            query = q["query"]
            expected = set(q["expected"])
            qvec = nl_search.embed_query(model, query)
            strict_vecs = None
            if cfg == "hybrid+strict":
                terms = nl_search.content_terms(query)
                if len(terms) >= 2:
                    strict_vecs = [nl_search.embed_query(model, t) for t in terms]
            if cfg == "semantic":
                top = run_semantic(index, qvec, args.top_k, None)
            else:
                top = run_one(index, lex, qvec, strict_vecs, query, args.top_k, None,
                              doc_cap, not args.no_mmr, args.w_dense, args.w_lex,
                              args.confidence, doc_ids, confidence, dead_mask)
            files = [files_all[i] for i in top]
            m = metrics_for(top, files, expected)
            config_results[cfg][q["id"]] = m
            per_query.append({"id": q["id"], "category": q.get("category", ""),
                              "query": query, cfg: m})
        dt = time.time() - t1
        print(f"  {cfg:14s} done in {dt:.1f}s")

    # aggregate
    rows = []
    for q in queries:
        row = {"id": q["id"], "category": q.get("category", "")}
        for cfg in ("semantic", "hybrid", "hybrid+strict"):
            row[cfg] = config_results[cfg][q["id"]]
        rows.append(row)

    def agg(subset):
        out = {}
        for cfg in ("semantic", "hybrid", "hybrid+strict"):
            out[cfg] = {
                "p10": np.mean([r[cfg]["p10"] for r in subset]),
                "r25": np.mean([r[cfg]["r25"] for r in subset]),
                "mrr": np.mean([r[cfg]["mrr"] for r in subset]),
            }
        return out

    print("\n== per-query ==")
    hdr = f"{'id':6s} {'cat':13s}"
    for cfg in ("semantic", "hybrid", "hybrid+strict"):
        hdr += f"  {cfg+' P10/R25/MRR':>22s}"
    print(hdr)
    for row in rows:
        line = f"{row['id']:6s} {row['category']:13s}"
        for cfg in ("semantic", "hybrid", "hybrid+strict"):
            m = row[cfg]
            line += f"  {m['p10']:6.2f}/{m['r25']:6.2f}/{m['mrr']:6.2f}"
        print(line)

    print("\n== aggregate (mean over queries) ==")
    alla = agg(rows)
    print(f"{'config':16s} {'precision@10':>13s} {'recall@25':>11s} {'MRR':>7s}")
    for cfg, m in alla.items():
        print(f"{cfg:16s} {m['p10']:13.3f} {m['r25']:11.3f} {m['mrr']:7.3f}")

    cats = sorted({r["category"] for r in rows if r["category"]})
    print("\n== by category ==")
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        a = agg(sub)
        print(f"  {cat:13s} " + "  ".join(
            f"{cfg}: P10 {a[cfg]['p10']:.2f} R25 {a[cfg]['r25']:.2f} MRR {a[cfg]['mrr']:.2f}"
            for cfg in a))

    if args.out:
        payload = {"config": {"doc_cap": doc_cap, "mmr": not args.no_mmr,
                              "confidence": args.confidence,
                              "w_dense": args.w_dense, "w_lex": args.w_lex},
                   "aggregate": alla, "per_query": rows}
        Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nper-query results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
