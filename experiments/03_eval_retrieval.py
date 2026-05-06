# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from common import ARTIFACT_DIR, BGE_M3_MODEL_NAME, DATA_PATH, append_jsonl, ensure_artifact_dir, load_chunks, make_qa_items
from src.bgem3_retriever import BGEM3Retriever


def hit_at_k(pred_titles: List[str], gold_titles: List[str], k: int) -> bool:
    pred = {x for x in pred_titles[:k] if x}
    gold = {x for x in gold_titles if x}
    return bool(pred & gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate coarse retrieval title recall on Hotpot chunks.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--index-prefix", default=str(ARTIFACT_DIR / "hotpot_bgem3_index"))
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-name", default=BGE_M3_MODEL_NAME)
    parser.add_argument("--out", default=str(ARTIFACT_DIR / "03_retrieval_eval.jsonl"))
    args = parser.parse_args()

    chunks = load_chunks(Path(args.data))
    qa_items = make_qa_items(chunks, limit=args.num_queries)
    cfg = {
        "model_name": args.model_name,
        "top_k_dense": 200,
        "top_k_sparse": 200,
        "top_k_final": args.top_k,
        "fusion_method": "rrf",
        "rrf_k": 60,
    }
    retriever = BGEM3Retriever(cfg)
    if not retriever.load_index(args.index_prefix):
        raise RuntimeError(f"Index not found: {args.index_prefix}. Run experiments/02_build_index.py first.")

    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()

    rows: List[Dict] = []
    hits = {5: 0, 10: 0, args.top_k: 0}
    for item in qa_items:
        docs = retriever.hybrid_doc_retrieval(item["question"], top_k_final=args.top_k)
        pred_titles = [str(d.get("title") or "") for d in docs]
        row = {
            "qa_id": item["qa_id"],
            "question": item["question"],
            "answer": item["answer"],
            "gold_titles": item["supporting_titles"],
            "pred_titles": pred_titles,
            "hit@5": hit_at_k(pred_titles, item["supporting_titles"], 5),
            "hit@10": hit_at_k(pred_titles, item["supporting_titles"], 10),
            f"hit@{args.top_k}": hit_at_k(pred_titles, item["supporting_titles"], args.top_k),
        }
        for k in hits:
            hits[k] += int(hit_at_k(pred_titles, item["supporting_titles"], k))
        rows.append(row)

    ensure_artifact_dir()
    append_jsonl(out_path, rows)
    summary = {f"recall@{k}": v / max(1, len(qa_items)) for k, v in hits.items()}
    print(f"Wrote {out_path}")
    print({"num_queries": len(qa_items), **summary})


if __name__ == "__main__":
    main()
