# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from common import ARTIFACT_DIR, BGE_M3_MODEL_NAME, DATA_PATH, chunks_to_documents, ensure_artifact_dir, load_chunks, make_qa_items, select_chunks_for_qa_ids
from src.bgem3_retriever import BGEM3Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or rebuild the BGE-M3 document index from hotpot_chunks.json.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument("--num-qa", type=int, default=None, help="Build an index over the first N QA groups; use this with 04_build_graph_subset.py.")
    parser.add_argument("--model-name", default=BGE_M3_MODEL_NAME)
    parser.add_argument("--index-prefix", default=str(ARTIFACT_DIR / "hotpot_bgem3_index"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.data), limit=args.limit_chunks)
    if args.num_qa is not None:
        qa_ids = [x["qa_id"] for x in make_qa_items(chunks, limit=args.num_qa)]
        chunks = select_chunks_for_qa_ids(chunks, qa_ids)
    docs = chunks_to_documents(chunks)
    ensure_artifact_dir()

    cfg = {
        "model_name": args.model_name,
        "use_fp16": True,
        "batch_size": args.batch_size,
        "top_k_dense": 200,
        "top_k_sparse": 200,
        "top_k_final": 20,
        "fusion_method": "rrf",
        "rrf_k": 60,
        "lambda_dense": 0.5,
        "lambda_sparse": 0.5,
    }

    print(f"chunks={len(chunks)} docs={len(docs)}")
    retriever = BGEM3Retriever(cfg)
    retriever.build_doc_index(docs)
    retriever.save_index(args.index_prefix)
    print(f"Wrote {args.index_prefix}.pkl")


if __name__ == "__main__":
    main()
