# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from common import ARTIFACT_DIR, DATA_PATH, chunks_to_documents, default_config, load_chunks, make_qa_items
from src.rag_system import RAGSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one end-to-end RAG query and print methodology metadata.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--index-prefix", default=str(ARTIFACT_DIR / "hotpot_bgem3_index"))
    parser.add_argument("--graph-path", default=str(ARTIFACT_DIR / "hotpot_subset_graph.pkl"))
    parser.add_argument("--qa-index", type=int, default=0)
    parser.add_argument("--query", default=None)
    parser.add_argument("--strict", action="store_true", default=True)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.data))
    docs = chunks_to_documents(chunks)
    qa_items = make_qa_items(chunks)
    item = qa_items[args.qa_index] if args.query is None else None
    query = args.query or item["question"]

    rag = RAGSystem(default_config(strict=args.strict))
    if not rag.bgem3_retriever.load_index(args.index_prefix):
        raise RuntimeError(f"Index not found: {args.index_prefix}. Run experiments/02_build_index.py first.")
    # Keep docs available when the index is freshly built by another process format.
    if not rag.bgem3_retriever.documents:
        rag.index_documents(docs)
    rag.load_global_graph(args.graph_path)

    result = rag.query(query)
    print("=" * 80)
    if item:
        print(f"qa_id: {item['qa_id']}")
        print(f"gold answer: {item['answer']}")
        print(f"gold titles: {item['supporting_titles']}")
    print(f"query: {query}")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
    print("metadata:")
    for k, v in result.get("metadata", {}).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
