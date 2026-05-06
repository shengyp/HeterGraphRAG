# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from common import ARTIFACT_DIR, DATA_PATH, default_config, ensure_artifact_dir, load_chunks, make_qa_items, select_chunks_for_qa_ids
from HeterGraphRAG.rag_system import RAGSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="为早期 RAG 实验构建一个小规模严格方法论图。")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--num-qa", type=int, default=5)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--graph-path", default=str(ARTIFACT_DIR / "hotpot_subset_graph.pkl"))
    args = parser.parse_args()

    chunks = load_chunks(Path(args.data))
    qa_items = make_qa_items(chunks, limit=args.num_qa)
    qa_ids = [x["qa_id"] for x in qa_items]
    subset_chunks = select_chunks_for_qa_ids(chunks, qa_ids)

    cfg = default_config(strict=args.strict)
    ensure_artifact_dir()
    rag = RAGSystem(cfg)
    rag.build_global_graph(subset_chunks)
    rag.save_global_graph(args.graph_path)

    print(
        {
            "num_qa": len(qa_items),
            "num_chunks": len(subset_chunks),
            "graph_path": args.graph_path,
            "strict_methodology": args.strict,
        }
    )


if __name__ == "__main__":
    main()
