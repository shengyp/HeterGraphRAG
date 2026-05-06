# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import ARTIFACT_DIR, DATA_PATH, append_jsonl, chunks_to_documents, default_config, ensure_artifact_dir, load_chunks, make_qa_items
from HeterGraphRAG.rag_system import RAGSystem


def configure_ablation(cfg: Dict[str, Any], ablation: str) -> None:
    if ablation == "ours":
        return
    if ablation == "no_strict":
        cfg["strict_methodology"] = False
        cfg["graph_construction"]["strict_methodology"] = False
        cfg["intent_representation"]["strict_methodology"] = False
        return
    if ablation == "less_semantic":
        cfg["structural_centrality"]["gamma_sem"] = 0.0
        return
    if ablation == "more_semantic":
        cfg["structural_centrality"]["gamma_sem"] = 1.0
        return
    raise ValueError(f"未知消融实验配置: {ablation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行小规模或批量端到端 RAG 实验。")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--index-prefix", default=str(ARTIFACT_DIR / "hotpot_bgem3_index"))
    parser.add_argument("--graph-path", default=str(ARTIFACT_DIR / "hotpot_subset_graph.pkl"))
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--ablation", default="ours", choices=["ours", "no_strict", "less_semantic", "more_semantic"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    chunks = load_chunks(Path(args.data))
    docs = chunks_to_documents(chunks)
    qa_items = make_qa_items(chunks, limit=args.num_queries)
    cfg = default_config(strict=args.strict)
    configure_ablation(cfg, args.ablation)

    rag = RAGSystem(cfg)
    if not rag.bgem3_retriever.load_index(args.index_prefix):
        raise RuntimeError(f"未找到索引: {args.index_prefix}。请先运行 experiments/02_build_index.py。")
    if not rag.bgem3_retriever.documents:
        rag.index_documents(docs)
    rag.load_global_graph(args.graph_path)

    out = Path(args.out) if args.out else ARTIFACT_DIR / f"06_batch_{args.ablation}.jsonl"
    ensure_artifact_dir()
    if out.exists():
        out.unlink()

    rows: List[Dict[str, Any]] = []
    ok = 0
    failed = 0
    fallback = 0
    for item in qa_items:
        row: Dict[str, Any] = {
            "qa_id": item["qa_id"],
            "question": item["question"],
            "gold_answer": item["answer"],
            "gold_titles": item["supporting_titles"],
            "ablation": args.ablation,
        }
        try:
            result = rag.query(item["question"])
            metadata = result.get("metadata", {})
            row.update(
                {
                    "ok": True,
                    "answer": result.get("answer", ""),
                    "metadata": metadata,
                    "entity_type_filter_fallback_used": metadata.get("entity_type_filter_fallback_used"),
                    "followup_used_firsthop_evidence": metadata.get("followup_used_firsthop_evidence"),
                    "legacy_intent_schema_used": metadata.get("legacy_intent_schema_used"),
                }
            )
            ok += 1
            fallback += int(bool(metadata.get("entity_type_filter_fallback_used")))
        except Exception as e:
            row.update({"ok": False, "error": repr(e)})
            failed += 1
        rows.append(row)
        append_jsonl(out, [row])

    print(
        {
            "out": str(out),
            "num_queries": len(qa_items),
            "ok": ok,
            "failed": failed,
            "entity_filter_fallback_rate": fallback / max(1, ok),
            "ablation": args.ablation,
        }
    )


if __name__ == "__main__":
    main()
