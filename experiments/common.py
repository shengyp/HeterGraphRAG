# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
DATA_PATH = ROOT / "data" / "hotpot_chunks.json"
ARTIFACT_DIR = ROOT / "experiments" / "artifacts"
BGE_M3_MODEL_NAME = os.environ.get("BGE_M3_MODEL_NAME", "/home/featurize/bge-m3")

if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


def ensure_artifact_dir(path: Path = ARTIFACT_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_chunks(path: Path = DATA_PATH, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    if limit is not None:
        chunks = chunks[: max(0, int(limit))]
    return chunks


def group_chunks_by_qa(chunks: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_qa: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ch in chunks:
        qid = str(ch.get("qa_id") or "").strip()
        if qid:
            by_qa[qid].append(ch)
    return dict(by_qa)


def chunks_to_documents(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for ch in chunks:
        qid = str(ch.get("qa_id") or "").strip()
        title = str(ch.get("title") or "").strip()
        if qid and title:
            grouped[(qid, title)].append(ch)

    docs: List[Dict[str, Any]] = []
    for (qid, title), items in grouped.items():
        items = sorted(items, key=lambda x: int(x.get("sent_idx", 0)))
        question = str(items[0].get("question") or "")
        answer = str(items[0].get("answer") or "")
        text = " ".join(str(x.get("text") or "").strip() for x in items if str(x.get("text") or "").strip())
        docs.append(
            {
                "id": f"D::{qid}::{title}",
                "qa_id": qid,
                "title": title,
                "question": question,
                "answer": answer,
                "text": text,
                "chunks": items,
                "has_supporting": any(bool(x.get("is_supporting")) for x in items),
            }
        )
    return docs


def make_qa_items(chunks: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    by_qa = group_chunks_by_qa(chunks)
    items: List[Dict[str, Any]] = []
    for qid, qchunks in by_qa.items():
        first = qchunks[0]
        supporting_chunks = [x for x in qchunks if bool(x.get("is_supporting"))]
        supporting_titles = sorted({str(x.get("title") or "") for x in supporting_chunks if x.get("title")})
        items.append(
            {
                "qa_id": qid,
                "question": str(first.get("question") or ""),
                "answer": str(first.get("answer") or ""),
                "supporting_titles": supporting_titles,
                "supporting_chunk_ids": [str(x.get("id")) for x in supporting_chunks if x.get("id")],
            }
        )
    items.sort(key=lambda x: x["qa_id"])
    if limit is not None:
        items = items[: max(0, int(limit))]
    return items


def select_chunks_for_qa_ids(chunks: Iterable[Dict[str, Any]], qa_ids: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = {str(x).strip() for x in qa_ids if str(x).strip()}
    return [ch for ch in chunks if str(ch.get("qa_id") or "").strip() in wanted]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_config(strict: bool = True) -> Dict[str, Any]:
    return {
        "strict_methodology": bool(strict),
        "coarse_retrieval": {
            "model_name": BGE_M3_MODEL_NAME,
            "use_fp16": True,
            "batch_size": 8,
            "top_k_dense": 200,
            "top_k_sparse": 200,
            "top_k_final": 20,
            "top_k_final_parallel": 40,
            "top_k_final_seq1": 10,
            "top_k_final_seq2": 20,
            "fusion_method": "rrf",
            "rrf_k": 60,
            "lambda_dense": 0.5,
            "lambda_sparse": 0.5,
        },
        "followup_evidence": {
            "feedback_docs": 5,
            "chunks_per_doc": 6,
            "max_snippets": 4,
            "snippet_chars": 180,
            "max_query_chars": 600,
            "dense_weight": 0.55,
            "sparse_weight": 0.25,
            "doc_weight": 0.15,
            "overlap_weight": 0.05,
            "min_chars": 40,
        },
        "graph_construction": {
            "ollama_base_url": "http://localhost:11434",
            "llm_model": "qwen2.5:14b",
            "embedding_model": "bge-m3:latest",
            "embedding_dim": 1024,
            "entity_type_labels": [
                "PERSON",
                "LOCATION",
                "ORGANIZATION",
                "WORK",
                "EVENT",
                "TIME",
                "NUMBER",
                "CONCEPT",
                "OTHER",
            ],
            "entity_type_cache_path": str(ARTIFACT_DIR / "entity_type_cache.json"),
            "debug_dir": str(ARTIFACT_DIR / "debug_logs"),
            "strict_methodology": bool(strict),
            "offline_struct_prior": {
                "alpha_core": 0.5,
                "alpha_betw": 0.5,
                "attr": "s_struct_global",
            },
        },
        "intent_representation": {
            "ollama_base_url": "http://localhost:11434",
            "llm_model": "qwen2.5:14b",
            "timeout": 90,
            "max_ctx_snippets": 4,
            "max_snippet_len": 260,
            "strict_methodology": bool(strict),
        },
        "semantic_anchor": {
            "ollama_base_url": "http://localhost:11434",
            "llm_model": "qwen2.5:14b",
            "llm_score_timeout": 20,
            "max_node_text_len": 420,
        },
        "structural_centrality": {
            "gamma_sem": 0.5,
            "beta_struct": 1.0,
            "struct_attr": "s_struct_global",
            "k_prop": 12,
            "k_ent": 12,
        },
        "rerank": {
            "top_anchor_p": 8,
            "top_entity_expand": 6,
            "max_final_p": 18,
            "max_chunks_per_p": 3,
            "expand_decay": 0.8,
            "expand_struct_mix": 0.3,
        },
        "answer_generation": {
            "ollama_base_url": "http://localhost:11434",
            "llm_model": "qwen2.5:14b",
            "k_per_type": 5,
            "max_len_prop": 520,
            "max_len_chunk": 650,
            "max_len_ent": 120,
            "max_len_type": 40,
            "temperature": 0.2,
            "num_predict": 240,
            "timeout": 90,
        },
    }
