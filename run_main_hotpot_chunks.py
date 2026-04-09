#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目主入口（基于 hotpot_chunks / hotpot_docs）

用途：
1. 从 data/hotpotqa/hotpot_chunks.json 中读取 flat chunks
2. 按“前 N 个问题（qa_id）”截取一个子集
3. 用该子集构建 global graph
4. 用对应 docs 建/加载 coarse retriever index
5. 对用户给定 query 执行一遍完整 RAG 流程

示例：
python run_main_hotpot_chunks.py --num-questions 5 --query "Which film starred both ... ?"
python run_main_hotpot_chunks.py --num-questions 3 --use-first-question-query
python run_main_hotpot_chunks.py --num-questions 10 --rebuild-index --rebuild-graph
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.rag_system import RAGSystem  # noqa: E402


logger = logging.getLogger("run_main_hotpot_chunks")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件不是 dict: {path}")
    return data


def resolve_project_path(path_str: str | None, default_relative: str) -> Path:
    if path_str:
        p = Path(path_str).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (REPO_ROOT / default_relative).resolve()


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    把当前项目里容易不一致的配置字段兜一层，避免主脚本因为字段名不一致直接挂掉。
    这里只做兼容，不改原始 config.yaml 文件。
    """
    cfg = dict(cfg)

    coarse = dict(cfg.get("coarse_retrieval", {}) or {})
    if "model_name" not in coarse and "embedding_model" in coarse:
        coarse["model_name"] = coarse["embedding_model"]
    cfg["coarse_retrieval"] = coarse

    graph = dict(cfg.get("graph_construction", {}) or {})
    if "entity_type_cache_path" in graph:
        cache_path = Path(graph["entity_type_cache_path"])
        if not cache_path.is_absolute():
            graph["entity_type_cache_path"] = str((REPO_ROOT / cache_path).resolve())
    cfg["graph_construction"] = graph

    return cfg


def select_first_n_qa_ids(chunks: List[Dict[str, Any]], num_questions: int) -> List[str]:
    if num_questions <= 0:
        raise ValueError("num_questions 必须 > 0")

    qa_ids: List[str] = []
    seen = set()
    for ch in chunks:
        qid = str(ch.get("qa_id", "")).strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qa_ids.append(qid)
        if len(qa_ids) >= num_questions:
            break

    if not qa_ids:
        raise RuntimeError("hotpot_chunks.json 中没有可用 qa_id")
    return qa_ids


def filter_chunks_by_qa_ids(chunks: List[Dict[str, Any]], qa_ids: List[str]) -> List[Dict[str, Any]]:
    qa_set = set(qa_ids)
    return [ch for ch in chunks if str(ch.get("qa_id", "")).strip() in qa_set]


def filter_docs_by_qa_ids(docs: List[Dict[str, Any]], qa_ids: List[str]) -> List[Dict[str, Any]]:
    qa_set = set(qa_ids)
    out: List[Dict[str, Any]] = []
    for doc in docs:
        qid = str(doc.get("qa_id", "")).strip()
        if qid in qa_set:
            out.append(doc)
    return out


def load_question_map(qa_path: Path | None) -> Dict[str, Dict[str, Any]]:
    if qa_path is None or not qa_path.exists():
        return {}
    data = load_json(qa_path)
    if not isinstance(data, list):
        return {}

    qmap: Dict[str, Dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("_id", "")).strip()
        if qid:
            qmap[qid] = row
    return qmap


def choose_query(args: argparse.Namespace, qa_ids: List[str], question_map: Dict[str, Dict[str, Any]]) -> str:
    if args.query:
        return args.query.strip()

    if args.use_first_question_query:
        first_qid = qa_ids[0]
        row = question_map.get(first_qid, {})
        q = str(row.get("question", "")).strip()
        if q:
            return q
        raise RuntimeError("use_first_question_query=True，但未在 hotpotqa.json 中找到对应 question")

    raise ValueError("必须提供 --query，或者使用 --use-first-question-query")


def maybe_load_or_build_index(
    rag: RAGSystem,
    docs: List[Dict[str, Any]],
    index_prefix: Path,
    rebuild_index: bool,
) -> None:
    index_pkl = Path(str(index_prefix) + ".pkl")

    if index_pkl.exists() and not rebuild_index:
        rag.bgem3_retriever.load_index(str(index_prefix))
        logger.info("已加载检索索引: %s", index_pkl)
        return

    logger.info("开始构建检索索引: docs=%d", len(docs))
    rag.index_documents(docs)
    index_prefix.parent.mkdir(parents=True, exist_ok=True)
    rag.bgem3_retriever.save_index(str(index_prefix))
    logger.info("检索索引已保存: %s", index_pkl)


def maybe_load_or_build_graph(
    rag: RAGSystem,
    chunks: List[Dict[str, Any]],
    graph_path: Path,
    rebuild_graph: bool,
) -> None:
    if graph_path.exists() and not rebuild_graph:
        rag.load_global_graph(str(graph_path))
        logger.info("已加载全局图: %s", graph_path)
        return

    logger.info("开始构建全局图: chunks=%d", len(chunks))
    rag.build_global_graph(chunks)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    rag.save_global_graph(str(graph_path))
    logger.info("全局图已保存: %s", graph_path)


def summarize_subset(qa_ids: List[str], chunks: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> None:
    logger.info("本次选取前 %d 个问题", len(qa_ids))
    logger.info("qa_ids(sample)=%s", qa_ids[:5])
    logger.info("选中 docs=%d | chunks=%d", len(docs), len(chunks))



def save_run_result(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("结果已保存: %s", output_path)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="项目主入口：基于 hotpot_chunks 的可裁剪启动脚本")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--docs", type=str, default="data/hotpotqa/hotpot_docs.json", help="doc 语料路径")
    parser.add_argument("--chunks", type=str, default="data/hotpotqa/hotpot_chunks.json", help="chunk 语料路径")
    parser.add_argument("--qa", type=str, default="data/hotpotqa/hotpotqa.json", help="原始 QA 文件路径，用于读取问题文本")

    parser.add_argument("--num-questions", type=int, default=5, help="只使用前几个问题对应的 chunk/doc 子集")
    parser.add_argument("--query", type=str, default="", help="要执行的查询")
    parser.add_argument(
        "--use-first-question-query",
        action="store_true",
        help="不手写 query，直接使用本次子集中第一个问题的 question 字段",
    )

    parser.add_argument(
        "--graph-output",
        type=str,
        default=None,
        help="全局图缓存路径；默认会自动按 num_questions 命名到 offline_struct/ 下",
    )
    parser.add_argument(
        "--index-prefix",
        type=str,
        default=None,
        help="检索索引前缀；默认会自动按 num_questions 命名到 data/hotpotqa/ 下",
    )
    parser.add_argument("--rebuild-graph", action="store_true", help="强制重建 global graph")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重建 retriever index")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果 JSON 路径；默认 results/run_main_hotpot_chunks_q{num_questions}.json",
    )
    parser.add_argument("--verbose", action="store_true", help="打印更详细日志")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    config_path = resolve_project_path(args.config, "config.yaml")
    docs_path = resolve_project_path(args.docs, "data/hotpotqa/hotpot_docs.json")
    chunks_path = resolve_project_path(args.chunks, "data/hotpotqa/hotpot_chunks.json")
    qa_path = resolve_project_path(args.qa, "data/hotpotqa/hotpotqa.json")

    graph_output = resolve_project_path(
        args.graph_output,
        f"offline_struct/global_graph_first_{args.num_questions}_questions.pkl",
    )
    index_prefix = resolve_project_path(
        args.index_prefix,
        f"data/hotpotqa/hotpot_bgem3_index_first_{args.num_questions}_questions",
    )
    output_path = resolve_project_path(
        args.output,
        f"results/run_main_hotpot_chunks_q{args.num_questions}.json",
    )

    logger.info("加载配置: %s", config_path)
    cfg = normalize_config(load_yaml(config_path))

    logger.info("加载 docs: %s", docs_path)
    docs = load_json(docs_path)
    logger.info("加载 chunks: %s", chunks_path)
    chunks = load_json(chunks_path)

    if not isinstance(docs, list) or not isinstance(chunks, list):
        raise ValueError("docs/chunks 文件都必须是 list")

    qa_ids = select_first_n_qa_ids(chunks, args.num_questions)
    sub_chunks = filter_chunks_by_qa_ids(chunks, qa_ids)
    sub_docs = filter_docs_by_qa_ids(docs, qa_ids)

    if not sub_chunks:
        raise RuntimeError("筛选后没有 chunk，无法构图")
    if not sub_docs:
        raise RuntimeError("筛选后没有 doc，无法建立 coarse index")

    summarize_subset(qa_ids, sub_chunks, sub_docs)

    question_map = load_question_map(qa_path)
    query = choose_query(args, qa_ids, question_map)
    logger.info("query=%s", query)

    rag = RAGSystem(cfg)

    maybe_load_or_build_index(
        rag=rag,
        docs=sub_docs,
        index_prefix=index_prefix,
        rebuild_index=args.rebuild_index,
    )
    maybe_load_or_build_graph(
        rag=rag,
        chunks=sub_chunks,
        graph_path=graph_output,
        rebuild_graph=args.rebuild_graph,
    )

    logger.info("开始执行完整 query 流程")
    result = rag.query(query)

    # 补充本次运行的输入上下文，方便排查/复现实验
    result["run_context"] = {
        "config": str(config_path),
        "docs": str(docs_path),
        "chunks": str(chunks_path),
        "qa": str(qa_path),
        "num_questions": args.num_questions,
        "selected_qa_ids": qa_ids,
        "graph_output": str(graph_output),
        "index_prefix": str(index_prefix),
        "query": query,
    }

    save_run_result(result, output_path)

    print("=" * 100)
    print("运行完成")
    print(f"query: {query}")
    print(f"num_questions: {args.num_questions}")
    print(f"selected_qa_ids: {qa_ids}")
    print(f"result_file: {output_path}")
    print("-" * 100)
    print("answer:")
    print(result.get("answer", "<empty answer>"))
    print("=" * 100)


if __name__ == "__main__":
    main()
