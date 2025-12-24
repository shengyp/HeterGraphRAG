#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统主程序入口（适配 HotpotQA:
- data/hotpotqa.json: list[QA-item]
- data/hotpotqa_corpus.json: list[{idx,title,text}]
）
"""
import json
import logging
import os
import pickle

import yaml

from src.rag_system import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def chunk_text(text, chunk_size=200, chunk_overlap=50):
    """
    把一篇文档切成多个chunk
    """
    words = (text or "").split()
    if not words:
        return []

    chunk_size = max(int(chunk_size), 1)
    chunk_overlap = max(int(chunk_overlap), 0)
    step = max(chunk_size - chunk_overlap, 1)

    chunks = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break
    return chunks


def build_docs_with_chunks(corpus_docs, chunk_size, chunk_overlap):
    """
    把 hotpotqa_corpus.json 的条目变成粗检需要的 docs 结构：
    """
    docs = []
    for item in corpus_docs:
        idx = item.get("idx")
        title = item.get("title", "")
        text = item.get("text", "") or ""


        if idx is None:
            idx = len(docs)

        doc_id = f"doc_{idx}"

        chunk_texts = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for i, ct in enumerate(chunk_texts):
            chunks.append(
                {
                    "id": f"c_{idx}_{i}",
                    "text": ct,
                    "source_doc_id": doc_id,
                    "title": title,
                }
            )

        docs.append(
            {
                "id": doc_id,
                "idx": idx,
                "title": title,
                "text": text,
                "chunks": chunks,
            }
        )
    return docs


def flatten_chunks(docs):
    """把 docs 里的 chunks 摊平成一个 list，用于建全局图"""
    all_chunks = []
    for doc in docs:
        for ch in doc.get("chunks", []) or []:
            all_chunks.append(ch)
    return all_chunks


def main():
    # 1) 加载配置
    logger.info("加载配置文件...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 2) 初始化 RAG 系统
    logger.info("初始化RAG系统...")
    rag_system = RAGSystem(config)

    # 3) 读 HotpotQA
    dataset_cfg = config.get("dataset", {}) or {}
    data_path = dataset_cfg.get("data_path", "data/hotpotqa.json")
    corpus_path = dataset_cfg.get("corpus_path", "data/hotpotqa_corpus.json")

    logger.info("加载 hotpotqa.json（问题集）: %s", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        questions = json.load(f)  # list[dict]

    logger.info("加载 hotpotqa_corpus.json（语料库）: %s", corpus_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_docs = json.load(f)  # list[dict]

    logger.info("加载完成：%d 个问题, %d 篇语料文档", len(questions), len(corpus_docs))

    # 4) 文档预处理：把 corpus 文档切成 chunks
    gc_cfg = config.get("graph_construction", {}) or {}
    chunk_size = gc_cfg.get("chunk_size", 200)
    chunk_overlap = gc_cfg.get("chunk_overlap", 50)

    logger.info("预处理语料：chunk_size=%s, chunk_overlap=%s", chunk_size, chunk_overlap)
    docs_with_chunks = build_docs_with_chunks(corpus_docs, chunk_size, chunk_overlap)
    all_chunks = flatten_chunks(docs_with_chunks)
    logger.info("切分完成：%d 篇文档 -> %d 个 chunks", len(docs_with_chunks), len(all_chunks))

    # 5) 离线阶段：构建粗检索引（基于文档）
    logger.info("构建文档索引（粗检）...")
    rag_system.index_documents(docs_with_chunks)

    # 6) 离线阶段：构建/加载全局图（基于 chunk 列表，确保 chunk_id 对齐）
    graph_path = "data/global_graph.pkl"
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)

    if os.path.exists(graph_path):
        logger.info("加载已保存的全局图: %s", graph_path)
        rag_system.load_global_graph(graph_path)
    else:
        logger.info("构建全局异构图（基于所有 chunks）...")
        rag_system.build_global_graph(all_chunks)

        logger.info("保存全局图到: %s", graph_path)
        rag_system.save_global_graph(graph_path)

    # 7) 在线查询示例：用前 N 个问题测试
    logger.info("\n" + "=" * 60)
    logger.info("开始查询示例")
    logger.info("=" * 60 + "\n")

    test_n = 3
    for qa in questions[:test_n]:
        query = qa.get("question", "")
        gold = qa.get("answer", "N/A")

        logger.info("\n问题: %s", query)
        logger.info("标准答案: %s", gold)

        result = rag_system.query(query)
        logger.info("系统答案: %s", result.get("answer", "N/A"))
        logger.info("元数据: %s", result.get("metadata", {}))
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
