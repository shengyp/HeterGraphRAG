# -*- coding: utf-8 -*-
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from HeterGraphRAG.bgem3_retriever import BGEM3Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    base_dir = ROOT
    data_dir = base_dir / "data"
    model_dir = base_dir / "models" / "bge-m3"

    docs_path = data_dir / "hotpot_docs.json"
    qa_path = data_dir / "hotpotqa.json"
    index_prefix = data_dir / "hotpot_bgem3_index"

    config = {
        # 改成你的本地模型路径
        "model_name": str(model_dir),

        "use_fp16": True,
        "batch_size": 8,

        "top_k_dense": 200,
        "top_k_sparse": 200,
        "top_k_final": 10,

        "fusion_method": "rrf",
        "rrf_k": 60,
        "lambda_dense": 0.5,
        "lambda_sparse": 0.5,
    }

    print("=" * 80)
    print("1) 路径检查")
    print(f"docs_path    = {docs_path}")
    print(f"qa_path      = {qa_path}")
    print(f"model_dir    = {model_dir}")
    print(f"index_prefix = {index_prefix}")

    print(f"docs exists  = {docs_path.exists()}")
    print(f"qa exists    = {qa_path.exists()}")
    print(f"model exists = {model_dir.exists()}")

    print("=" * 80)
    print("2) 加载数据")
    docs = load_json(docs_path)
    qa_data = load_json(qa_path)
    print(f"docs 数量: {len(docs)}")
    print(f"qa 数量: {len(qa_data)}")

    print("=" * 80)
    print("3) 初始化检索器")
    retriever = BGEM3Retriever(config)

    print("=" * 80)
    print("4) 首次建索引（你现在没有索引，所以这里一定会执行）")
    retriever.build_doc_index(docs)
    retriever.save_index(str(index_prefix))
    print(f"索引已保存到: {index_prefix}.pkl")

    print("=" * 80)
    print("5) 取一条 query 做单测")
    sample = qa_data[0]
    query = sample["question"]
    gold_titles = list({x[0] for x in sample["supporting_facts"]})

    print(f"query       = {query}")
    print(f"标准答案 = {sample['answer']}")
    print(f"标准标题 = {gold_titles}")

    print("=" * 80)
    print("6) 执行粗检")
    top_docs = retriever.hybrid_doc_retrieval(query, top_k_final=10)

    print(f"返回 doc 数: {len(top_docs)}")
    pred_titles = [doc.get("title") for doc in top_docs]
    print(f"pred_titles(top10) = {pred_titles}")

    for rank, doc in enumerate(top_docs, start=1):
        print("-" * 80)
        print(f"[Rank {rank}] title={doc.get('title')}")
        print(f"score={doc.get('score'):.6f}")
        chunks = doc.get("chunks", [])
        print(f"chunk 数={len(chunks)}")
        for ch in chunks[:2]:
            print(f"  chunk_id={ch.get('id')}")
            print(f"  text={ch.get('text', '')[:150]}")

if __name__ == "__main__":
    main()
