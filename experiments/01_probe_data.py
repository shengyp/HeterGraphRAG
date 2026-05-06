# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from collections import Counter

from common import DATA_PATH, ARTIFACT_DIR, ensure_artifact_dir, load_chunks, make_qa_items, chunks_to_documents, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="在运行 RAG 实验前检查 Hotpot chunk 数据。")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--limit-chunks", type=int, default=None)
    args = parser.parse_args()

    chunks = load_chunks(path=DATA_PATH if args.data == str(DATA_PATH) else __import__("pathlib").Path(args.data), limit=args.limit_chunks)
    qa_items = make_qa_items(chunks)
    docs = chunks_to_documents(chunks)

    support_counter = Counter(bool(ch.get("is_supporting")) for ch in chunks)
    chunks_per_qa = Counter(ch.get("qa_id") for ch in chunks)
    docs_per_qa = Counter(doc.get("qa_id") for doc in docs)

    report = {
        "num_chunks": len(chunks),
        "num_qa": len(qa_items),
        "num_docs": len(docs),
        "supporting_chunks": support_counter.get(True, 0),
        "non_supporting_chunks": support_counter.get(False, 0),
        "avg_chunks_per_qa": sum(chunks_per_qa.values()) / max(1, len(chunks_per_qa)),
        "avg_docs_per_qa": sum(docs_per_qa.values()) / max(1, len(docs_per_qa)),
        "first_qa": qa_items[0] if qa_items else None,
        "first_chunk": chunks[0] if chunks else None,
    }

    out = ensure_artifact_dir() / "01_probe_data_report.json"
    write_json(out, report)
    print(f"已写入 {out}")
    print(report)


if __name__ == "__main__":
    main()
