

from typing import Any, Dict, List, Set, Tuple


def sample_to_documents(sample: Dict[str, Any], dedup_by_title: bool = True) -> List[Dict[str, Any]]:
    """
    把一条 HotpotQA 样本转成 documents
    """
    docs: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    qa_id = sample["_id"].strip()
    question = sample["question"].strip()
    answer = sample["answer"].strip()
    context = sample.get("context")
    supporting_facts = sample.get("supporting_facts")

    # 转成 set，方便判断某句是不是 supporting fact
    supporting_set: Set[Tuple[str, int]] = set()
    for item in supporting_facts:
        if isinstance(item, list) and len(item) == 2:
            title, sent_idx = item
            if isinstance(title, str) and isinstance(sent_idx, int):
                supporting_set.add((title.strip(), sent_idx))

    for item in context:
        if not (isinstance(item, list) and len(item) == 2):
            continue

        title, sents = item[0], item[1]
        if not isinstance(title, str) or not isinstance(sents, list):
            continue

        t = title.strip()
        if not t:
            continue

        if dedup_by_title and t in seen:
            continue
        seen.add(t)

        chunks: List[Dict[str, Any]] = []
        #把句子列表 sents 逐个取出来，同时拿到编号 i
        for i, s in enumerate(sents):
            if not isinstance(s, str):
                continue

            sent = s.strip()
            if not sent:
                continue

            cid = f"C::{qa_id}::{t}::s{i}"
            is_supporting = (t, i) in supporting_set

            chunks.append({
                "id": cid,
                "text": sent,
                "title": t,
                "sent_idx": i,
                "qa_id": qa_id,
                "question": question,
                "answer": answer,
                "is_supporting": is_supporting,
            })

        docs.append({
            "id": f"D::{qa_id}::{t}",
            "title": t,
            "qa_id": qa_id,
            "question": question,
            "answer": answer,
            "chunks": chunks,
        })

    return docs


def documents_to_chunks(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    把 documents 拉平成 chunks
    """
    out: List[Dict[str, Any]] = []
    for d in documents :
        for ch in (d.get("chunks")):
            if ch.get("id") and ch.get("text"):
                out.append(ch)
    return out


def dataset_to_documents(dataset: List[Dict[str, Any]], dedup_by_title: bool = True) -> List[Dict[str, Any]]:
    """
    把整个 HotpotQA 数据集转成所有 documents
    """
    all_docs: List[Dict[str, Any]] = []
    for sample in dataset :
        if not isinstance(sample, dict):
            continue
        docs = sample_to_documents(sample, dedup_by_title=dedup_by_title)
        all_docs.extend(docs)
    return all_docs


def dataset_to_chunks(dataset: List[Dict[str, Any]], dedup_by_title: bool = True) -> List[Dict[str, Any]]:
    """
    把整个 HotpotQA 数据集直接转成所有 flat chunks
    """
    all_chunks: List[Dict[str, Any]] = []
    for sample in dataset or []:
        if not isinstance(sample, dict):
            continue
        docs = sample_to_documents(sample, dedup_by_title=dedup_by_title)
        chunks = documents_to_chunks(docs)
        all_chunks.extend(chunks)
    return all_chunks