# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class AnswerGenerator:


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = cfg

        self.k_per_type = int(cfg.get("k_per_type", 5))
        self.k_p = int(cfg.get("k_p", self.k_per_type))
        self.k_e = int(cfg.get("k_e", self.k_per_type))
        self.k_c = int(cfg.get("k_c", self.k_per_type))
        self.k_z = int(cfg.get("k_z", self.k_per_type))

        self.max_len_prop = int(cfg.get("max_len_prop", 520))
        self.max_len_chunk = int(cfg.get("max_len_chunk", 650))
        self.max_len_ent = int(cfg.get("max_len_ent", 120))
        self.max_len_type = int(cfg.get("max_len_type", 40))

        self.llm_model = str(cfg.get("llm_model", "qwen2.5:14b"))
        self.ollama_base_url = str(cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")

        logger.info("AnswerGenerator init | model=%s", self.llm_model)

    @staticmethod
    def _safe_text(x: str) -> str:
        return x.strip()

    @staticmethod
    def _clip(x: str, n: int) -> str:
        x = x.strip()
        return x[:n] + ("..." if len(x) > n else "")

    @staticmethod
    #对一个字典列表按某个键去重，同时保留第一次出现的顺序
    def _dedup_keep_order(items: List[Dict[str, Any]], key: str = "id") -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in items or []:
            k = it.get(key)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    #把某一类节点列表做统一清洗、补全文本、去重、排序、截断，最后取 top-k
    def _normalize_bucket(self, bucket: List[Dict[str, Any]], node_type: str, graph, topk: int) -> List[Dict[str, Any]]:
        if not bucket:
            return []

        normed: List[Dict[str, Any]] = []
        for it in bucket:
            nid = it.get("id")
            if not nid:
                continue
            score = float(it.get("score", 0.0))
            text = self._safe_text(it.get("text", ""))

            if (not text) and graph is not None and hasattr(graph, "nodes") and nid in graph.nodes:
                text = self._safe_text(graph.nodes[nid].get("text", ""))

            obj = {"id": nid, "text": text, "score": score}
            for k in ("entity_ids", "source_chunk_ids", "fact_ids"):
                if k in it:
                    obj[k] = it.get(k)
            normed.append(obj)

        normed = self._dedup_keep_order(normed, key="id")
        normed.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        if node_type == "proposition":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_prop)
        elif node_type == "chunk":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_chunk)
        elif node_type == "entity":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_ent)
        elif node_type == "type":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_type)

        return normed[: max(0, int(topk))]

    def select_evidence_nodes(self, ranked_nodes_by_type: Dict[str, List[Dict[str, Any]]], graph) -> Dict[str, List[Dict[str, Any]]]:
        buckets = ranked_nodes_by_type or {}
        evidence = {
            "proposition": self._normalize_bucket(buckets.get("proposition") or [], "proposition", graph, self.k_p),
            "entity": self._normalize_bucket(buckets.get("entity") or [], "entity", graph, self.k_e),
            "chunk": self._normalize_bucket(buckets.get("chunk") or [], "chunk", graph, self.k_c),
            "type": self._normalize_bucket(buckets.get("type") or [], "type", graph, self.k_z),
        }
        logger.info(
            "Evidence selected: P=%d E=%d C=%d Z=%d",
            len(evidence["proposition"]),
            len(evidence["entity"]),
            len(evidence["chunk"]),
            len(evidence["type"]),
        )
        return evidence

    @staticmethod
    def _sem_hint(sem: str) -> str:
        sem = (sem or "OTHER").strip().upper()
        if sem == "HUM":
            return "This question is likely about a person, such as an actor, author, or other individual."
        if sem == "LOC":
            return "This question is likely about a location, such as a place, city, country, or birthplace."
        return "No type-specific hint is needed for this question."

    @staticmethod
    def _topo_hint(topo: str) -> str:
        topo = (topo or "Sequential").strip()
        if topo == "Parallel":
            return "Note: this question may require collecting multiple evidence points in parallel and then combining them."
        return "Note: this question likely requires finding a bridge entity or fact first, then using it to reach the final answer."

    @staticmethod
    def _type_header(type_nodes: List[Dict[str, Any]]) -> str:
        labels = []
        for z in type_nodes or []:
            t = (z.get("text") or "").strip().upper()
            if t and t not in labels:
                labels.append(t)
        if not labels:
            return ""
        return "Entity Type Hint：" + ", ".join(labels[:4])

    def build_rag_prompt(
            self,
            query: str,
            evidence: Dict[str, List[Dict[str, Any]]],
            intent_topo: str,
            intent_sem: str
    ) -> str:
        query = self._safe_text(query)

        parts: List[str] = []
        parts.append("# Role")
        parts.append("You are a rigorous multi-hop QA assistant.")
        parts.append("Answer using only the evidence below.")
        parts.append("Do not use outside knowledge or guess.")
        parts.append(
            "If the evidence is insufficient, answer: Insufficient evidence, and briefly state what is missing.")

        parts.append("\n# Question")
        parts.append(query)

        parts.append("\n# Intent")
        parts.append(f"- Topology: {intent_topo} ({self._topo_hint(intent_topo)})")
        parts.append(f"- Semantic focus: {intent_sem} ({self._sem_hint(intent_sem)})")

        th = self._type_header(evidence.get("type") or [])
        if th:
            parts.append("\n# Entity Type Hint")
            parts.append(th)

        if evidence.get("entity"):
            parts.append("\n# Entities")
            parts.append("These are for disambiguation only.")
            for i, e in enumerate(evidence["entity"], 1):
                parts.append(f"{i}. {e.get('text', '')}")

        if evidence.get("proposition"):
            parts.append("\n# Propositions")
            parts.append("These are for intermediate reasoning only. Do not cite them.")
            for i, p in enumerate(evidence["proposition"], 1):
                parts.append(f"(P{i}) {p.get('text', '')}")

        if evidence.get("chunk"):
            parts.append("\n# Evidence Chunks")
            parts.append("Only chunks may be cited.")
            for i, ch in enumerate(evidence["chunk"], 1):
                parts.append(f"(chunk:[{i}]) {ch.get('text', '')}")

        parts.append("\n# Rules")
        parts.append("- Cite only chunks, using (chunk:[k]).")
        parts.append("- Every key factual claim must have at least one chunk citation.")
        parts.append("- Do not cite entities, propositions, or type hints.")
        parts.append("- Do not infer unsupported facts.")

        parts.append("\n# Output")
        parts.append("Provide only the final answer, concise and with inline chunk citations.")

        return "\n".join(parts)

    def generate_answer_with_rag(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(self.config.get("temperature", 0.2)),
                    "num_predict": int(self.config.get("num_predict", 240)),
                },
            },
            timeout=int(self.config.get("timeout", 90)),
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def generate_answer(
        self,
        query: str,
        ranked_nodes_by_type: Dict[str, List[Dict[str, Any]]],
        graph,
        intent_topo: str = "Sequential",
        intent_sem: str = "OTHER",
    ) -> Dict[str, Any]:
        logger.info("开始生成答案: query='%s...'", (query or "")[:80])
        evidence = self.select_evidence_nodes(ranked_nodes_by_type, graph)
        prompt = self.build_rag_prompt(query, evidence, intent_topo=intent_topo, intent_sem=intent_sem)
        answer = self.generate_answer_with_rag(prompt)
        return {"answer": answer, "supporting_nodes": evidence, "prompt": prompt}
