# -*- coding: utf-8 -*-

import logging
import pickle
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.bgem3_retriever import BGEM3Retriever
from src.graph_builder import GraphBuilder
from src.intent_representation import IntentRepresentation
from src.semantic_anchor_selector import SemanticAnchorSelector
from src.structural_centrality_ranker import StructuralCentralityRanker
from src.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class RAGSystem:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_graph: Optional[nx.DiGraph] = None
        methodology_cfg = self.config.get("methodology", {}) or {}
        self.strict_methodology = bool(
            self.config.get("strict_methodology", methodology_cfg.get("strict_methodology", False))
        )
        self._methodology_metadata: Dict[str, Any] = {}

        self.bgem3_retriever = BGEM3Retriever(self.config["coarse_retrieval"])
        graph_cfg = dict(self.config["graph_construction"])
        graph_cfg.setdefault("strict_methodology", self.strict_methodology)
        self.graph_builder = GraphBuilder(graph_cfg)

        self.intent_repr = IntentRepresentation(self.config["intent_representation"])
        self.intent_repr.strict_methodology = bool(
            self.strict_methodology or getattr(self.intent_repr, "strict_methodology", False)
        )
        self.anchor_selector = SemanticAnchorSelector(self.config["semantic_anchor"])
        self.centrality_ranker = StructuralCentralityRanker(self.config["structural_centrality"])
        self.answer_generator = AnswerGenerator(self.config["answer_generation"])

        rerank_cfg = self.config.get("rerank", {}) or {}
        self.top_anchor_p = int(rerank_cfg.get("top_anchor_p", 8))
        self.top_entity_expand = int(rerank_cfg.get("top_entity_expand", 6))
        self.max_final_p = int(rerank_cfg.get("max_final_p", 18))
        self.max_chunks_per_p = int(rerank_cfg.get("max_chunks_per_p", 3))

        # 扩展分数传播参数
        self.expand_decay = float(rerank_cfg.get("expand_decay", 0.8))
        self.expand_struct_mix = float(rerank_cfg.get("expand_struct_mix", 0.3))

        follow_cfg = self.config.get("followup_evidence", {}) or {}
        self.followup_feedback_docs = int(follow_cfg.get("feedback_docs", 5))
        self.followup_chunks_per_doc = int(follow_cfg.get("chunks_per_doc", 6))
        self.followup_max_snippets = int(follow_cfg.get("max_snippets", 4))
        self.followup_snippet_chars = int(follow_cfg.get("snippet_chars", 180))
        self.followup_max_query_chars = int(follow_cfg.get("max_query_chars", 600))
        self.followup_dense_weight = float(follow_cfg.get("dense_weight", 0.55))
        self.followup_sparse_weight = float(follow_cfg.get("sparse_weight", 0.25))
        self.followup_doc_weight = float(follow_cfg.get("doc_weight", 0.15))
        self.followup_overlap_weight = float(follow_cfg.get("overlap_weight", 0.05))
        self.followup_min_chars = int(follow_cfg.get("min_chars", 40))

        logger.info(
            "RAGSystem ready | gamma_sem=%.2f struct_attr=%s top_anchor_p=%d max_final_p=%d",
            self.centrality_ranker.gamma_sem,
            self.centrality_ranker.struct_attr,
            self.top_anchor_p,
            self.max_final_p,
        )

    # -------------------------
    # Graph I/O
    # -------------------------
    def build_global_graph(self, chunks: List[Dict[str, Any]]):
        logger.info("开始构建全局图: chunks=%d", len(chunks))
        self.global_graph = self.graph_builder.build_heterogeneous_graph(chunks)
        logger.info(
            "全局图构建完成: nodes=%d edges=%d",
            self.global_graph.number_of_nodes(),
            self.global_graph.number_of_edges(),
        )

    def save_global_graph(self, path: str):
        if self.global_graph is None:
            raise RuntimeError("global_graph 为空，无法保存")
        with open(path, "wb") as f:
            pickle.dump(self.global_graph, f)
        logger.info("全局图已保存: %s", path)

    def load_global_graph(self, path: str):
        with open(path, "rb") as f:
            self.global_graph = pickle.load(f)
        logger.info(
            "全局图已加载: nodes=%d edges=%d",
            self.global_graph.number_of_nodes(),
            self.global_graph.number_of_edges(),
        )

    # -------------------------
    # Retriever index
    # -------------------------
    def index_documents(self, documents: List[Dict[str, Any]]):
        self.bgem3_retriever.build_doc_index(documents)

    # -------------------------
    # Query
    # -------------------------
    def query(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query 为空")

        if self.global_graph is None:
            raise RuntimeError("global_graph 尚未构建")
        self._methodology_metadata = {
            "strict_methodology": bool(self.strict_methodology),
            "entity_type_filter_candidate_count": 0,
            "entity_type_filter_matched_count": 0,
            "entity_type_filter_fallback_used": False,
            "followup_used_firsthop_evidence": False,
            "followup_firsthop_snippet_count": 0,
            "legacy_intent_schema_used": False,
            "type_cache_hit": False,
            "type_cache_label": [],
            "type_cache_observed_entity_count": 0,
            "type_cache_missing_entity_count": 0,
        }
        # 5.2（前置）：只用 query 做意图识别
        _, reasoning_type, answer_type = self.intent_repr.predict_intent(query)
        self._methodology_metadata.update(getattr(self.intent_repr, "last_metadata", {}) or {})

        # 5.1（按 topo 控制）：粗检
        candidate_docs = self._coarse_retrieve_intent_controlled(query, reasoning_type)

        seed_chunk_ids: List[str] = []
        seed_chunks_payload: List[Dict[str, Any]] = []

        for doc in candidate_docs:
            for ch in (doc.get("chunks") or []):
                cid = ch.get("id")
                if cid:
                    seed_chunk_ids.append(cid)
                    seed_chunks_payload.append(ch)

        seed_chunk_ids = list(dict.fromkeys(seed_chunk_ids))
        seed_chunk_ids = [cid for cid in seed_chunk_ids if self.global_graph.has_node(cid)]

        if not seed_chunk_ids:
            raise RuntimeError("粗检索未命中任何 chunk")

        # 5.3 查询子图：以 seed chunks 触发，但只走 P–E–P 两跳
        G_q, p_seed = self._build_query_subgraph_from_seeds(seed_chunk_ids)
        if not p_seed:
            raise RuntimeError("seed chunks 未绑定到任何 proposition")

        # 5.4 语义锚点评分：只对 proposition
        sem_out = self.anchor_selector.select_semantic_anchors(G_q, query, reasoning_type, answer_type)
        s_sem = sem_out.get("scores") or {}
        if not s_sem:
            raise RuntimeError("SemanticAnchorSelector 未返回任何语义分")

        # 5.5 融合结构先验 + 语义分，选 top-k proposition 作为锚点
        anchor_props = self._select_anchor_props(G_q, s_sem)

        # 5.6 意图控制的两跳扩展（P_anchor -> E -> P），Layer2 强约束
        expanded_props = self._intent_controlled_expand(anchor_props, answer_type)

        # 最终 P 集合：锚点 + 扩展
        # 合并 anchor + expanded，并去重
        merged_props = list(dict.fromkeys(anchor_props + expanded_props))

        # 按 query-time final score 统一重排
        merged_props.sort(
            key=lambda pid: float(self.global_graph.nodes[pid].get("s_final_q", 0.0)),
            reverse=True,
        )

        # 截断
        final_props = merged_props[: self.max_final_p]

        logger.info(
            "Final proposition merge done: anchors=%d expanded=%d merged=%d kept=%d",
            len(anchor_props),
            len(expanded_props),
            len(merged_props),
            len(final_props),
        )
        for rank, pid in enumerate(final_props[:10], start=1):
            ptxt = str(self.global_graph.nodes[pid].get("text", ""))[:120].replace("\n", " ")
            logger.info(
                "[FinalMergedProp %02d] pid=%s | s_final_q=%.4f | s_struct_q=%.4f | s_sem_q=%.4f | text=%s",
                rank,
                pid,
                float(self.global_graph.nodes[pid].get("s_final_q", 0.0)),
                float(self.global_graph.nodes[pid].get("s_struct_q", 0.0)),
                float(self.global_graph.nodes[pid].get("s_sem_q", 0.0)),
                ptxt,
            )

        # 5.7 证据聚合：P -> C，E -> Z 用于消歧
        buckets = self._build_evidence_buckets(final_props)
        ranked_nodes = {
            "chunk": buckets["chunk"],
            "proposition": buckets["proposition"],
            "entity": buckets["entity"],
            "type": buckets["type"],
        }

        # 5.8 答案生成
        result = self.answer_generator.generate_answer(
            query,
            ranked_nodes,
            self.global_graph,
            intent_topo=reasoning_type,
            intent_sem=answer_type,
        )

        result["metadata"] = {
            "reasoning_type": reasoning_type,
            "answer_type": answer_type,
            "intent_topo": reasoning_type,
            "intent_sem": answer_type,
            "num_seed_chunks": len(seed_chunk_ids),
            "num_q_nodes": G_q.number_of_nodes(),
            "num_q_edges": G_q.number_of_edges(),
            "num_anchor_props": len(anchor_props),
            "num_final_props": len(final_props),
            "gamma_sem": self.centrality_ranker.gamma_sem,
            "beta_struct": self.centrality_ranker.beta_struct,
            **self._methodology_metadata,
        }
        return result

    @staticmethod
    def _norm01(value: float, min_v: float, max_v: float) -> float:
        if max_v <= min_v:
            return 0.0
        return (float(value) - min_v) / (max_v - min_v)

    @staticmethod
    def _token_set(text: str) -> Set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 1}

    @staticmethod
    def _sparse_dot(a: Dict[Any, float], b: Dict[Any, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return float(sum(float(v) * float(b.get(k, 0.0)) for k, v in a.items()))

    def _fallback_followup_snippets(self, docs: List[Dict[str, Any]]) -> List[str]:
        snippets: List[str] = []
        for d in docs[:2]:
            title = str(d.get("title") or d.get("id") or "").strip()
            for ch in (d.get("chunks") or [])[:2]:
                text = (ch.get("text") or "").strip().replace("\n", " ")
                if len(text) < self.followup_min_chars:
                    continue
                text = text[: self.followup_snippet_chars]
                snippets.append(f"{title}: {text}" if title else text)
        return snippets[: self.followup_max_snippets]

    def _select_followup_evidence_snippets(self, query: str, docs: List[Dict[str, Any]]) -> List[str]:
        candidates: List[Dict[str, Any]] = []
        feedback_docs = max(1, self.followup_feedback_docs)
        chunks_per_doc = max(1, self.followup_chunks_per_doc)

        for doc_rank, d in enumerate(docs[:feedback_docs], start=1):
            title = str(d.get("title") or d.get("id") or "").strip()
            doc_score = float(d.get("score") or 0.0)
            for chunk_rank, ch in enumerate((d.get("chunks") or [])[:chunks_per_doc], start=1):
                text = (ch.get("text") or "").strip().replace("\n", " ")
                if len(text) < self.followup_min_chars:
                    continue
                candidates.append(
                    {
                        "title": title,
                        "text": text,
                        "doc_score": doc_score,
                        "chunk_rank": chunk_rank,
                    }
                )

        if not candidates:
            return []

        try:
            texts = [query] + [c["text"][:512] for c in candidates]
            dense_vecs, lexical_list = self.bgem3_retriever._encode_batch(texts, return_dense=True, return_sparse=True)
            dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
            q_vec = dense_vecs[0]
            c_vecs = dense_vecs[1:]
            q_norm = float(np.linalg.norm(q_vec)) or 1e-10
            c_norms = np.linalg.norm(c_vecs, axis=1)
            c_norms = np.where(c_norms == 0, 1e-10, c_norms)
            dense_scores = (c_vecs @ q_vec) / (c_norms * q_norm)

            q_sparse = (lexical_list or [{}])[0] or {}
            sparse_items = list((lexical_list or [])[1:])
            if len(sparse_items) < len(candidates):
                sparse_items.extend({} for _ in range(len(candidates) - len(sparse_items)))
            sparse_scores = [self._sparse_dot(q_sparse, sparse or {}) for sparse in sparse_items]
        except Exception as e:
            logger.warning("follow-up evidence rerank failed, fallback to first chunks: %s", e)
            return self._fallback_followup_snippets(docs)

        q_terms = self._token_set(query)
        overlap_scores = []
        for c in candidates:
            c_terms = self._token_set(c["text"])
            overlap_scores.append(len(q_terms & c_terms) / max(1, len(q_terms)))

        doc_scores = [float(c["doc_score"]) for c in candidates]
        dense_min, dense_max = float(np.min(dense_scores)), float(np.max(dense_scores))
        sparse_min, sparse_max = min(sparse_scores), max(sparse_scores)
        doc_min, doc_max = min(doc_scores), max(doc_scores)

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for i, c in enumerate(candidates):
            dense = self._norm01(float(dense_scores[i]), dense_min, dense_max)
            sparse = self._norm01(float(sparse_scores[i]), sparse_min, sparse_max)
            doc_rel = self._norm01(float(c["doc_score"]), doc_min, doc_max)
            overlap = float(overlap_scores[i])
            early_chunk_bias = 1.0 / max(1, int(c["chunk_rank"]))
            score = (
                self.followup_dense_weight * dense
                + self.followup_sparse_weight * sparse
                + self.followup_doc_weight * doc_rel
                + self.followup_overlap_weight * overlap
                + 0.02 * early_chunk_bias
            )
            ranked.append((score, c))

        ranked.sort(key=lambda x: x[0], reverse=True)

        snippets: List[str] = []
        seen_texts: Set[str] = set()
        max_snippets = max(1, self.followup_max_snippets)
        for _, c in ranked:
            normalized = re.sub(r"\s+", " ", c["text"].lower())[:120]
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            text = c["text"][: self.followup_snippet_chars]
            snippets.append(f"{c['title']}: {text}" if c["title"] else text)
            if len(snippets) >= max_snippets:
                break
        return snippets

    def _make_followup_query_from_docs(self, query: str, docs: List[Dict[str, Any]]) -> str:
        snippets = self._select_followup_evidence_snippets(query, docs)

        if not snippets:
            self._methodology_metadata["followup_used_firsthop_evidence"] = False
            self._methodology_metadata["followup_firsthop_snippet_count"] = 0
            if self.strict_methodology:
                raise RuntimeError("strict_methodology=True but follow-up query has no first-hop evidence snippets")
            return query[: self.followup_max_query_chars]

        evidence = " ; ".join(snippets[: self.followup_max_snippets])
        self._methodology_metadata["followup_used_firsthop_evidence"] = True
        self._methodology_metadata["followup_firsthop_snippet_count"] = len(snippets[: self.followup_max_snippets])
        return f"{query} Key first-hop evidence: {evidence}".strip()[: self.followup_max_query_chars]

    def _coarse_retrieve_intent_controlled(self, query: str, intent_topo: str) -> List[Dict[str, Any]]:
        topo = (intent_topo or "BRIDGE").strip().upper()

        cr_cfg = self.config["coarse_retrieval"]
        par_k = int(cr_cfg.get("top_k_final_parallel", cr_cfg.get("top_k_final", 20) * 2))
        seq_k1 = int(cr_cfg.get("top_k_final_seq1", max(10, cr_cfg.get("top_k_final", 20) // 2)))
        seq_k2 = int(cr_cfg.get("top_k_final_seq2", cr_cfg.get("top_k_final", 20)))

        if topo in {"COMPARISON", "PARALLEL"}:
            return self.bgem3_retriever.hybrid_doc_retrieval(query, top_k_final=par_k)

        docs1 = self.bgem3_retriever.hybrid_doc_retrieval(query, top_k_final=seq_k1)
        follow_q = self._make_followup_query_from_docs(query, docs1)
        docs2 = self.bgem3_retriever.hybrid_doc_retrieval(follow_q, top_k_final=seq_k2)

        merged: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for d in (docs1 + docs2):
            key = str(d.get("id") or d.get("title") or d.get("_id") or "")
            if not key:
                key = str(hash((d.get("text", "")[:200], len(d.get("chunks") or []))))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged
    # =========================
    # 5.3 Query subgraph: seed C -> P -> E -> P
    # =========================
    def _build_query_subgraph_from_seeds(self, seed_chunk_ids: List[str]) -> Tuple[nx.DiGraph, List[str]]:
        if self.global_graph is None:
            raise RuntimeError("global_graph 为空")

        # seed chunks -> propositions
        p1: Set[str] = set()
        for cid in seed_chunk_ids:
            for nb in self.global_graph.predecessors(cid):
                ed = self.global_graph.get_edge_data(nb, cid) or {}
                if ed.get("edge_type") == "SUPPORTED_BY" and self.global_graph.nodes[nb].get("node_type") == "proposition":
                    p1.add(nb)

        # propositions -> entities
        e: Set[str] = set()
        for pid in p1:
            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") == "ASSERTS" and self.global_graph.nodes[nb].get("node_type") == "entity":
                    e.add(nb)

        # entities -> propositions (two-hop reach)
        p2: Set[str] = set(p1)
        for eid in e:
            for nb in self.global_graph.successors(eid):
                ed = self.global_graph.get_edge_data(eid, nb) or {}
                if ed.get("edge_type") == "INVOLVED_IN" and self.global_graph.nodes[nb].get("node_type") == "proposition":
                    p2.add(nb)

        # 子图节点：P+E+seed chunks
        nodes = set(seed_chunk_ids) | p2 | e
        G_q = self.global_graph.subgraph(nodes).copy()

        return G_q, sorted(p1)

    # =========================
    # 5.5 Anchor selection
    # =========================
    def _select_anchor_props(self, G_q: nx.DiGraph, s_sem: Dict[str, float]) -> List[str]:
        """
        从查询子图中选择 anchor propositions。
        现在排序逻辑统一委托给 StructuralCentralityRanker。
        """
        prop_candidates = [
            n for n, d in G_q.nodes(data=True)
            if d.get("node_type") == "proposition"
        ]
        if not prop_candidates:
            raise RuntimeError("查询子图中没有 proposition 节点，无法选择 anchor")

        ranked = self.centrality_ranker.rank_anchor_nodes(
            self.global_graph,
            prop_candidates=prop_candidates,
            ent_candidates=None,
            s_sem=s_sem,
        )

        ranked_props = ranked["proposition"]

        if not ranked_props:
            raise RuntimeError("StructuralCentralityRanker 未返回任何 proposition 排序结果")


        for pid, final_score, struct_score, sem_score in ranked_props:
            self.global_graph.nodes[pid]["s_final_q"] = float(final_score)
            self.global_graph.nodes[pid]["s_struct_q"] = float(struct_score)
            self.global_graph.nodes[pid]["s_sem_q"] = float(sem_score)

        ranked_map = {
            pid: (final_score, struct_score, sem_score)
            for pid, final_score, struct_score, sem_score in ranked_props
        }
        for pid in prop_candidates:
            if pid not in ranked_map:
                struct_score = float(self.global_graph.nodes[pid].get("s_struct_global", 0.0))
                sem_score = float(s_sem.get(pid, 0.0))
                final_score = (
                        self.centrality_ranker.beta_struct * struct_score
                        + self.centrality_ranker.gamma_sem * sem_score
                )
                self.global_graph.nodes[pid]["s_final_q"] = float(final_score)
                self.global_graph.nodes[pid]["s_struct_q"] = float(struct_score)
                self.global_graph.nodes[pid]["s_sem_q"] = float(sem_score)

        anchor_props = [pid for pid, _, _, _ in ranked_props[: self.top_anchor_p]]

        logger.info(
            "Anchor propositions selected by StructuralCentralityRanker: total_candidates=%d, returned=%d, top_anchor=%d",
            len(prop_candidates),
            len(ranked_props),
            len(anchor_props),
        )
        return anchor_props



    # =========================
    # 5.6 Intent-controlled P-E-P expansion
    # =========================
    def _entity_type_labels(self, eid: str) -> Set[str]:
        labels: Set[str] = set()
        if self.global_graph is None or not self.global_graph.has_node(eid):
            return labels

        for nb in self.global_graph.successors(eid):
            ed = self.global_graph.get_edge_data(eid, nb) or {}
            if ed.get("edge_type") != "IS_A":
                continue
            if self.global_graph.nodes[nb].get("node_type") != "type":
                continue
            label = (self.global_graph.nodes[nb].get("text") or "").strip().upper()
            if label:
                labels.add(label)
        return labels

    def _entity_matches_answer_type(self, eid: str, answer_type: str) -> bool:
        atype = (answer_type or "OTHER").strip().upper()
        if atype in {"OTHER", "DESCRIPTION", "BOOLEAN"}:
            return True

        labels = self._entity_type_labels(eid)
        if atype in labels:
            return True

        if atype in {"TIME", "NUMBER"} and labels.intersection({atype, "CONCEPT"}):
            return True
        return False

    def _collect_and_rerank_expand_entities(
            self,
            anchor_props: List[str],
            intent_sem: str,
    ) -> Tuple[List[str], Dict[str, List[str]], Dict[str, float]]:
        """
        从 anchor propositions 收集候选 entity：
        1) 先按 intent_sem 做类型过滤
        2) 再用 StructuralCentralityRanker 做 entity rerank
        3) 返回:
           - top_entity_ids: 最终用于扩展的 entity
           - ent2anchors: 每个 entity 是由哪些 anchor propositions 连接到的
           - ent_score_map: entity 的 query-time rerank 分数
        """
        candidate_entities: Set[str] = set()
        ent2anchors: Dict[str, List[str]] = {}

        for pid in anchor_props:
            if not self.global_graph.has_node(pid):
                continue

            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") != "ASSERTS":
                    continue
                if self.global_graph.nodes[nb].get("node_type") != "entity":
                    continue
                candidate_entities.add(nb)
                ent2anchors.setdefault(nb, []).append(pid)

        if not candidate_entities:
            self._methodology_metadata["entity_type_filter_candidate_count"] = 0
            self._methodology_metadata["entity_type_filter_matched_count"] = 0
            self._methodology_metadata["entity_type_filter_fallback_used"] = False
            if self.strict_methodology:
                raise RuntimeError("strict_methodology=True but no candidate entities were collected for expansion")
            return [], {}, {}

        matched_entities = {
            eid for eid in candidate_entities
            if self._entity_matches_answer_type(eid, intent_sem)
        }
        fallback_used = not matched_entities
        if fallback_used:
            msg = (
                "No candidate entities matched expected answer_type="
                f"{intent_sem}; falling back to all candidate entities"
            )
            if self.strict_methodology:
                raise RuntimeError("strict_methodology=True: " + msg)
            logger.warning(msg)

        entities_to_rank = matched_entities if matched_entities else candidate_entities
        type_cache_labels: List[str] = []
        type_cache_hit = False
        type_cache_observed_count = 0
        type_cache_missing_count = 0
        for eid in entities_to_rank:
            nd = self.global_graph.nodes[eid]
            if "type_cache_hit" in nd:
                type_cache_observed_count += 1
            else:
                type_cache_missing_count += 1
            if bool(nd.get("type_cache_hit", False)):
                type_cache_hit = True
            label = str(nd.get("type_cache_label") or "").strip().upper()
            if not label:
                labels = sorted(self._entity_type_labels(eid))
                label = labels[0] if labels else ""
            if label and label not in type_cache_labels:
                type_cache_labels.append(label)

        if self.strict_methodology and type_cache_missing_count:
            raise RuntimeError(
                "strict_methodology=True but selected expansion entities do not expose "
                f"type_cache_hit/type_cache_label metadata: missing={type_cache_missing_count}"
            )

        self._methodology_metadata["entity_type_filter_candidate_count"] = len(candidate_entities)
        self._methodology_metadata["entity_type_filter_matched_count"] = len(matched_entities)
        self._methodology_metadata["entity_type_filter_fallback_used"] = bool(fallback_used)
        self._methodology_metadata["type_cache_hit"] = bool(type_cache_hit)
        self._methodology_metadata["type_cache_label"] = type_cache_labels
        self._methodology_metadata["type_cache_observed_entity_count"] = type_cache_observed_count
        self._methodology_metadata["type_cache_missing_entity_count"] = type_cache_missing_count

        ranked_entities = self.centrality_ranker.rank_entities(
            self.global_graph,
            ent_candidates=list(entities_to_rank),
            s_sem=None,
        )
        # ranked_entities: [(eid, final_score, struct_score, sem_score), ...]

        top_ranked = ranked_entities[: self.top_entity_expand]
        top_entity_ids = [eid for eid, _, _, _ in top_ranked]

        ent_score_map: Dict[str, float] = {}
        for eid, final_score, struct_score, sem_score in top_ranked:
            ent_score_map[eid] = float(final_score)
            # 也顺手写回图，后面调试好看
            self.global_graph.nodes[eid]["s_ent_q"] = float(final_score)
            self.global_graph.nodes[eid]["s_ent_struct_q"] = float(struct_score)
            self.global_graph.nodes[eid]["s_ent_sem_q"] = float(sem_score)

        logger.info(
            "Expand-entity rerank done: candidates=%d selected=%d",
            len(entities_to_rank),
            len(top_entity_ids),
        )
        for rank, (eid, final_score, struct_score, sem_score) in enumerate(top_ranked, start=1):
            etxt = str(self.global_graph.nodes[eid].get("text", ""))[:100].replace("\n", " ")
            logger.info(
                "[ExpandEntity %02d] eid=%s | final=%.4f | struct=%.4f | sem=%.4f | text=%s",
                rank, eid, final_score, struct_score, sem_score, etxt
            )

        return top_entity_ids, ent2anchors, ent_score_map

    def _backfill_expanded_prop_scores_by_decay(
            self,
            prop2sources: Dict[str, List[Tuple[str, str]]],
            ent_score_map: Dict[str, float],
    ) -> None:
        """
        给新扩展出来的 proposition 补 query-time 分数。

        逻辑：
        1) 新 proposition 的上游来源是 (anchor_pid, eid)
        2) 先从所有来源里取最强 anchor 分数
        3) 传播分 = expand_decay * best_anchor_score
        4) 最终分 = expand_struct_mix * struct_score + (1 - expand_struct_mix) * 传播分

        这里只给“还没有 s_final_q”的 proposition 补分；
        已经在 anchor 选择阶段打过分的 proposition 不覆盖。
        """
        for pid, srcs in prop2sources.items():
            if not self.global_graph.has_node(pid):
                continue

            nd = self.global_graph.nodes[pid]
            if nd.get("node_type") != "proposition":
                continue

            # 已经有 query-time 分数的不覆盖
            if "s_final_q" in nd:
                continue

            if not srcs:
                continue

            upstream_anchor_scores: List[float] = []
            for anchor_pid, eid in srcs:
                anchor_score = float(self.global_graph.nodes[anchor_pid].get("s_final_q", 0.0))
                upstream_anchor_scores.append(anchor_score)

            best_anchor_score = max(upstream_anchor_scores) if upstream_anchor_scores else 0.0
            propagated_score = self.expand_decay * best_anchor_score

            struct_score = float(nd.get(self.centrality_ranker.struct_attr, 0.0))
            final_score = (
                    self.expand_struct_mix * struct_score
                    + (1.0 - self.expand_struct_mix) * propagated_score
            )

            nd["s_struct_q"] = float(struct_score)
            nd["s_sem_q"] = float(propagated_score)  # 这里把“传播相关性”记到 s_sem_q，便于统一看日志
            nd["s_final_q"] = float(final_score)

            logger.info(
                "[BackfillExpandedProp] pid=%s | struct=%.4f | propagated=%.4f | final=%.4f",
                pid, struct_score, propagated_score, final_score
            )

    def _intent_controlled_expand(self, anchor_props: List[str], intent_sem: str) -> List[str]:
        expanded: Set[str] = set()

        # 1) 收集并 rerank entity
        top_entity_ids, ent2anchors, ent_score_map = self._collect_and_rerank_expand_entities(
            anchor_props, intent_sem
        )

        # 2) 从 top entities 往外扩 proposition，并记录来源
        prop2sources: Dict[str, List[Tuple[str, str]]] = {}

        for eid in top_entity_ids:
            anchor_list = ent2anchors.get(eid, [])
            for nb in self.global_graph.successors(eid):
                ed = self.global_graph.get_edge_data(eid, nb) or {}
                if ed.get("edge_type") != "INVOLVED_IN":
                    continue
                if self.global_graph.nodes[nb].get("node_type") != "proposition":
                    continue

                expanded.add(nb)
                for anchor_pid in anchor_list:
                    prop2sources.setdefault(nb, []).append((anchor_pid, eid))

        # 3) 不把 anchors 自己重复加回去
        expanded_list = [p for p in expanded if p not in set(anchor_props)]

        # 4) 给新扩展 proposition 补 query-time 分数（衰减传播版）
        self._backfill_expanded_prop_scores_by_decay(prop2sources, ent_score_map)

        # 5) 再按 s_final_q 排序
        expanded_list.sort(
            key=lambda x: float(self.global_graph.nodes[x].get("s_final_q", 0.0)),
            reverse=True
        )

        logger.info(
            "Intent-controlled expansion done: anchors=%d top_entities=%d expanded_props=%d",
            len(anchor_props),
            len(top_entity_ids),
            len(expanded_list),
        )
        for rank, pid in enumerate(expanded_list[:10], start=1):
            ptxt = str(self.global_graph.nodes[pid].get("text", ""))[:120].replace("\n", " ")
            logger.info(
                "[ExpandedProp %02d] pid=%s | s_final_q=%.4f | s_struct_q=%.4f | s_sem_q=%.4f | text=%s",
                rank,
                pid,
                float(self.global_graph.nodes[pid].get("s_final_q", 0.0)),
                float(self.global_graph.nodes[pid].get("s_struct_q", 0.0)),
                float(self.global_graph.nodes[pid].get("s_sem_q", 0.0)),
                ptxt,
            )

        return expanded_list

    # =========================
    # 5.7 Evidence buckets
    # =========================
    def _build_evidence_buckets(self, prop_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        chunks: Set[str] = set()
        entities: Set[str] = set()
        types: Set[str] = set()

        chunk_score: Dict[str, float] = {}
        ent_score: Dict[str, float] = {}
        type_score: Dict[str, float] = {}

        for pid in prop_ids:
            if not self.global_graph.has_node(pid):
                continue
            pnd = self.global_graph.nodes[pid]
            pscore = float(pnd.get("s_final_q", 0.0))

            # P -> C
            support_chunks = []
            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") == "SUPPORTED_BY":
                    support_chunks.append(nb)

            # 先为每个 support chunk 计算一个 chunk_score：
            # 定义为“所有支持到该 chunk 的 proposition 中，最大的 s_final_q”
            chunk_scored = []
            for cid in support_chunks:
                cscore = 0.0

                # 找所有指向这个 chunk 的 proposition
                for pred in self.global_graph.predecessors(cid):
                    if self.global_graph.nodes[pred].get("node_type") != "proposition":
                        continue
                    ed2 = self.global_graph.get_edge_data(pred, cid) or {}
                    if ed2.get("edge_type") != "SUPPORTED_BY":
                        continue
                    pscosre = float(self.global_graph.nodes[pred].get("s_final_q", 0.0))
                    cscore = max(cscore, pscore)

                chunk_scored.append((cid, cscore))

            # 按 chunk_score 降序排序，再截断
            chunk_scored.sort(key=lambda x: x[1], reverse=True)
            support_chunks = [cid for cid, _ in chunk_scored[: self.max_chunks_per_p]]
            for cid in support_chunks:
                if not self.global_graph.has_node(cid):
                    continue
                if self.global_graph.nodes[cid].get("node_type") != "chunk":
                    continue
                chunk_score[cid] = max(chunk_score.get(cid, 0.0), pscore)
                chunks.add(cid)

            # P -> E
            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") == "ASSERTS" and self.global_graph.nodes[nb].get("node_type") == "entity":
                    ent_score[nb] = max(ent_score.get(nb, 0.0), pscore)
                    entities.add(nb)

        # E -> Z
        for eid in list(entities):
            for nb in self.global_graph.successors(eid):
                ed = self.global_graph.get_edge_data(eid, nb) or {}
                if ed.get("edge_type") == "IS_A" and self.global_graph.nodes[nb].get("node_type") == "type":
                    type_score[nb] = max(type_score.get(nb, 0.0), ent_score.get(eid, 0.0))
                    types.add(nb)

        return {
            "proposition": self._make_bucket("proposition", prop_ids, score_attr="s_final_q"),
            "chunk": self._make_bucket("chunk", list(chunks), score_map=chunk_score),
            "entity": self._make_bucket("entity", list(entities), score_map=ent_score),
            "type": self._make_bucket("type", list(types), score_map=type_score),
        }

    def _make_bucket(
        self,
        node_type: str,
        node_ids: List[str],
        score_attr: Optional[str] = None,
        score_map: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for nid in node_ids:
            if not self.global_graph.has_node(nid):
                continue
            nd = self.global_graph.nodes[nid]
            if nd.get("node_type") != node_type:
                continue

            if score_map is not None:
                sc = float(score_map.get(nid, 0.0))
            elif score_attr is not None:
                sc = float(nd.get(score_attr, 0.0))
            else:
                sc = 0.0

            it: Dict[str, Any] = {"id": nid, "text": nd.get("text", ""), "score": sc}
            if node_type == "proposition":
                for k in ("entity_ids", "source_chunk_ids", "fact_ids"):
                    if k in nd:
                        it[k] = nd.get(k)
            out.append(it)

        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out
