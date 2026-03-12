# -*- coding: utf-8 -*-

import logging
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

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
        self.global_graph: nx.DiGraph

        self.bgem3_retriever = BGEM3Retriever(self.config["coarse_retrieval"])
        self.graph_builder = GraphBuilder(self.config["graph_construction"])

        self.intent_repr = IntentRepresentation(self.config["intent_representation"])
        self.anchor_selector = SemanticAnchorSelector(self.config["semantic_anchor"])
        self.centrality_ranker = StructuralCentralityRanker(self.config["structural_centrality"])
        self.answer_generator = AnswerGenerator(self.config["answer_generation"])

        rerank_cfg = self.config["rerank"]
        self.alpha_sem = float(rerank_cfg["alpha_sem"])
        self.beta_struct = float(rerank_cfg["beta_struct"])

        # 语义锚点（top-k propositions）
        self.top_anchor_p = int(rerank_cfg["top_anchor_p"])
        # 扩展后最多保留多少 P
        self.max_final_p = int(rerank_cfg["max_final_p"])
        self.max_chunks_per_p = int(rerank_cfg["max_chunks_per_p"])

        logger.info(
            "RAGSystem ready | α=%.2f β=%.2f top_anchor_p=%d max_final_p=%d",
            self.alpha_sem,
            self.beta_struct,
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

        # 5.2（前置）：只用 query 做意图识别
        _, intent_topo, intent_sem = self.intent_repr.predict_intent(query)

        # 5.1（按 topo 控制）：粗检
        candidate_docs = self._coarse_retrieve_intent_controlled(query, intent_topo)

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
        sem_out = self.anchor_selector.select_semantic_anchors(G_q, query, intent_topo, intent_sem)
        s_sem = sem_out.get("scores") or {}
        if not s_sem:
            raise RuntimeError("SemanticAnchorSelector 未返回任何语义分")

        # 5.5 融合结构先验 + 语义分，选 top-k proposition 作为锚点
        anchor_props = self._select_anchor_props(G_q, s_sem)

        # 5.6 意图控制的两跳扩展（P_anchor -> E -> P），Layer2 强约束
        expanded_props = self._intent_controlled_expand(anchor_props, intent_sem)

        # 最终 P 集合：锚点 + 扩展
        final_props = list(dict.fromkeys(anchor_props + expanded_props))[: self.max_final_p]

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
            intent_topo=intent_topo,
            intent_sem=intent_sem,
        )

        result["metadata"] = {
            "intent_topo": intent_topo,
            "intent_sem": intent_sem,
            "num_seed_chunks": len(seed_chunk_ids),
            "num_q_nodes": G_q.number_of_nodes(),
            "num_q_edges": G_q.number_of_edges(),
            "num_anchor_props": len(anchor_props),
            "num_final_props": len(final_props),
            "alpha_sem": self.alpha_sem,
            "beta_struct": self.beta_struct,
        }
        return result

    def _make_followup_query_from_docs(self, query: str, docs: List[Dict[str, Any]]) -> str:
        parts: List[str] = [query]
        for d in docs[:2]:
            for ch in (d.get("chunks") or [])[:3]:
                t = (ch.get("text") or "").strip()
                if t:
                    parts.append(t)
        return (" ".join(parts).strip())[:600]

    def _coarse_retrieve_intent_controlled(self, query: str, intent_topo: str) -> List[Dict[str, Any]]:
        topo = (intent_topo or "Sequential").strip()

        cr_cfg = self.config["coarse_retrieval"]
        par_k = int(cr_cfg.get("top_k_final_parallel", cr_cfg.get("top_k_final", 20) * 2))
        seq_k1 = int(cr_cfg.get("top_k_final_seq1", max(10, cr_cfg.get("top_k_final", 20) // 2)))
        seq_k2 = int(cr_cfg.get("top_k_final_seq2", cr_cfg.get("top_k_final", 20)))

        if topo == "Parallel":
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
        prop_candidates = [n for n, d in G_q.nodes(data=True) if d.get("node_type") == "proposition"]
        if not prop_candidates:
            raise RuntimeError("查询子图中没有 proposition")

        scored: List[Tuple[str, float]] = []
        for pid in prop_candidates:
            if pid not in s_sem:
                continue
            nd = self.global_graph.nodes.get(pid) or {}
            if "s_struct_global" not in nd:
                raise KeyError(f"Missing s_struct_global on proposition={pid}（请先离线写入）")

            sem = float(s_sem[pid])
            st = float(nd["s_struct_global"])
            sc = self.alpha_sem * sem + self.beta_struct * st
            scored.append((pid, sc))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 回写
        for pid, sc in scored:
            self.global_graph.nodes[pid]["s_final_q"] = float(sc)
            self.global_graph.nodes[pid]["s_sem_q"] = float(s_sem.get(pid, 0.0))

        return [pid for pid, _ in scored[: self.top_anchor_p]]

    # =========================
    # 5.6 Intent-controlled P-E-P expansion
    # =========================
    def _entity_matches_sem(self, eid: str, intent_sem: str) -> bool:
        """Layer2 强约束：通过 E -> Z(IS_A) 的 Z.text 判定"""
        sem = (intent_sem or "OTHER").strip().upper()
        if sem == "OTHER":
            return True

        # 找到实体的 type label
        for nb in self.global_graph.successors(eid):
            ed = self.global_graph.get_edge_data(eid, nb) or {}
            if ed.get("edge_type") != "IS_A":
                continue
            if self.global_graph.nodes[nb].get("node_type") != "type":
                continue
            label = (self.global_graph.nodes[nb].get("text") or "").strip().upper()
            if sem == "HUM" and label == "PERSON":
                return True
            if sem == "LOC" and label == "LOCATION":
                return True
        return False

    def _intent_controlled_expand(self, anchor_props: List[str], intent_sem: str) -> List[str]:
        expanded: Set[str] = set()

        for pid in anchor_props:
            if not self.global_graph.has_node(pid):
                continue

            # P -> E
            eids: List[str] = []
            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") == "ASSERTS" and self.global_graph.nodes[nb].get("node_type") == "entity":
                    if self._entity_matches_sem(nb, intent_sem):
                        eids.append(nb)

            # E -> P
            for eid in eids:
                for nb in self.global_graph.successors(eid):
                    ed = self.global_graph.get_edge_data(eid, nb) or {}
                    if ed.get("edge_type") == "INVOLVED_IN" and self.global_graph.nodes[nb].get("node_type") == "proposition":
                        expanded.add(nb)

        # 不把 anchors 自己重复加回去
        expanded_list = [p for p in expanded if p not in set(anchor_props)]

        # 按 s_final_q 排一把（没有就 0）
        expanded_list.sort(key=lambda x: float(self.global_graph.nodes[x].get("s_final_q", 0.0)), reverse=True)
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
            support_chunks: List[str] = []
            for nb in self.global_graph.successors(pid):
                ed = self.global_graph.get_edge_data(pid, nb) or {}
                if ed.get("edge_type") == "SUPPORTED_BY":
                    support_chunks.append(nb)
            support_chunks = support_chunks[: self.max_chunks_per_p]

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
