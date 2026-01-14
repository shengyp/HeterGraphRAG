import logging
import pickle
from typing import Dict, List, Any
import networkx as nx

from src.bgem3_retriever import BGEM3Retriever
from src.graph_builder import GraphBuilder
from src.intent_representation import IntentRepresentation
from src.semantic_anchor_selector import SemanticAnchorSelector
from src.structural_centrality_ranker import StructuralCentralityRanker
from src.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    最终流程：
    1) 粗检 -> candidate chunk_ids
    2) 从 chunk_ids 在全图中做 2-hop -> 子图 G_q
    3) 只在子图 G_q 上做语义锚点评分，阈值 >= 0.25 选 anchors
    4) 只对这些 anchors，在全图上做结构分排序
       chunk=6 proposition=6 concept=3 summary=3
    5) 对每个 chunk anchor 做邻居扩展（4P/2Z/1S/2相似chunk跨doc）
    6) 把所有选中的节点按类型组织，送给 AnswerGenerator
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.global_graph = None

        logger.info("初始化RAGSystem...")

        
        self.bgem3_retriever = BGEM3Retriever(self.config.get("coarse_retrieval", {}) or {})

        # GraphBuilder：只有离线建图才需要
        self.graph_builder = None
        graph_cfg = self.config.get("graph_construction", {}) or {}
        try:
            required = [
                "entity_similarity_threshold",
                "concept_num_clusters",
                "proposition_similarity_threshold",
                "concept_similarity_threshold",
                "summary_similarity_threshold",
                "ollama_base_url",
                "llm_model",
            ]
            ok = True
            for k in required:
                if k not in graph_cfg:
                    ok = False
                    break
            if ok:
                self.graph_builder = GraphBuilder(graph_cfg)
            else:
                logger.info("GraphBuilder skipped")
        except Exception as e:
            logger.warning("GraphBuilder init failed, skip: %s", e)
            self.graph_builder = None

        #其他配置读取
        self.intent_repr = IntentRepresentation(self.config.get("intent_representation", {}) or {})
        self.anchor_selector = SemanticAnchorSelector(self.config.get("semantic_anchor", {}) or {})
        self.centrality_ranker = StructuralCentralityRanker(self.config.get("structural_centrality", {}) or {})
        self.answer_generator = AnswerGenerator(self.config.get("answer_generation", {}) or {})

        # subgraph hops
        sub_cfg = self.config.get("subgraph", {}) or {}
        self.subgraph_hops = int(sub_cfg.get("hops", 2))

        logger.info("RAGSystem初始化完成.")

    # -------------------------
    # Graph I/O
    # -------------------------
    #开始构图
    def build_global_graph(self, chunks):
        if self.graph_builder is None:
            raise RuntimeError("图构建器还没有构建好")

        logger.info("开始构建全局图: chunks=%d", len(chunks))
        self.global_graph = self.graph_builder.build_heterogeneous_graph(chunks)
        logger.info("全局图构建完成: nodes=%d edges=%d", self.global_graph.number_of_nodes(), self.global_graph.number_of_edges())
    #保存为pkl
    def save_global_graph(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.global_graph, f)
        logger.info("全局图已保存: %s", path)
    #加载的函数
    def load_global_graph(self, path: str):
        with open(path, "rb") as f:
            self.global_graph = pickle.load(f)
        logger.info("全局图已加载: nodes=%d", self.global_graph.number_of_nodes())

    # -------------------------
    # Index build
    # -------------------------
    #文档级别的构建，用于粗检
    def index_documents(self, documents: List[Dict[str, Any]]):
        logger.info("开始索引文档: %d", len(documents))
        self.bgem3_retriever.build_doc_index(documents)
        logger.info("文档索引完成")

    # -------------------------
    # Query
    # -------------------------
    def query(self, query):
        if self.global_graph is None:
            raise RuntimeError("图为空")

        logger.info("=" * 60)
        logger.info("开始查询: %s", query)
        logger.info("=" * 60)

        # 1) 粗检
        candidate_docs = self.bgem3_retriever.hybrid_doc_retrieval(query)

        candidate_chunks = []
        chunk_ids = []
        for doc in candidate_docs:
            chunks = doc.get("chunks") or []
            for ch in chunks:
                cid = ch.get("id")
                if cid:
                    chunk_ids.append(cid)
                    candidate_chunks.append(ch)

        logger.info("粗检: docs=%d chunks=%d", len(candidate_docs), len(chunk_ids))

        # 2) 子图 G_q（2-hop）
        G_q = self._extract_subgraph(chunk_ids, hops=self.subgraph_hops)
        logger.info("子图: nodes=%d edges=%d", G_q.number_of_nodes(), G_q.number_of_edges())

        # 3) 意图
        h_intent, intent_type = self.intent_repr.build_intent_vector(query, candidate_chunks)
        logger.info("intent_type=%s", intent_type)

        # 4) 语义锚点：只在子图上打分
        sem_result = self.anchor_selector.select_semantic_anchors(G_q, query, h_intent, intent_type)
        s_sem = sem_result["scores"]

        tau_sem = float((self.config.get("semantic_anchor", {}) or {}).get("tau_sem", 0.25))
        anchor_candidates = []
        for nid, sc in s_sem.items():
            if sc >= tau_sem:
                anchor_candidates.append(nid)

        # 全局保底（子图内）
        if not anchor_candidates:
            top_k = int((self.config.get("semantic_anchor", {}) or {}).get("top_k_fallback", 10))
            items = list(s_sem.items())
            items.sort(key=lambda x: x[1], reverse=True)
            anchor_candidates = [nid for nid, _ in items[:top_k]]

        logger.info("语义锚点候选(子图): %d", len(anchor_candidates))

        # 5) 结构排序：只对这些 anchors，在全图上排序（设计A：按类型截断）
        top_by_type = self.centrality_ranker.rank_anchor_nodes(self.global_graph, anchor_candidates, s_sem)

        chunk_anchors = top_by_type["chunk"]
        prop_anchors = top_by_type["proposition"]
        concept_anchors = top_by_type["concept"]
        summary_anchors = top_by_type["summary"]

        logger.info("结构排序结果: chunk=%d prop=%d concept=%d summary=%d",
                    len(chunk_anchors), len(prop_anchors), len(concept_anchors), len(summary_anchors))

        # 6) 邻居扩展
        expanded = self._expand_from_chunk_anchors(chunk_anchors, s_sem)

        # 7) 合并所有节点（锚点 + 扩展）
        all_chunks = set()
        all_props = set()
        all_concepts = set()
        all_summaries = set()

        # 先把结构锚点都加进去
        for nid in chunk_anchors:
            all_chunks.add(nid)
        for nid in prop_anchors:
            all_props.add(nid)
        for nid in concept_anchors:
            all_concepts.add(nid)
        for nid in summary_anchors:
            all_summaries.add(nid)

        # 再加扩展
        for nid in expanded["chunk"]:
            all_chunks.add(nid)
        for nid in expanded["proposition"]:
            all_props.add(nid)
        for nid in expanded["concept"]:
            all_concepts.add(nid)
        for nid in expanded["summary"]:
            all_summaries.add(nid)

        # 8) 组织成 AnswerGenerator 需要的 buckets
        ranked_nodes = {}
        ranked_nodes["chunk"] = self._make_bucket("chunk", all_chunks, s_sem)
        ranked_nodes["proposition"] = self._make_bucket("proposition", all_props, s_sem)
        ranked_nodes["concept"] = self._make_bucket("concept", all_concepts, s_sem)
        ranked_nodes["summary"] = self._make_bucket("summary", all_summaries, s_sem)

        logger.info("最终送入LLM的节点数: chunk=%d prop=%d concept=%d summary=%d",
                    len(ranked_nodes["chunk"]), len(ranked_nodes["proposition"]),
                    len(ranked_nodes["concept"]), len(ranked_nodes["summary"]))

        # 9) 答案生成
        result = self.answer_generator.generate_answer(query, ranked_nodes, self.global_graph)

        result["metadata"] = {
            "intent_type": intent_type,
            "num_candidate_chunks": len(chunk_ids),
            "num_subgraph_nodes": G_q.number_of_nodes(),
            "num_subgraph_edges": G_q.number_of_edges(),
            "num_chunk_anchors": len(chunk_anchors),
        }

        logger.info("查询结束")
        return result

    # -------------------------
    # Subgraph (N-hop)
    # -------------------------
    def _extract_subgraph(self, chunk_ids: List[str], hops: int = 2) -> nx.Graph:
        sub_nodes = set()

        # 起点：粗检出来的 chunks
        for cid inchunk_ids:
            if self.global_graph.has_node(cid):
                sub_nodes.add(cid)

        current = set(sub_nodes)

        for hop in range(hops):
            next_nodes = set()
            for nid in current:
                if not self.global_graph.has_node(nid):
                    continue
                for nb in self.global_graph.neighbors(nid):
                    next_nodes.add(nb)

            for x in next_nodes:
                sub_nodes.add(x)
            current = next_nodes

            logger.info("  hop=%d new_nodes=%d", hop + 1, len(next_nodes))

        return self.global_graph.subgraph(sub_nodes).copy()

    # -------------------------
    # Expand from chunk anchors
    # -------------------------
    def _expand_from_chunk_anchors(self, chunk_anchors: List[str], s_sem: Dict[str, float]):
        """
        对每个 chunk anchor 扩展：
        - 4 个 proposition（HAS_PROP）
        - 2 个 concept（HAS_CONCEPT）
        - 1 个 summary（SUMMARIZED_BY）
        - 2 个 相似 chunk（SIMILAR_TO，跨 doc）
        """
        out = {"chunk": [], "proposition": [], "concept": [], "summary": []}

        for cid in chunk_anchors:
            if not self.global_graph.has_node(cid):
                continue

            props = self._pick_props_for_chunk(cid, s_sem, k=4)
            concepts = self._pick_concepts_for_chunk(cid, k=2)
            summaries = self._pick_summaries_for_chunk(cid, k=1)
            sim_chunks = self._pick_similar_chunks_for_chunk(cid, k=2)

            for x in props:
                out["proposition"].append(x)
            for x in concepts:
                out["concept"].append(x)
            for x in summaries:
                out["summary"].append(x)
            for x in sim_chunks:
                out["chunk"].append(x)

        return out

    def _pick_props_for_chunk(self, chunk_id: str, s_sem: Dict[str, float], k: int):
       """
        遍历 chunk 的所有邻居节点 nb：
        nb 必须是 node_type == "proposition"
        chunk—nb 这条边必须是 edge_type == "HAS_PROP"
        给 nb 一个候选评分：(s_sem[nb], len(text))
        按（语义分，长度）降序排序
        取前 k 个
       """
        cand = []
        for nb in self.global_graph.neighbors(chunk_id):
            if not self.global_graph.has_node(nb):
                continue
            nd = self.global_graph.nodes[nb]
            if nd.get("node_type") != "proposition":
                continue
            # 只 HAS_PROP
            ed = self.global_graph.get_edge_data(chunk_id, nb) or {}
            et = ed.get("edge_type")
            if et != "HAS_PROP":
                continue

            score = s_sem.get(nb, 0.0)
            text = nd.get("text") or ""
            text_len = len(text)

            cand.append((nb, score, text_len))

        # 先按语义分，再按文本长度
        cand.sort(key=lambda x: (x[1], x[2]), reverse=True)

        out = []
        for nb, _, _ in cand[:k]:
            out.append(nb)
        return out

    def _pick_concepts_for_chunk(self, chunk_id: str, k: int):
        """
        邻居 nb 必须是 concept
        边必须是 HAS_CONCEPT
        用边属性 weight 排序取前 k
        """
        cand = []

        for nb in self.global_graph.neighbors(chunk_id):
            nd = self.global_graph.nodes[nb]
            if nd.get("node_type") != "concept":
                continue

            ed = self.global_graph.get_edge_data(chunk_id, nb) or {}
            if ed.get("edge_type") != "HAS_CONCEPT":
                continue

            w = ed.get("weight", 0.0)
            cand.append((nb, float(w)))

        cand.sort(key=lambda x: x[1], reverse=True)

        out = []
        for nb, _ in cand[:k]:
            out.append(nb)
        return out

    def _pick_summaries_for_chunk(self, chunk_id: str, k: int):
        cand = []

        for nb in self.global_graph.neighbors(chunk_id):
            nd = self.global_graph.nodes[nb]
            if nd.get("node_type") != "summary":
                continue

            ed = self.global_graph.get_edge_data(chunk_id, nb) or {}
            if ed.get("edge_type") != "SUMMARIZED_BY":
                continue

            cand.append(nb)

        # summary一般不多
        return cand[:k]

    def _pick_similar_chunks_for_chunk(self, chunk_id: str, k: int):
        """
        取 SIMILAR_TO 的 chunk，要求跨 doc，按 cosine/weight 降序取前 k
        """
        doc_i = self.global_graph.nodes[chunk_id].get("doc_id")

        cand = []
        for nb in self.global_graph.neighbors(chunk_id):
            nd = self.global_graph.nodes[nb]
            if nd.get("node_type") != "chunk":
                continue

            ed = self.global_graph.get_edge_data(chunk_id, nb) or {}
            if ed.get("edge_type") != "SIMILAR_TO":
                continue

            # 跨 doc
            doc_j = nd.get("doc_id")
            if doc_i is not None and doc_j is not None:
                if doc_i == doc_j:
                    continue

            cos = ed.get("cosine")
            if cos is None:
                cos = ed.get("weight", 0.0)

            cand.append((nb, float(cos)))

        cand.sort(key=lambda x: x[1], reverse=True)

        out = []
        for nb, _ in cand[:k]:
            out.append(nb)
        return out

    # -------------------------
    # Build bucket for AnswerGenerator
    # -------------------------
    def _make_bucket(self, node_type: str, node_ids, s_sem: Dict[str, float]):
        """
        返回列表 [{"id","text","score"}, ...]
        score 用语义分
        """
        out = []
        for nid in node_ids:
            if not self.global_graph.has_node(nid):
                continue

            nd = self.global_graph.nodes[nid]
            if nd.get("node_type") != node_type:
                continue

            out.append({
                "id": nid,
                "text": nd.get("text", ""),
                "score": float(s_sem.get(nid, 0.0)),
            })

        # 让 AnswerGenerator 自己再 topK 截断，这里只做一个简单排序
        out.sort(key=lambda x: x["score"], reverse=True)
        return out
