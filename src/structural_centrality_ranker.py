import logging
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class StructuralCentralityRanker:
    """
    结构排序（设计 A）：
    - 在全图上计算结构中心性（k-core + betweenness）
    - 只对传入的 anchor 节点取分
    - 按类型分别排序并截断：
        chunk=6, proposition=6, concept=3, summary=3
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}

        self.alpha_core = float(cfg.get("alpha_core", 0.5))
        self.alpha_betw = float(cfg.get("alpha_betw", 0.5))
        self.gamma_sem = float(cfg.get("gamma_sem", 0.0))

        self.k_chunk = int(cfg.get("k_chunk", 6))
        self.k_prop = int(cfg.get("k_prop", 6))
        self.k_concept = int(cfg.get("k_concept", 3))
        self.k_summary = int(cfg.get("k_summary", 3))

        self.allowed_types = ["chunk", "proposition", "concept", "summary"]

        logger.info(
            "StructuralCentralityRanker init: "
            "kC=%d kP=%d kZ=%d kS=%d",
            self.k_chunk, self.k_prop, self.k_concept, self.k_summary
        )

    def _compute_structural_scores(self, graph: nx.Graph):
        """
        在整个图上计算每个节点的结构分
        """
        scores = {}

        if graph.number_of_nodes() == 0:
            return scores

        g = graph.to_undirected()

        core_num = nx.core_number(g)
        betw = nx.betweenness_centrality(g)

        # -------- core 归一化 --------
        max_core = 0
        for v in core_num.values():
            if v > max_core:
                max_core = v
        if max_core == 0:
            max_core = 1

        # -------- betweenness 归一化 --------
        max_betw = 0.0
        for v in betw.values():
            if v > max_betw:
                max_betw = v
        if max_betw == 0.0:
            max_betw = 1.0

        # -------- 合成结构分 --------
        for node in g.nodes():
            core_score = core_num.get(node, 0) / max_core
            betw_score = betw.get(node, 0.0) / max_betw
            scores[node] = (
                self.alpha_core * core_score
                + self.alpha_betw * betw_score
            )

        return scores

    def rank_anchor_nodes(self, graph: nx.Graph, anchors, s_sem=None):
        """
        输入：
            anchors: 语义锚点（来自子图）
            s_sem: {node_id: semantic_score}
        输出：
            {
              "chunk": [...],
              "proposition": [...],
              "concept": [...],
              "summary": [...]
            }
        """
        result = {
            "chunk": [],
            "proposition": [],
            "concept": [],
            "summary": [],
        }

        if not anchors:
            return result

        if s_sem is None:
            s_sem = {}

        struct_scores = self._compute_structural_scores(graph)

        # 1) 给 anchor 打总分
        buckets = defaultdict(list)

        for node_id in anchors:
            if node_id not in graph:
                continue

            node_type = graph.nodes[node_id].get("node_type")
            if node_type not in self.allowed_types:
                continue

            score = struct_scores.get(node_id, 0.0)

            if self.gamma_sem != 0.0:
                score += self.gamma_sem * s_sem.get(node_id, 0.0)

            buckets[node_type].append((node_id, score))

        # 2) 每类排序
        for node_type in buckets:
            buckets[node_type].sort(key=lambda x: x[1], reverse=True)

        # 3) 每类截断
        result["chunk"] = [n for n, _ in buckets.get("chunk", [])[: self.k_chunk]]
        result["proposition"] = [n for n, _ in buckets.get("proposition", [])[: self.k_prop]]
        result["concept"] = [n for n, _ in buckets.get("concept", [])[: self.k_concept]]
        result["summary"] = [n for n, _ in buckets.get("summary", [])[: self.k_summary]]

        logger.info(
            "Structural anchors selected: C=%d P=%d Z=%d S=%d",
            len(result["chunk"]),
            len(result["proposition"]),
            len(result["concept"]),
            len(result["summary"]),
        )

        return result
