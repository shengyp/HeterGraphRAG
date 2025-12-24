"""
结构中心度排序模块（朴素版，在子图G_q上算结构分）

目标：
- 在子图 G_q 上计算结构分（k-core + betweenness）
- 只对 anchors 这批候选锚点做排序，返回 top-k0
"""

import logging
import networkx as nx

logger = logging.getLogger(__name__)


class StructuralCentralityRanker:
    """结构中心度排序器"""

    def __init__(self, config):
        cfg = config or {}
        self.config = cfg
        self.alpha_core = float(cfg.get("alpha_core", 0.5))
        self.alpha_betw = float(cfg.get("alpha_betw", 0.5))
        self.k0 = int(cfg.get("k0", 10))

        logger.info(
            "StructuralCentralityRanker initialized: alpha_core=%.2f, alpha_betw=%.2f, k0=%d",
            self.alpha_core,
            self.alpha_betw,
            self.k0,
        )

    def rank_anchor_nodes(self, graph, anchors, s_sem=None):
        """对 anchors 在子图上做结构排序，返回 top-k0 的锚点id列表"""
        logger.info("开始锚点结构排序：anchors=%d", len(anchors))
        if not anchors:
            return []

        s_struc = self.compute_structural_importance(graph, anchors)
        sorted_anchors = sorted(s_struc.items(), key=lambda x: x[1], reverse=True)
        top_anchors = [nid for nid, _ in sorted_anchors[: self.k0]]

        logger.info("锚点结构排序完成：返回 top-%d", len(top_anchors))
        return top_anchors

    def compute_structural_importance(self, graph, anchors):
        """
        在子图上计算结构重要性分数
        步骤：
        1) 在 G_q 上转无向图
        2) 在无向图上计算 core_number 与 betweenness
        3) 归一化后加权得到每个节点的结构分
        4) 只返回 anchors 的结构分
        """
        logger.info("计算结构重要性：anchors=%d, subgraph_nodes=%d", len(anchors), graph.number_of_nodes())

        if graph.number_of_nodes() == 0:
            return {a: 0.0 for a in anchors}

        undirected = graph.to_undirected()

        core_num = nx.core_number(undirected)
        betw = nx.betweenness_centrality(undirected)

        max_core = max(core_num.values()) if core_num else 1
        max_betw = max(betw.values()) if betw else 1
        if max_core <= 0:
            max_core = 1
        if max_betw <= 0:
            max_betw = 1

        s_all = {}
        for n in undirected.nodes():
            core_norm = core_num.get(n, 0) / max_core
            betw_norm = betw.get(n, 0.0) / max_betw
            s_all[n] = self.alpha_core * core_norm + self.alpha_betw * betw_norm

        s_anchor = {}
        for a in anchors:

            s_anchor[a] = float(s_all.get(a, 0.0))

        return s_anchor
