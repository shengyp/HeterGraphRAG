# -*- coding: utf-8 -*-

import logging
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class StructuralCentralityRanker:
    """
    查询时排序器：
    - 读取离线写入的全局结构先验，例如 s_struct_global。
    - 可选融合语义分 s_sem。
    - 输出排序后的 proposition / entity。
    """

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}

        # 最终分数 = 结构权重 * 结构分 + 语义权重 * 语义分
        self.gamma_sem = float(cfg.get("gamma_sem", 0.5))
        self.beta_struct = float(cfg.get("beta_struct", 1.0))

        # 从节点的哪个属性读取结构分。
        self.struct_attr = str(cfg.get("struct_attr", "s_struct_global"))

        # 排序结果截断数量。
        self.k_prop = int(cfg.get("k_prop", 12))
        self.k_ent = int(cfg.get("k_ent", 12))

        logger.info(
            "StructuralCentralityRanker init | kP=%d kE=%d gamma_sem=%.3f beta_struct=%.3f struct_attr=%s",
            self.k_prop,
            self.k_ent,
            self.gamma_sem,
            self.beta_struct,
            self.struct_attr,
        )

    def _get_struct_score(self, graph: nx.Graph, node_id: str) -> float:
        if node_id not in graph:
            raise KeyError(f"节点不在图里: {node_id}")
        return float(graph.nodes[node_id].get(self.struct_attr, 0.0))

    def _score_candidates(
        self,
        graph: nx.Graph,
        candidates: Iterable[str],
        s_sem: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, float, float]]:
        """
        返回 [(node_id, final_score, struct_score, sem_score), ...]
        """
        s_sem = s_sem or {}
        rows: List[Tuple[str, float, float, float]] = []

        for nid in candidates:
            struct_score = self._get_struct_score(graph, nid)
            sem_score = float(s_sem.get(nid, 0.0))
            final_score = self.beta_struct * struct_score + self.gamma_sem * sem_score
            rows.append((nid, final_score, struct_score, sem_score))

        rows.sort(key=lambda x: x[1], reverse=True)
        return rows

    def rank_propositions(
        self,
        graph: nx.Graph,
        prop_candidates: Iterable[str],
        s_sem: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, float, float]]:
        rows = self._score_candidates(graph, prop_candidates, s_sem=s_sem)
        return rows[: self.k_prop]

    def rank_entities(
        self,
        graph: nx.Graph,
        ent_candidates: Iterable[str],
        s_sem: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float, float, float]]:
        rows = self._score_candidates(graph, ent_candidates, s_sem=s_sem)
        return rows[: self.k_ent]

    def rank_anchor_nodes(
        self,
        graph: nx.Graph,
        prop_candidates: Iterable[str],
        ent_candidates: Optional[Iterable[str]] = None,
        s_sem: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Tuple[str, float, float, float]]]:
        """
        统一接口，返回 proposition / entity 两类排序结果。
        """
        result = {
            "proposition": self.rank_propositions(graph, prop_candidates, s_sem=s_sem),
            "entity": [],
        }

        if ent_candidates is not None:
            result["entity"] = self.rank_entities(graph, ent_candidates, s_sem=s_sem)

        return result
