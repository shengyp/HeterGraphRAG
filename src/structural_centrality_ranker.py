

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import networkx as nx

logger = logging.getLogger(__name__)

class StructuralCentralityRanker:


    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}

        self.gamma_sem = float(cfg.get("gamma_sem", 0.5))

        self.struct_attr = str(cfg.get("struct_attr", "s_struct_global"))


        self.k_prop = int(cfg.get("k_prop", 12))
        self.k_ent = int(cfg.get("k_ent", 12))

        logger.info(
            "StructuralCentralityRanker init | kP=%d kE=%d gamma_sem=%.3f struct_attr=%s",
            self.k_prop,
            self.k_ent,
            self.gamma_sem,
            self.struct_attr,
        )

    def _get_struct_score(self, graph: nx.Graph, node_id: str) -> float:

        if node_id not in graph:
            raise KeyError(f"节点不在图里: {node_id}")

        if self.struct_attr not in graph.nodes[node_id]:
            raise KeyError(
                f"Missing '{self.struct_attr}' on node={node_id}. "
                "请先在离线阶段预计算并写入。"
            )

        val = graph.nodes[node_id][self.struct_attr]
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"'{self.struct_attr}' 必须是数字"
            )
        return float(val)


    def get_struct_scores(self, graph: nx.Graph, nodes: Iterable[str]) -> Dict[str, float]:

        out: Dict[str, float] = {}
        for nid in nodes:
            if nid not in graph:
                continue

            ntype = graph.nodes[nid].get("node_type")
            if ntype not in ("proposition", "entity"):
                continue

            out[nid] = self._get_struct_score(graph, nid)
        return out

    #对输入的 anchor 节点进行打分和排序，分别选出 top-k proposition 和 entity 节点。
    def rank_anchor_nodes(
        self,
        graph: nx.Graph,
        anchors: Iterable[str],
        s_sem: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[str]]:

        result = {"proposition": [],
                  "entity": [], "chunk": [], "type": []}

        if not anchors:
            return result

        s_sem = s_sem or {}
        #用来按节点类型存放 (node_id, score)
        buckets = defaultdict(list)

        for node_id in anchors:
            if node_id not in graph:
                continue

            node_type = graph.nodes[node_id].get("node_type")
            
            if node_type not in ("proposition", "entity"):
                continue

            score = self._get_struct_score(graph, node_id)


            if self.gamma_sem != 0.0:
                if node_id not in s_sem:
                    raise KeyError(
                        f"{node_id没有语义分}. "

                    )
                score += self.gamma_sem * float(s_sem[node_id])

            buckets[node_type].append((node_id, score))


        if buckets.get("proposition"):
            #按分数降序排序
            buckets["proposition"].sort(key=lambda x: x[1], reverse=True)
            #取前 k_prop 个，只保留节点 id
            result["proposition"] = [n for n, _ in buckets["proposition"][: self.k_prop]]

        if buckets.get("entity"):
            # 按分数降序排序
            buckets["entity"].sort(key=lambda x: x[1], reverse=True)
            #同理
            result["entity"] = [n for n, _ in buckets["entity"][: self.k_ent]]

        logger.info("Structural anchors 选择了 | P=%d E=%d", len(result["proposition"]), len(result["entity"]))
        return result