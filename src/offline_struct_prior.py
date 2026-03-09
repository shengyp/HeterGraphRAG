
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import networkx as nx


def _minmax_norm(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def build_ep_skeleton(G: nx.Graph) -> nx.Graph:

    H = nx.Graph()

    # 1) 加节点
    for n, data in G.nodes(data=True):
        t = data.get("node_type")
        if t in ("proposition", "entity"):
            H.add_node(n, **data)

    # 2) 加边（只收 P-E）
    for u, v, edata in G.edges(data=True):
        if u not in H or v not in H:
            continue
        tu = H.nodes[u].get("node_type")
        tv = H.nodes[v].get("node_type")
        if {tu, tv} == {"proposition", "entity"}:
            H.add_edge(u, v, **edata)

    return H


def compute_and_write_struct_prior(
    G: nx.Graph,
    *,
    alpha_core: float = 0.5,
    alpha_betw: float = 0.5,
    attr: str = "s_struct_global",
) -> None:

    H = build_ep_skeleton(G)

    # core number（k-core）
    core = nx.core_number(H) if H.number_of_nodes() > 0 else {}
    core = {k: float(v) for k, v in core.items()}

    # betweenness（无权）
    betw = nx.betweenness_centrality(H, normalized=True) if H.number_of_nodes() > 0 else {}
    betw = {k: float(v) for k, v in betw.items()}

    core_n = _minmax_norm(core)
    betw_n = _minmax_norm(betw)

    for n in H.nodes():
        score = alpha_core * core_n.get(n, 0.0) + alpha_betw * betw_n.get(n, 0.0)
        G.nodes[n][attr] = float(score)