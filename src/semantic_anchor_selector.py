"""
LLM语义锚点选择模块

混合打分：
s_total = α * s_intent + β * s_text + γ * s_llm

说明：
- 这里返回“所有被评估节点”的分数表 {node_id: s_total}
- 是否过阈值、是否TopK保底，由RAGSystem 控制
"""

import logging
import re
import numpy as np
import requests

logger = logging.getLogger(__name__)


class SemanticAnchorSelector:
    """混合打分语义锚点选择器"""

    def __init__(self, config):
        
        cfg = config["semantic_anchor"]
       

        self.config = cfg
        self.ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        self.llm_model = cfg.get("llm_model", "qwen2.5:14b")

        
        self.tau_sem = float(cfg.get("tau_sem", 0.25))
        self.top_k_fallback = int(cfg.get("top_k_fallback", 5))

        # 权重配置（移除beta_text）
        self.alpha_intent = float(cfg.get("alpha_intent", 0.5))
        self.gamma_llm = float(cfg.get("gamma_llm", 0.5))

        # 每种类型最多评估多少个
        self.max_props = int(cfg.get("max_props", 10))
        self.max_concepts = int(cfg.get("max_concepts", 5))
        self.max_summaries = int(cfg.get("max_summaries", 5))

        logger.info(
            "SemanticAnchorSelector initialized: tau=%.3f, top_k=%d, weights(intent=%.2f, llm=%.2f)",
            self.tau_sem,
            self.top_k_fallback,
            self.alpha_intent,
            self.gamma_llm,
        )

  
    # 基础工具：余弦相似度 / 文本相似度


    def cosine_sim(self, a, b):
        """计算余弦相似度（维度不匹配或零向量则返回0）"""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        if a.shape[0] != b.shape[0]:
            return 0.0

        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (na * nb))

    def text_similarity(self, query, text):
        """
        文本相似度：用简化的 Jaccard 关键词重叠
        注意：这不是最强方案，但成本低，适合做轻量信号
        """
        q_words = set((query or "").lower().split())
        t_words = set((text or "").lower().split())

        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "for"}
        q_words -= stop
        t_words -= stop

        if not q_words:
            return 0.0

        inter = len(q_words & t_words)
        uni = len(q_words | t_words)
        return (inter / uni) if uni else 0.0

    # -----------------------------
    # LLM 评分（英文prompt）
    # -----------------------------

    def llm_score_node_as_anchor(self, query, intent_type, node_type, node_text):
        """
        调用LLM，输出 0.0~1.0 的相关性分数
        失败则降级为关键词重叠比例
        """
        node_text = (node_text or "")[:300]

        prompt = f"""# Role
You are a relevance assessment expert who evaluates how useful information is for answering questions.

# Task
Evaluate whether this node's information would be directly cited or strongly relied upon when answering the query.

# Scoring Guidelines
- 1.0: Essential information, directly answers the query
- 0.7-0.9: Highly relevant, provides key supporting information
- 0.4-0.6: Moderately relevant, provides context
- 0.1-0.3: Weakly relevant, tangentially related
- 0.0: Not relevant

# Example
Query: "Who developed the theory of relativity?"
Intent: FACTUAL
Node Type: chunk
Node Content: "Albert Einstein developed the theory of relativity in 1905."
Score: 0.95

Query: "Who developed the theory of relativity?"
Intent: FACTUAL
Node Type: chunk
Node Content: "Einstein was born in Germany in 1879."
Score: 0.3

# Your Task
Query: {query}
Query Intent: {intent_type}
Node Type: {node_type}
Node Content:
{node_text}

Relevance Score (0.0-1.0):"""

        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 10, "temperature": 0.3},
                },
                timeout=int(self.config.get("llm_score_timeout", 15)),
            )
            if resp.status_code != 200:
                raise RuntimeError(f"non-200: {resp.status_code}")

            out = (resp.json().get("response") or "").strip()

            # 从输出里抓一个 [0,1] 的数字（只取第一个，避免花哨输出）
            nums = re.findall(r"(?:0\.\d+|1\.0|1|0)", out)
            if nums:
                v = float(nums[0])
                return max(0.0, min(v, 1.0))

        except Exception as e:
            logger.debug("LLM scoring failed (fallback to keyword overlap): %s", e)

        # 降级：简单重叠比例
        q = set((query or "").lower().split())
        t = set((node_text or "").lower().split())
        overlap = len(q & t)
        return min(overlap / max(len(q), 1), 1.0)



    def select_semantic_anchors(self, graph, query, h_intent, intent_type):
        """
        输入：
        - graph：子图 G_q
        - query：查询文本
        - h_intent：统一意图向量（一般是 [h_t; h_q; h_ctx]）
        - intent_type：意图类型（FACTUAL/DEFINITION/CAUSAL/COMPARISON）

        输出：
        - s_scores：{node_id: s_total}（被评估节点的混合得分）
        """
        n_nodes = graph.number_of_nodes()
        logger.info("开始混合打分锚点选择：nodes=%d, intent_type=%s", n_nodes, intent_type)

        # 1) 从 h_intent 中尽量提取 h_q（如果拼接规则是 3段等长）
        h_q = self._extract_h_q(h_intent)

        # 2) 把节点按类型分组（只对关心的几类做配额）
        nodes_by_type = {"chunk": [], "proposition": [], "concept": [], "summary": []}
        for node_id, node_data in graph.nodes(data=True):
            t = node_data.get("node_type", "unknown")
            if t in nodes_by_type:
                nodes_by_type[t].append((node_id, node_data))

        # 3) 决定要评估哪些节点（控制LLM成本）
        #    策略：chunk 全评估；其他类型先做轻量排序，再取前 N
        nodes_to_eval = []
        nodes_to_eval.extend(nodes_by_type["chunk"])

        nodes_to_eval.extend(self._pick_top_by_lightweight_score(nodes_by_type["proposition"], query, self.max_props))
        nodes_to_eval.extend(self._pick_top_by_lightweight_score(nodes_by_type["concept"], query, self.max_concepts))
        nodes_to_eval.extend(self._pick_top_by_lightweight_score(nodes_by_type["summary"], query, self.max_summaries))

        logger.info(
            "选择评估节点：%d 个 (chunk=%d全量, prop<=%d, concept<=%d, summary<=%d)",
            len(nodes_to_eval),
            len(nodes_by_type["chunk"]),
            self.max_props,
            self.max_concepts,
            self.max_summaries,
        )

        # 4) 开始打分
        s_scores = {}
        for i, (node_id, node_data) in enumerate(nodes_to_eval):
            if i % 10 == 0:
                logger.info("评分进度：%d/%d", i, len(nodes_to_eval))

            node_text = node_data.get("text", "")
            node_type = node_data.get("node_type", "unknown")

            # s_intent：embedding 与 h_q 的余弦
            node_emb = node_data.get("embedding")
            s_intent = self.cosine_sim(node_emb, h_q) if node_emb is not None else 0.0

            # s_llm：LLM语义判别
            s_llm = self.llm_score_node_as_anchor(query, intent_type, node_type, node_text)

            # 混合打分（移除s_text）
            s_total = self.alpha_intent * s_intent + self.gamma_llm * s_llm
            s_scores[node_id] = float(s_total)

            if i < 5:
                logger.info(
                    "node=%s type=%s: intent=%.3f llm=%.3f -> total=%.3f",
                    str(node_id)[:30],
                    node_type,
                    s_intent,
                    s_llm,
                    s_total,
                )

        logger.info("混合打分完成：scored_nodes=%d", len(s_scores))
        return s_scores

    def _extract_h_q(self, h_intent):
      
        try:
            vec = np.array(h_intent, dtype=np.float32)
            if vec.shape[0] % 3 == 0 and vec.shape[0] >= 3:
                seg = vec.shape[0] // 3
                return vec[seg : 2 * seg]
            return vec
        except Exception:
            return np.array(h_intent, dtype=np.float32)

    def _pick_top_by_lightweight_score(self, nodes, query, k):
        """
        对候选节点做轻量排序再截断：
        - 轻量分 = text_similarity(query, node_text)
        这样比“直接取前k个”更合理，也更像工程代码
        """
        if not nodes or k <= 0:
            return []

        scored = []
        for node_id, node_data in nodes:
            score = self.text_similarity(query, node_data.get("text", ""))
            scored.append((score, node_id, node_data))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [(nid, nd) for _, nid, nd in scored[:k]]
        return top
