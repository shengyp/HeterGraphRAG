import logging
import requests
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    答案生成器 - 基于多粒度证据选择生成答案

    
    1) 每个类型都选前 K（默认 5），而不是 C/P/Z/S 不同 K
    2) Prompt 明确告诉 LLM：相关 concept 节点是什么、如何帮助定位实体/答案
    3) 更稳：ranked_nodes_by_type 缺 text 时从 graph 回填
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = cfg

        # 统一每类 topK
        self.k_per_type = int(cfg.get("k_per_type", 5))

        # 但现在的目标是“每类都 5”，建议只用 k_per_type。
        self.k_c = int(cfg.get("k_c", self.k_per_type))
        self.k_p = int(cfg.get("k_p", self.k_per_type))
        self.k_z = int(cfg.get("k_z", self.k_per_type))
        self.k_s = int(cfg.get("k_s", self.k_per_type))

        # Prompt 中每条证据的截断长度（避免 prompt 爆）
        self.max_len_prop = int(cfg.get("max_len_prop", 520))
        self.max_len_summary = int(cfg.get("max_len_summary", 420))
        self.max_len_concept = int(cfg.get("max_len_concept", 220))
        self.max_len_chunk = int(cfg.get("max_len_chunk", 650))

        # LLM
        self.llm_model = cfg.get("llm_model", "qwen2.5:14b")
        self.ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")

        logger.info(
            "AnswerGenerator initialized: k_per_type=%d (C=%d,P=%d,Z=%d,S=%d) model=%s",
            self.k_per_type, self.k_c, self.k_p, self.k_z, self.k_s, self.llm_model
        )

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _safe_text(x: str) -> str:
        return (x or "").strip()

    @staticmethod
    def _clip(x: str, n: int) -> str:
        x = (x or "").strip()
        return x[:n] + ("..." if len(x) > n else "")

    @staticmethod
    def _dedup_keep_order(items: List[Dict[str, Any]], key: str = "id"):
        seen = set()
        out = []
        for it in items:
            k = it.get(key)
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    def _normalize_bucket(
        self,
        bucket: List[Dict[str, Any]],
        node_type: str,
        graph,
        topk: int,
    ) -> List[Dict[str, Any]]:
        """
        bucket 元素形如 {"id":..., "text":..., "score":...}
        - 按 score desc
        - text 缺失则从 graph 回填
        - 截断
        """
        if not bucket:
            return []

        # 补全/规整
        normed: List[Dict[str, Any]] = []
        for it in bucket:
            nid = it.get("id")
            if not nid:
                continue
            score = float(it.get("score", 0.0))
            text = self._safe_text(it.get("text", ""))

            if (not text) and graph is not None and hasattr(graph, "nodes") and nid in graph.nodes:
                text = self._safe_text(graph.nodes[nid].get("text", ""))

            normed.append({"id": nid, "text": text, "score": score})

        # 去重 + 排序
        normed = self._dedup_keep_order(normed, key="id")
        normed.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # 截断
        if node_type == "proposition":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_prop)
        elif node_type == "summary":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_summary)
        elif node_type == "concept":
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_concept)
        else:  # chunk
            for it in normed:
                it["text"] = self._clip(it.get("text", ""), self.max_len_chunk)
       
        #按 score 降序排序
        return normed[: max(0, int(topk))]

    # -----------------------------
    # Evidence selection
    # -----------------------------
    def select_evidence_nodes(self, ranked_nodes_by_type: Dict[str, List[Dict[str, Any]]], graph) -> Dict[str, List[Dict[str, Any]]]:
        """
        现在的 ranked_nodes_by_type 来自 RAGSystem：
          buckets["chunk"/"proposition"/"concept"/"summary"] = [{"id","text","score"},...]

       每个类型都分别 topK（默认 5）
        """
        buckets = ranked_nodes_by_type or {}

        evidence = {
            "chunk": self._normalize_bucket(buckets.get("chunk") or [], "chunk", graph, self.k_c),
            "prop": self._normalize_bucket(buckets.get("proposition") or [], "proposition", graph, self.k_p),
            "concept": self._normalize_bucket(buckets.get("concept") or [], "concept", graph, self.k_z),
            "summary": self._normalize_bucket(buckets.get("summary") or [], "summary", graph, self.k_s),
        }

        logger.info(
            "证据选择完成: C=%d, P=%d, Z=%d, S=%d",
            len(evidence["chunk"]),
            len(evidence["prop"]),
            len(evidence["concept"]),
            len(evidence["summary"]),
        )
        return evidence

    # -----------------------------
    # Prompt building
    # -----------------------------
    def build_rag_prompt(self, query: str, evidence: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    构造结构化 RAG Prompt（英文作为实际提示词；中文注释为逐段翻译，便于论文与答辩说明）
    """
    query = self._safe_text(query)
    parts: List[str] = []
   
    # 1) ROLE
    # 中文翻译：
    # 你是一个“基于证据”的问答助手。
    # 你必须仅从所提供的证据中推导答案。
    # 你的目标是输出一个精确、最小化、且直接回答问题的答案。
    parts.append("# Role")
    parts.append(
        "You are an evidence-grounded question-answering assistant. "
        "You MUST derive your answer solely from the provided evidence. "
        "Your goal is to output a precise, minimal answer that directly satisfies the question."
    )

    # 2) TASK
    # 中文翻译：
    # 请仔细阅读问题和证据。
    # 为了可靠作答：
    # 1）识别问题中的最终目标实体（即使是间接指代，例如“其中一位主演”）；
    # 2）识别问题询问的属性类型（例如：因何出名 / 角色 / 定义）；
    # 3）使用证据将目标实体与所询问的属性连接起来。
    # 只返回问题要求的信息，不要多写。
    parts.append("\n# Task")
    parts.append(
        "Read the question and evidence carefully.\n"
        "To answer reliably:\n"
        "1) Identify the ultimate target entity in the question (even if referred indirectly, e.g., 'one of the stars').\n"
        "2) Identify what attribute is being asked (e.g., known for / role / definition).\n"
        "3) Use the evidence to connect the target entity to the requested attribute.\n"
        "Return only the requested information, nothing more."
    )

    # 3) QUESTION
    # 中文翻译：
    # 问题：输出原始问题文本（已做安全规整）。
    parts.append("\n# Question")
    parts.append(query)

    # 4) HOW TO USE CONCEPTS
    # 中文翻译：
    # 概念节点（用于定位正确实体并缩小搜索范围）
    # 以下概念节点本身不是最终答案。
    # 它们的作用是帮助你：
    # - 消歧实体（名字 / 别名）；
    # - 聚焦到正确的电影 / 人物 / 主题；
    # - 选择应当依赖哪些证据段落或事实。
    # 最终答案仍必须从【关键事实 / 总结 / 详细证据】中提取。
    if evidence.get("concept"):
        parts.append("\n# Concepts (for locating the correct entity and narrowing search)")
        parts.append(
            "The following concept nodes are NOT final answers by themselves. "
            "They are cues to help you:\n"
            "- disambiguate entities (names/aliases)\n"
            "- focus on the correct film/person/topic\n"
            "- choose which evidence chunks/facts to rely on\n"
            "You must still extract the final answer from Key Facts / Summaries / Detailed Evidence."
        )
        for i, c in enumerate(evidence["concept"], 1):
            txt = c.get("text", "")
            parts.append(f"{i}. {txt}")

    # 5) KEY FACTS (propositions)
    # 中文翻译：
    # 关键事实（高置信度陈述）。
    # 这些通常是可直接用于回答问题的原子事实语句。
    if evidence.get("prop"):
        parts.append("\n# Key Facts (high-confidence statements)")
        for i, p in enumerate(evidence["prop"], 1):
            parts.append(f"{i}. {p.get('text','')}")

    # 6) SUMMARIES
    # 中文翻译：
    # 总结（高层次压缩证据）。
    # 用于快速把握全局语义与桥接多跳关系，但属于压缩信息。
    if evidence.get("summary"):
        parts.append("\n# Summaries (high-level condensed evidence)")
        for i, s in enumerate(evidence["summary"], 1):
            parts.append(f"{i}. {s.get('text','')}")

    # 7) DETAILED EVIDENCE (chunks)
    # 中文翻译：
    # 详细证据（原文摘录）。
    # 用于提供最忠实的文本依据，在不确定时应以原文为准。
    if evidence.get("chunk"):
        parts.append("\n# Detailed Evidence (verbatim excerpts)")
        for i, ch in enumerate(evidence["chunk"], 1):
            parts.append(f"[{i}] {ch.get('text','')}")

    # 8) REASONING GUIDELINES
    # 中文翻译：
    # 推理规则：
    # - 不使用外部知识；
    # - 若存在多个候选实体，选择最符合问题约束的那个；
    # - 若问题问“其中一位主演”，先从证据中确定主演，再回答其因何出名；
    # - 优先使用证据中出现的直接“known for”表述；
    # - 避免被仅共享关键词但无关的证据误导（例如 newcomers 作为普通词）。
    parts.append("\n# Reasoning Guidelines")
    parts.append(
        "- Do not use external knowledge.\n"
        "- If multiple candidate entities exist, pick the one that matches the question's constraints.\n"
        "- If the question asks 'one of the stars', first identify a star from the evidence, then answer what that star is known for.\n"
        "- Prefer direct 'known for' phrasing when present.\n"
        "- Avoid being distracted by unrelated evidence that shares a keyword (e.g., 'newcomers' as a common word)."
    )

    # 9) ANSWER REQUIREMENTS
    # 中文翻译：
    # 答案要求：
    # - 只输出回答问题所需的特定信息；
    # - 简短（通常几个词）；
    # - 不要复述问题；
    # - 不要解释；
    # - 不要给多个答案；
    # - 答案必须完全由证据支持。
    parts.append("\n# Answer Requirements")
    parts.append(
        "- Output only the specific information that answers the question.\n"
        "- Be brief (usually a few words).\n"
        "- Do NOT restate the question.\n"
        "- Do NOT explain.\n"
        "- Do NOT provide multiple answers.\n"
        "- The answer must be fully supported by the evidence."
    )

    # Final Answer
    # 中文翻译：
    # 最终答案：从这里开始输出答案正文（不要附加解释）。
    parts.append("\n# Final Answer:")
    return "\n".join(parts)

    # -----------------------------
    # LLM call
    # -----------------------------
    def generate_answer_with_rag(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(self.config.get("temperature", 0.3)),
                    "num_predict": int(self.config.get("num_predict", 200)),
                },
            },
            timeout=int(self.config.get("timeout", 60)),
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    # -----------------------------
    # Public entry
    # -----------------------------
    def generate_answer(self, query: str, ranked_nodes_by_type: Dict[str, List[Dict[str, Any]]], graph) -> Dict[str, Any]:
        logger.info("开始生成答案: query='%s...'", (query or "")[:80])

        evidence = self.select_evidence_nodes(ranked_nodes_by_type, graph)
        prompt = self.build_rag_prompt(query, evidence)
        answer = self.generate_answer_with_rag(prompt)

        logger.info("答案生成完成")
        return {"answer": answer, "supporting_nodes": evidence, "prompt": prompt}
