"""
新意图表征模块（升级版）

构造统一查询意图向量: h_intent = [h_t; h_q; h_ctx]

意图类型（新版三分类）：
- BRIDGING（桥接型）：典型多跳/跨文档桥接问题，需要先找“中间实体/桥梁证据”再到答案。
- COMPARISON（比较型）：比较/对比多个对象（相同点、不同点、优劣、区别）。
- LOGICAL_REASONING（逻辑推理型/分析题）：需要推理、归纳、计算、条件推导、因果链解释等分析过程。

说明：
- h_q: query embedding
- h_ctx: 精选候选 chunks 的语义摘要（平均池化；你可后续改成加权池化）
- h_t: 意图类型描述模板 embedding（把“意图”也投到语义空间）
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
import requests

logger = logging.getLogger(__name__)


class IntentRepresentation:
    """意图表征器 - 构造统一查询意图向量（新版三分类）"""

    def __init__(self, config: Dict):
        """
        初始化意图表征器

        Args:
            config: 配置字典
                - ollama_base_url: embeddings / generate 的 Ollama 服务地址
                - llm_model: 用于意图分类的 LLM（例如 qwen2.5:7b）
                - embedding_model: 用于编码的 embedding 模型（默认 bge-m3）
                - max_ctx_chunks: 参与 h_ctx 的最多 chunk 数（默认 5）
                - cls_timeout: 意图分类接口超时（默认 10s）
                - emb_timeout: embedding 接口超时（默认 30s）
        """
        self.config = config or {}
        self.ollama_base_url = self.config.get("ollama_base_url", "http://localhost:11434")
        self.llm_model = self.config.get("llm_model", "qwen2.5:14b")
        self.embedding_model = self.config.get("embedding_model", "bge-m3")

        self.max_ctx_chunks = int(self.config.get("max_ctx_chunks", 20))
        self.cls_timeout = int(self.config.get("cls_timeout", 10))
        self.emb_timeout = int(self.config.get("emb_timeout", 30))


        self.intent_templates = {
            "BRIDGING": (
                "This is a bridging multi-hop question. It requires finding an intermediate entity or "
                "bridge evidence first, then using it to reach the final answer across documents."
            ),
            "COMPARISON": (
                "This is a comparison question. It asks to compare or contrast two or more entities, "
                "highlighting similarities, differences, or relative properties."
            ),
            "LOGICAL_REASONING": (
                "This is a logical reasoning or analysis question. It requires reasoning steps such as "
                "deduction, induction, calculations, causal-chain explanation, or constraint-based inference."
            ),
        }

        logger.info("意图表征器初始化完成")

    # -------------------------
    # Embedding
    # -------------------------
    def _encode_text(self, text: str) -> np.ndarray:
        """使用 embedding_model（默认 BGE-M3）编码文本为 1024d 向量。"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=self.emb_timeout,
            )

            if response.status_code != 200:
                logger.error(f"Embeddings API失败: status={response.status_code}, response={response.text}")
                return np.zeros(1024, dtype=np.float32)

            result = response.json()
            embedding = result.get("embedding")
            if embedding is None:
                logger.error(f"Embeddings API返回格式错误: {result}")
                return np.zeros(1024, dtype=np.float32)

            emb_array = np.asarray(embedding, dtype=np.float32)

            # 检查维度
            if emb_array.ndim != 1 or emb_array.shape[0] != 1024:
                logger.error(f"Embedding维度错误: {emb_array.shape}, 预期(1024,)")
                return np.zeros(1024, dtype=np.float32)

            # 检查是否是零向量
            if np.linalg.norm(emb_array) < 1e-6:
                logger.warning("生成的embedding是零向量（norm≈0）")

            return emb_array

        except Exception as e:
            logger.error(f"编码失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(1024, dtype=np.float32)

    # -------------------------
    # Query + Context encoding
    # -------------------------
    def encode_query_and_context(self, query: str, coarse_chunks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对查询和精选候选chunk上下文进行编码

        coarse_chunks：已经过初步筛选后的精选候选 chunks（数量较少且更相关）

        Returns:
            (h_q, h_ctx)
        """
        # 1) query embedding
        h_q = self._encode_text(query)

        # 2) context embedding：最多取 max_ctx_chunks
        chunk_embeddings: List[np.ndarray] = []
        for chunk in (coarse_chunks or [])[: self.max_ctx_chunks]:
 
            emb = chunk.get("embedding", None)

            if emb is not None:
                emb_arr = np.asarray(emb, dtype=np.float32)
                if emb_arr.ndim == 1 and emb_arr.shape[0] == 1024 and np.linalg.norm(emb_arr) >= 1e-6:
                    chunk_embeddings.append(emb_arr)
                    continue 

            text = chunk.get("text", "")
            if text:
                chunk_embeddings.append(self._encode_text(text))

        if chunk_embeddings:
            h_ctx = np.mean(np.stack(chunk_embeddings, axis=0), axis=0)
        else:
            h_ctx = np.zeros_like(h_q)

        return h_q, h_ctx

    # -------------------------
    # Intent classification (3-way)
    # -------------------------
    def classify_intent_type(self, query: str, coarse_chunks: List[Dict]) -> str:
        """
        基于 query预测意图类型

        Returns:
            "BRIDGING" | "COMPARISON" | "LOGICAL_REASONING"
        """

        # 取少量 ctx snippet 作为消歧信号
        ctx_snippets = []
        for c in (coarse_chunks or [])[:3]:
            t = (c.get("text") or "").strip().replace("\n", " ")
            if t:
                ctx_snippets.append(t[:250])
        ctx_block = "\n".join([f"- {s}" for s in ctx_snippets]) if ctx_snippets else "(none)"

        prompt = f"""# Role

You are a question analysis expert who classifies questions by their intent type.

# Task

Classify the intent type of the given question into one of three categories. Output one label only.

# Intent Types
- BRIDGING: multi-hop bridging questions requiring an intermediate entity or bridge evidence
- COMPARISON: compare/contrast two or more entities or properties
- LOGICAL_REASONING: analysis requiring reasoning steps (deduction/induction/calculation/causal chain/inference)

# Examples

# Which actor starred in Film A and later directed Film B? → BRIDGING
Question: "Which actor starred in Film A and later directed Film B?"
Answer: BRIDGING

# How is Python different from Java? → COMPARISON
Question: "How is Python different from Java?"
Answer: COMPARISON

# If all A are B and some B are C, what follows? → LOGICAL_REASONING
Question: "If all A are B and some B are C, what follows?"
Answer: LOGICAL_REASONING

# Context (optional)
# The following are a few evidence snippets that the system currently considers relevant. 
They are included solely for disambiguation purposes and should not override the intent implied by the question itself.
{ctx_block}

# Your Task
 #Classify the following question and output exactly one label from: BRIDGING, COMPARISON, LOGICAL_REASONING.”
Question: {query}

Answer (one label only):"""

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 8, "temperature": 0.2},
                },
                timeout=self.cls_timeout,
            )

            if response.status_code == 200:
                raw = (response.json().get("response", "") or "").strip().upper()

              
                if "BRIDGING" in raw:
                    return "BRIDGING"
                if "COMPARISON" in raw:
                    return "COMPARISON"
                if "LOGICAL_REASONING" in raw or "LOGICAL" in raw or "REASONING" in raw:
                    return "LOGICAL_REASONING"

        except Exception as e:
            logger.warning(f"意图分类失败: {e}")

   
        return "BRIDGING"

    # -------------------------
    # Intent type encoding
    # -------------------------
    def encode_intent_type(self, intent_type: str) -> np.ndarray:
        """
        将意图类型对应的描述性模板编码为向量

        Returns:
            h_t
        """
        intent_type = (intent_type or "").strip().upper()
        template = self.intent_templates.get(intent_type, self.intent_templates["BRIDGING"])
        return self._encode_text(template)

    # -------------------------
    # Build unified intent vector
    # -------------------------
    def build_intent_vector(self, query: str, coarse_chunks: List[Dict]) -> Tuple[np.ndarray, str]:
        """
        构造统一查询意图向量: h_intent = [h_t; h_q; h_ctx]

        Returns:
            (h_intent, intent_type)
        """
        logger.info(f"构造意图向量: query='{(query or '')[:50]}...'")

        # 1) 编码 query + ctx
        h_q, h_ctx = self.encode_query_and_context(query, coarse_chunks)

        # 2) LLM 意图分类
        intent_type = self.classify_intent_type(query, coarse_chunks)

        # 3) 编码意图类型模板
        h_t = self.encode_intent_type(intent_type)

        # 4) 拼接
        h_intent = np.concatenate([h_t, h_q, h_ctx], axis=0)

        logger.info(f"意图向量构造完成: intent_type={intent_type}, dim={h_intent.shape[0]}")
        return h_intent, intent_type
