"""
新图构建模块

构建包含 C / P / Z / S 四类节点的异构图（不包含关系节点 R）
- C: chunk
- P: proposition
- Z: concept
- S: summary
"""

import logging
import re
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import networkx as nx
import requests
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# -----------------------------
# Optional ANN backend: hnswlib
# -----------------------------
try:
    import hnswlib  # pip install hnswlib
    _HNSW_AVAILABLE = True
except Exception:
    _HNSW_AVAILABLE = False


def _cosine_matrix_topk_bruteforce(X: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force topk cosine neighbors for ALL points.

    Returns:
        idx: (N, topk) neighbor indices
        sim: (N, topk) cosine similarities
    Note: includes self unless filtered by caller.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    S = Xn @ Xn.T  # (N, N)

    k = min(topk, S.shape[1] - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :topk]

    row = np.arange(S.shape[0])[:, None]
    sim = S[row, idx]
    order = np.argsort(-sim, axis=1)
    idx = idx[row, order]
    sim = sim[row, order]
    return idx, sim


class GraphBuilder:
    def __init__(self, config: Dict):
        """初始化图构建器"""
        self.config = config

        required_keys = [
            "entity_similarity_threshold",
            "concept_num_clusters",
            "proposition_similarity_threshold",
            "concept_similarity_threshold",
            "summary_similarity_threshold",
            "ollama_base_url",
            "llm_model",
        ]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ValueError(f"GraphBuilder 配置缺少必填项: {', '.join(missing)}")

        self.entity_similarity_threshold = config["entity_similarity_threshold"]
        self.concept_num_clusters = config["concept_num_clusters"]
        self.proposition_similarity_threshold = config["proposition_similarity_threshold"]
        self.concept_similarity_threshold = config["concept_similarity_threshold"]
        self.summary_similarity_threshold = config["summary_similarity_threshold"]
        self.ollama_base_url = str(config["ollama_base_url"]).rstrip("/")
        self.llm_model = config["llm_model"]

        # ---- 简单召回与分组参数（可调）----
        self.recall_top_k = int(config.get("recall_top_k", 25))  # embedding 近邻候选
        self.edge_cosine_threshold = float(config.get("edge_cosine_threshold", 0.72))  # 建边门槛
        self.entity_overlap_threshold = int(config.get("entity_overlap_threshold", 1))  # 实体交集个数门槛
        self.cross_doc_only = bool(config.get("cross_doc_only", True))  # 强制跨文档
        self.max_groups = int(config.get("max_groups", 8))  # 最终最大 summary 组数
        self.max_llm_group_input = int(config.get("max_llm_group_input", 12))  # 单次喂给 LLM 的 chunk 数量
        self.embedding_dim = int(config.get("embedding_dim", 1024))  # bge-m3 常见为 1024

        logger.info("GraphBuilder init done (simple recall -> LLM grouping)")

    # -------------------------
    # 通用小工具
    # -------------------------
    def _norm_entity(self, s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)  # 多空格归一
        s = re.sub(r"\s*\(.*?\)\s*$", "", s)  # 去尾部括号解释
        return s

    def _safe_doc_id(self, chunk: Dict) -> str:
        """获取 chunk 的文档ID（尽量稳定）"""
        for k in ("doc_id", "source_doc_id", "document_id", "title"):
            v = chunk.get(k)
            if v:
                return str(v)
        cid = str(chunk.get("id", "unknown"))
        return cid.split("::")[0] if "::" in cid else "unknown_doc"

    def _encode_text(self, text: str) -> List[float]:
        """调用 Ollama embeddings 接口，将文本编码为向量"""
        text = (text or "").strip()
        if not text:
            return np.zeros(self.embedding_dim, dtype=float).tolist()

        url = f"{self.ollama_base_url}/api/embeddings"
        payload = {"model": "bge-m3:latest", "prompt": text}

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            emb = (resp.json() or {}).get("embedding")
            if not emb:
                raise ValueError("Empty embedding in response")

            emb = np.asarray(emb, dtype=float).flatten()
            if emb.shape[0] != self.embedding_dim:
                if emb.shape[0] > self.embedding_dim:
                    emb = emb[: self.embedding_dim]
                else:
                    emb = np.pad(emb, (0, self.embedding_dim - emb.shape[0]))
            return emb.tolist()

        except Exception as e:
            logger.warning(f"编码失败: {e}")
            return np.zeros(self.embedding_dim, dtype=float).tolist()

    def _extract_first_json_object(self, raw: str) -> str:
        """
        从文本中提取第一个完整 JSON object（用大括号计数），避免贪婪解析导致 Extra data
        """
        if not raw:
            raise ValueError("Empty LLM output")

        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        start = s.find("{")
        if start < 0:
            raise ValueError("No '{' found in LLM output")

        depth = 0
        in_str = False
        escape = False

        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]

        raise ValueError("No complete JSON object found (unbalanced braces)")

    # -------------------------
    # Step 1: 从 chunk 抽取 facts & entities
    # -------------------------
    def extract_facts_and_entities_from_chunk(self, chunk: Dict) -> Tuple[List[Dict], List[str]]:
        """使用 LLM 从 chunk 文本中抽取 facts 和实体"""
        text = chunk.get("text", "") or ""
        facts: List[Dict] = []
        entities: List[str] = []

        prompt = f"""# Role
You are a knowledge extraction expert specializing in identifying factual statements, entities, and semantic relationships from text.

# Task
Extract all facts and entities from the given paragraph. For each fact, generate semantic triplets (head entity, predicate, tail entity).

# Instructions
- First identify all important entities (people, places, organizations, concepts, etc.)
- Use the most informative name for each entity (avoid pronouns like "he", "it")
- Each fact should be a complete, standalone statement
- Each triplet format: [head_entity, predicate, tail_entity]
- Output ONLY valid JSON in the specified format

# Example
Input:
Paragraph: "Albert Einstein developed the theory of relativity in 1905. The theory revolutionized physics."

Output:
{{"entities": ["Albert Einstein", "theory of relativity", "physics", "1905"], "facts": {{"f1": {{"fact": "Albert Einstein developed the theory of relativity in 1905.", "triplets": [["Albert Einstein", "developed", "theory of relativity"], ["theory of relativity", "developed in", "1905"]]}}, "f2": {{"fact": "The theory of relativity revolutionized physics.", "triplets": [["theory of relativity", "revolutionized", "physics"]]}}}}}}

# Your Task
Paragraph:
{text}

Output (JSON only):"""

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3},
                },
                timeout=60,
            )
            response.raise_for_status()

            result = (response.json() or {}).get("response", "").strip()
            json_str = self._extract_first_json_object(result)
            data = json.loads(json_str)

            entities = data.get("entities", []) or []

            facts_data = data.get("facts", {}) or {}
            facts = []
            for _, fact_info in facts_data.items():
                if not (isinstance(fact_info, dict) and "fact" in fact_info):
                    continue

                fact_text = fact_info["fact"]
                triplets = fact_info.get("triplets", []) or []

                # fact_entities: 先用“实体列表在 fact 文本中出现”作为粗命中
                fact_entities = []
                ft_low = (fact_text or "").lower()
                for entity in entities:
                    if (entity or "").lower() in ft_low:
                        fact_entities.append(entity)

                # 再从 triplets 中补充（只收集在 entities 列表中出现的）
                for triplet in triplets:
                    if not (isinstance(triplet, list) and len(triplet) >= 3):
                        continue
                    head, _, tail = triplet[0], triplet[1], triplet[2]
                    if head in entities and head not in fact_entities:
                        fact_entities.append(head)
                    if tail in entities and tail not in fact_entities:
                        fact_entities.append(tail)

                facts.append(
                    {
                        "id": f"{chunk.get('id')}_fact_{len(facts)}",
                        "text": fact_text,
                        "entity_ids": fact_entities,
                        "triplets": triplets,
                        "source_chunk_id": chunk.get("id"),
                    }
                )

            if facts and entities:
                logger.info(f"LLM成功提取 {len(facts)} 个facts, {len(entities)} 个实体")
                return facts, entities

            logger.warning("LLM返回了空的facts或entities列表")

        except Exception as e:
            logger.warning(f"extract_facts_and_entities_from_chunk failed: {e}")

        return facts, entities

    # -------------------------
    # Step 2: facts -> propositions
    # -------------------------
    def build_proposition_nodes(self, facts: List[Dict]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        构建命题节点：按实体分桶 -> 桶内拼接 -> 按文本去重

        Returns:
            (propositions, fact_assignments)
            fact_assignments: prop_id -> [fact_id...]
        """
        logger.info(f"构建命题节点: {len(facts)} 个facts")

        # 按实体分桶
        entity_buckets: Dict[str, List[Dict]] = {}
        for fact in facts:
            for entity in (fact.get("entity_ids") or []):
                entity_buckets.setdefault(entity, []).append(fact)

        prop_dedup_map: Dict[str, Dict] = {}

        for _, bucket_facts in entity_buckets.items():
            if not bucket_facts:
                continue

            # 桶内去重拼接
            seen_texts = set()
            unique_fact_texts = []
            for f in bucket_facts:
                t = f.get("text", "")
                if t and t not in seen_texts:
                    unique_fact_texts.append(t)
                    seen_texts.add(t)

            prop_text = " ".join(unique_fact_texts[:10])  # 最多10个不重复fact

            all_entities = set()
            for f in bucket_facts:
                all_entities.update(f.get("entity_ids") or [])

            dedup_key = prop_text  # 保持原逻辑：只按文本去重

            if dedup_key in prop_dedup_map:
                prop_dedup_map[dedup_key]["entity_ids"].update(all_entities)
                prop_dedup_map[dedup_key]["fact_ids"].extend([f["id"] for f in bucket_facts if "id" in f])
            else:
                prop_dedup_map[dedup_key] = {
                    "text": prop_text,
                    "entity_ids": all_entities,
                    "fact_ids": [f["id"] for f in bucket_facts if "id" in f],
                }

        propositions: List[Dict] = []
        fact_assignments: Dict[str, List[str]] = {}

        for i, (_, prop_data) in enumerate(prop_dedup_map.items()):
            prop_id = f"prop_{i}"
            prop_text = prop_data["text"]
            propositions.append(
                {
                    "id": prop_id,
                    "text": prop_text,
                    "entity_ids": list(prop_data["entity_ids"]),
                    "embedding": self._encode_text(prop_text),
                }
            )
            fact_assignments[prop_id] = list(set(prop_data["fact_ids"]))

        logger.info(f"创建了 {len(propositions)} 个命题节点（按文本去重后）")
        return propositions, fact_assignments

    # -------------------------
    # Step 3: propositions/entities -> concepts
    # -------------------------
    def build_concept_nodes_from_props(self, props: List[Dict]) -> List[Dict]:
        """
        基于命题的实体集合聚类生成概念节点（逻辑不变）：
        1) 收集所有实体
        2) 编码 + KMeans 聚类
        3) 每簇用 LLM 生成 1-3词概念名
        """
        if not props:
            return []

        logger.info(f"构建概念节点: {len(props)} 个命题")

        # 1) 收集实体
        all_entities = set()
        for prop in props:
            all_entities.update(prop.get("entity_ids", []) or [])
        all_entities = list(all_entities)
        logger.info(f"收集到 {len(all_entities)} 个唯一实体")

        if not all_entities:
            return []

        # 2) 实体编码
        entity_embeddings = []
        valid_entities = []
        for entity in all_entities:
            try:
                emb = np.asarray(self._encode_text(entity), dtype=float).flatten()
                if emb.shape[0] == self.embedding_dim:
                    entity_embeddings.append(emb)
                    valid_entities.append(entity)
                else:
                    logger.warning(f"实体 '{entity}' embedding维度错误: {emb.shape}, 期望({self.embedding_dim},)")
            except Exception as e:
                logger.warning(f"实体 '{entity}' 编码失败: {e}")

        if not entity_embeddings:
            logger.warning("没有有效的实体embeddings，跳过概念节点构建")
            return []

        all_entities = valid_entities
        entity_embeddings = np.vstack(entity_embeddings)

        # 3) 聚类数：sqrt 自适应
        n_entities = len(all_entities)
        min_k = 20
        max_k = 1000
        n_clusters = int(np.sqrt(n_entities))
        n_clusters = max(min_k, n_clusters)
        n_clusters = min(max_k, n_clusters, n_entities)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(entity_embeddings)

        # 4) 概念节点
        concepts: List[Dict] = []
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            cluster_entities = [all_entities[j] for j in idx]

            concept_text = self._generate_concept_text_from_entities(cluster_entities)

            concepts.append(
                {
                    "id": f"concept_{i}",
                    "text": concept_text,
                    "entity_ids": cluster_entities,
                    "embedding": kmeans.cluster_centers_[i].astype(float).tolist(),
                }
            )

        logger.info(f"创建了 {len(concepts)} 个概念节点")
        return concepts

    def _generate_concept_text_from_entities(self, cluster_entities: List[str]) -> str:
        """使用 LLM 从实体列表生成概念文本（1-3词）"""
        if not cluster_entities:
            return "Concept"

        entities_str = ", ".join(cluster_entities[:10])

        prompt = f"""# Role
You are a concept extraction expert who identifies the core concept from a list of entities.

# Task
Extract a SHORT concept name (1-3 words) that represents the main theme connecting these entities.

# Instructions
- Output ONLY 1-3 words
- Use a noun or noun phrase
- Capture the core concept that connects the entities
- Do NOT write full sentences

# Examples
Entities: Albert Einstein, theory of relativity, physics, Nobel Prize
Output: "Theoretical Physics"

Entities: Paris, France, capital, Eiffel Tower
Output: "French Capital"

Entities: 1905, publication, special relativity
Output: "Relativity Publication"

# Your Task
Entities: {entities_str}

Concept (1-3 words only):"""

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 10, "temperature": 0.3},
                },
                timeout=30,
            )
            response.raise_for_status()
            concept_text = (response.json() or {}).get("response", "").strip()
            if concept_text:
                return concept_text
        except Exception as e:
            logger.warning(f"LLM概念生成失败: {e}")

        # 降级：使用实体列表前三个做提示
        return f"Concept: {', '.join(cluster_entities[:3])}"

    def _generate_concept_text_from_entities(self, cluster_entities: List[str]) -> str:
    """
    使用 LLM 从实体列表生成概念文本（1-3词）

    输入：一个概念簇内的实体列表，例如：
        ["Albert Einstein", "theory of relativity", "physics", "Nobel Prize"]
    输出：一个短概念名，例如：
        "Theoretical Physics"

    目的：把 entity cluster 压缩成可读的 concept label（Z 节点 text）
    """
    if not cluster_entities:
        return "Concept"

    # 只取前 10 个实体，控制 prompt 长度，避免 LLM 输入爆炸
    entities_str = ", ".join(cluster_entities[:10])

    prompt = f"""# Role
You are a concept extraction expert who identifies the core concept from a list of entities.

# Task
Extract a SHORT concept name (1-3 words) that represents the main theme connecting these entities.

# Instructions
- Output ONLY 1-3 words
- Use a noun or noun phrase
- Capture the core concept that connects the entities
- Do NOT write full sentences

# Examples
Entities: Albert Einstein, theory of relativity, physics, Nobel Prize
Output: "Theoretical Physics"

Entities: Paris, France, capital, Eiffel Tower
Output: "French Capital"

Entities: 1905, publication, special relativity
Output: "Relativity Publication"

# Your Task
Entities: {entities_str}

Concept (1-3 words only):"""

    try:
        # 调用 Ollama generate：让 LLM 输出概念名
        response = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                # num_predict=10 基本够 1-3 个词；temperature=0.3 保持一定多样性但不太发散
                "options": {"num_predict": 10, "temperature": 0.3},
            },
            timeout=30,
        )
        response.raise_for_status()
        concept_text = (response.json() or {}).get("response", "").strip()

        # 若 LLM 返回非空，直接使用
        # （建议：你可以额外做一次“词数裁剪/清洗”，防止模型输出带引号或多余句子）
        if concept_text:
            return concept_text
    except Exception as e:
        logger.warning(f"LLM概念生成失败: {e}")

    # 降级策略：不用 LLM 时，用前三个实体拼一个占位概念
    # 注意：这种降级输出并不满足“1-3词”的严格要求，但至少可追踪簇内容
    return f"Concept: {', '.join(cluster_entities[:3])}"


    # -------------------------
    # Summary 节点生成：简单召回 + LLM 分组
    # -------------------------
    def _call_llm_for_summary(self, chunks: List[Dict]) -> str:
        """
        调用 LLM 为一组 chunks 生成 2-3 句摘要，强调逻辑连接

        设计：
        - 只选前 6 个 chunk（控制 prompt 长度）
        - 每个 chunk 最多截取 800 字符
        - 强制 summary 输出包含逻辑连接词（because/therefore/...）
        - 禁止编造 chunk 里没有的事实
        """
        selected = chunks[:6]
        block = []
        for i, c in enumerate(selected, start=1):
            # 给每段 chunk 加上编号与 id，便于 LLM 引用和对齐
            block.append(f"[Chunk {i} | id={c.get('id')}]\n{(c.get('text') or '').strip()[:800]}")
        combined = "\n\n".join(block)

        prompt = f"""# Role
    You are a professional summarizer and reasoning writer.

    # Task
    Write a 2-3 sentence summary that:
    1) captures the key information across the chunks, and
    2) explicitly describes the logical relationship between them.

    # Requirements
    - 2-3 sentences total
    - MUST include explicit logical connectors (because/therefore/however/first...then/as a result/if...then)
    - Do NOT invent facts not present in the chunks

    # Input
    {combined}

    Summary (2-3 sentences):"""

        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    # temperature=0.2 更稳；num_predict=240 足够 2-3 句
                    "options": {"temperature": 0.2, "num_predict": 240},
                },
                timeout=50,
            )
            resp.raise_for_status()
            out = (resp.json() or {}).get("response", "").strip()

            # 若 LLM 输出为空，降级返回截断后的输入（至少不至于空 summary）
            return out if out else combined[:250]
        except Exception as e:
            logger.warning(f"LLM摘要生成失败: {e}")
            return combined[:250]


    def _cosine_topk(self, embs: np.ndarray, top_k: int) -> List[List[Tuple[int, float]]]:
        """
        计算每个向量的 top-k 余弦相似邻居（排除自身）

        输入：
            embs: (N, D) 向量矩阵
            top_k: 每个点取多少近邻
        输出：
            out[i] = [(j, sim_ij), ...]  (长度 top_k)
        """
        # 归一化到单位向量，避免 dot 受模长影响
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        u = embs / norms

        # 相似度矩阵 sim = U * U^T，形状 (N, N)
        sim = u @ u.T

        # 排除自身：把对角线设为 -1，避免 topk 选到自己
        np.fill_diagonal(sim, -1.0)

        out = []
        # top_k 不能超过 N-1
        top_k = min(top_k, sim.shape[0] - 1)

        for i in range(sim.shape[0]):
            # argpartition：O(N) 取前 top_k 大的索引（无序）
            idx = np.argpartition(sim[i], -top_k)[-top_k:]
            # 再按相似度降序排列
            idx = idx[np.argsort(sim[i][idx])[::-1]]
            out.append([(int(j), float(sim[i, j])) for j in idx])
        return out


    def _build_candidate_graph(self, chunks: List[Dict]) -> nx.Graph:
        """
        构建候选相关图 UG（无向图）：

        步骤：
        1) 为每个 chunk 准备 embedding（缺失则编码）
        2) 为每个 chunk 提取实体集合 entity_ids（归一化后放 set）
        3) 对每个 chunk 计算 topK 余弦近邻
        4) 对每个 (i, neighbor) 判断是否建边：
            - cos >= edge_cosine_threshold  或
            - shared_entities_count >= entity_overlap_threshold
        满足任一条件则连边，并记录：
            - weight（优先用 cosine，否则给一个固定权重 0.70）
            - cosine
            - shared_entities（最多 10 个）

        可选：cross_doc_only=True 时，只允许跨文档连边（同 doc 的边被丢弃）
        """
        UG = nx.Graph()
        if not chunks:
            return UG

        # 只有 1 个 chunk：只加节点，不加边
        if len(chunks) < 2:
            for c in chunks:
                UG.add_node(c["id"], doc_id=self._safe_doc_id(c))
            return UG

        ids = [c["id"] for c in chunks]

        # 1) 准备 embedding 矩阵 embs: (N, D)
        embs = []
        for c in chunks:
            if "embedding" not in c or c["embedding"] is None:
                # 缺 embedding 就即时编码：注意这会触发 embeddings API 调用
                c["embedding"] = self._encode_text(c.get("text", ""))
            embs.append(np.asarray(c["embedding"], dtype=float))
        embs = np.vstack(embs)  # (N, D)

        # 2) 为每个 chunk 构建实体集合（归一化后存 set）
        chunk_entities: Dict[str, set] = {}
        for c in chunks:
            raw = c.get("entity_ids", []) or []
            chunk_entities[c["id"]] = {self._norm_entity(x) for x in raw if self._norm_entity(x)}

        # 3) 计算每个 chunk 的 topK 近邻（排除自身）
        topk = self._cosine_topk(embs, top_k=self.recall_top_k)

        # 4) 先把所有节点放入图，并记录 doc_id
        for c in chunks:
            UG.add_node(c["id"], doc_id=self._safe_doc_id(c))

        # 下标 -> chunk_id
        id_at = {i: ids[i] for i in range(len(ids))}

        # 5) 遍历近邻，按阈值规则建边
        for i, cid in enumerate(ids):
            doc_i = UG.nodes[cid]["doc_id"]
            for j, cos in topk[i]:
                nid = id_at[j]
                doc_j = UG.nodes[nid]["doc_id"]

                # 可选：只允许跨文档边
                if self.cross_doc_only and doc_i == doc_j:
                    continue

                # 实体交集
                inter = chunk_entities[cid] & chunk_entities[nid]
                ent_ok = len(inter) >= self.entity_overlap_threshold if self.entity_overlap_threshold > 0 else False

                # 余弦阈值
                cos_ok = cos >= self.edge_cosine_threshold

                # 只要 cosine 或实体交集满足，就建边
                if cos_ok or ent_ok:
                    # 权重策略：cos_ok 用 cosine 作为 weight，否则用固定 0.70（可视为“弱边”）
                    w = float(cos) if cos_ok else 0.70
                    UG.add_edge(
                        cid,
                        nid,
                        weight=w,
                        cosine=float(cos),
                        shared_entities=list(inter)[:10],
                    )

        return UG


    def _initial_groups_by_connected_components(self, UG: nx.Graph) -> List[List[str]]:
        """
        用连通分量得到初始相关簇（connected components）

        输出是按簇大小从大到小排序的 chunk_id 列表集合。
        """
        if UG.number_of_nodes() == 0:
            return []
        comps = [list(c) for c in nx.connected_components(UG)]
        comps.sort(key=len, reverse=True)
        return comps


    def _llm_group_chunk_ids(self, chunk_items: List[Dict], max_groups: int = 8) -> List[List[str]]:
        """
        LLM 对小规模 chunk 集合做“逻辑分组”

        输入:
            chunk_items: [{"id":..., "text":...}, ...]
        输出:
            [[id1, id2], [id3], ...]  （只返回 chunk_id 分组）

        重要约束：
        - 每个 chunk id 必须出现在且只出现在一个 group
        - 组内优先 2-5 个 chunk
        - 最多 max_groups 个组，超了就合并近似组

        失败降级：
        - 若 LLM/JSON 解析失败：每个 chunk 单独成组（singleton）
        """
        if not chunk_items:
            return []
        if len(chunk_items) == 1:
            return [[chunk_items[0]["id"]]]

        # 控制输入长度：每段 text 截断 500 字符
        items = [{"id": x["id"], "text": (x.get("text") or "").strip()[:500]} for x in chunk_items]
        items_json = json.dumps(items, ensure_ascii=False, indent=2)

        prompt = f"""# Role
    You are an expert analyst for chunk understanding.

    # Task
    Group the given chunks into clusters based on LOGICAL relatedness across documents.
    Logical relatedness includes:
    - cause-effect
    - contrast
    - progression
    - condition-conclusion
    - timeline
    - problem-solution
    - other

    # Output Requirements
    - Output ONLY valid JSON.
    - JSON schema:
    {{
    "groups": [
        {{
        "group_id": "g0",
        "relation_type": "cause-effect | contrast | progression | condition-conclusion | timeline | problem-solution | other",
        "rationale": "one short sentence explaining the logical connection",
        "chunk_ids": ["c1", "c7"]
        }}
    ]
    }}
    - Each chunk id must appear in exactly ONE group.
    - Prefer 2-5 chunks per group when possible.
    - If unrelated, keep singleton.
    - Create at most {max_groups} groups; merge the closest groups if needed.

    # Input chunks (JSON):
    {items_json}

    Output (JSON only):"""

        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 700},
                },
                timeout=60,
            )
            resp.raise_for_status()
            raw = (resp.json() or {}).get("response", "").strip()

            # 从可能包含多余文本的 raw 中抽取第一个 JSON 对象字符串
            json_str = self._extract_first_json_object(raw)
            data = json.loads(json_str)

            groups = data.get("groups", []) or []
            if not groups:
                raise ValueError("LLM returned empty groups")

            used = set()
            out: List[List[str]] = []
            all_ids = {x["id"] for x in items}

            # 按 LLM 给出的 groups 顺序收集 chunk_ids，并去重，保证每个 chunk 只被分配一次
            for g in groups:
                ids = g.get("chunk_ids", []) or []
                ids = [cid for cid in ids if cid in all_ids]  # 过滤非法 id
                ids_uniq = []
                for cid in ids:
                    if cid not in used:
                        ids_uniq.append(cid)
                        used.add(cid)
                if ids_uniq:
                    out.append(ids_uniq)

            # 对未被覆盖的 chunk，补单例组，保证“每个 chunk 都在某个 group 里”
            for cid in all_ids:
                if cid not in used:
                    out.append([cid])

            return out

        except Exception as e:
            logger.warning(f"LLM分组失败，降级为每个chunk单独成组: {e}")
            return [[x["id"]] for x in chunk_items]


    def build_summary_nodes_simple_then_llm(self, chunks: List[Dict]) -> List[Dict]:
        """
        Summary 节点生成（简单召回 + LLM 分组）

        Pipeline：
        1) 构建候选相关图 UG（embedding + entity overlap 建边）
        2) 取连通分量得到初始簇（粗分组）
        3) 每个初始簇交给 LLM 做逻辑细分（细分组）
        4) 对每个最终组生成 summary 节点，并记录 source_chunk_ids

        输出：
            summary_nodes: List[Dict]，每个 dict 包含：
                - id
                - text (summary)
                - source_chunk_ids
                - embedding (summary embedding)
        """
        if not chunks:
            return []

        # 1) 构图
        UG = self._build_candidate_graph(chunks)

        # 2) 初始簇：连通分量
        initial_groups = self._initial_groups_by_connected_components(UG)
        if not initial_groups:
            return []

        # 只取前若干个初始簇，避免 LLM 调用爆炸
        initial_groups = initial_groups[: max(1, self.max_groups * 2)]

        # 方便通过 id 快速取 chunk
        id_to_chunk = {c["id"]: c for c in chunks if c.get("id")}
        final_groups: List[List[str]] = []

        # 3) 对每个初始簇做 LLM 细分
        for comp in initial_groups:
            comp = [cid for cid in comp if cid in id_to_chunk]
            if not comp:
                continue

            # 如果簇太大，按 max_llm_group_input 切块，分批喂给 LLM
            # 注意：切块会破坏全局最优分组（因为跨子块的逻辑关系无法被 LLM 看到）
            if len(comp) > self.max_llm_group_input:
                for i in range(0, len(comp), self.max_llm_group_input):
                    sub = comp[i : i + self.max_llm_group_input]
                    items = [{"id": cid, "text": id_to_chunk[cid].get("text", "")} for cid in sub]
                    llm_groups = self._llm_group_chunk_ids(items, max_groups=8)
                    final_groups.extend(llm_groups)
            else:
                items = [{"id": cid, "text": id_to_chunk[cid].get("text", "")} for cid in comp]
                llm_groups = self._llm_group_chunk_ids(items, max_groups=8)
                final_groups.extend(llm_groups)

        # 只保留前 max_groups 个最终组（按组大小降序），避免 summary 节点数量失控
        final_groups = sorted(final_groups, key=len, reverse=True)[: self.max_groups]

        # 4) 生成 summary 节点
        summary_nodes: List[Dict] = []
        for idx, gids in enumerate(final_groups):
            group_chunks = [id_to_chunk[cid] for cid in gids if cid in id_to_chunk]
            if not group_chunks:
                continue

            summary_text = self._call_llm_for_summary(group_chunks)

            summary_nodes.append(
                {
                    "id": f"summary_{idx}",
                    "text": summary_text,
                    "source_chunk_ids": [c["id"] for c in group_chunks],
                    "embedding": self._encode_text(summary_text),
                }
            )

        return summary_nodes


    # -------------------------
    # SIMILAR_TO 边
    # -------------------------
    def add_similarity_edges(self, graph: nx.Graph, node_type: str, top_k: int, sim_threshold: float) -> None:
        """
        在同类型节点之间添加 SIMILAR_TO 边（余弦相似 >= threshold）
        逻辑保持“同类型内部 topK 近邻建边”的常见做法。
        """
        nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == node_type]
        if len(nodes) < 2:
            return

        embs = []
        for n in nodes:
            emb = graph.nodes[n].get("embedding")
            if emb is None:
                txt = graph.nodes[n].get("text", "")
                emb = self._encode_text(txt)
                graph.nodes[n]["embedding"] = emb
            embs.append(np.asarray(emb, dtype=float))
        X = np.vstack(embs)

        neigh = self._cosine_topk(X, top_k=min(top_k, len(nodes) - 1))

        for i, n1 in enumerate(nodes):
            for j, cos in neigh[i]:
                if cos < sim_threshold:
                    continue
                n2 = nodes[j]
                if n1 == n2:
                    continue
                # 去重：无向图只保留一次；若是有向图也没关系
                if graph.has_edge(n1, n2):
                    # 不覆盖已有边类型（例如 HAS_PROP/HAS_CONCEPT），只在没有边时或边类型不同才加
                    # 这里保持谨慎：若已有边则跳过
                    continue
                graph.add_edge(n1, n2, edge_type="SIMILAR_TO", weight=float(cos), cosine=float(cos))

    # -------------------------
    # 主入口：构建异构图
    # -------------------------
    def build_heterogeneous_graph(self, chunks_coref: List[Dict]) -> nx.Graph:
        """
        构建包含 chunk / proposition / concept / summary 四类节点的异构图（不包含关系节点 R）
       
        1) chunks: 抽取 facts/entities + embedding；加入图
        2) propositions: 优先使用预构建，否则由 facts 构建；加入图
        3) C->P: HAS_PROP
        4) concepts: 由 props 生成；加入图
        5) C->Z: HAS_CONCEPT（按实体命中）
        6) summaries: simple recall + LLM grouping；加入图并连 C->S: SUMMARIZED_BY
        7) 同类型 SIMILAR_TO
        """
        graph = nx.Graph()
        if not chunks_coref:
            return graph

        # 1) 加入 chunk 节点，并抽取 facts/entities
        all_facts: List[Dict] = []
        for chunk in chunks_coref:
            facts, entity_ids = self.extract_facts_and_entities_from_chunk(chunk)
            chunk["entity_ids"] = entity_ids
            chunk["facts"] = facts
            all_facts.extend(facts)

            if "embedding" not in chunk or chunk["embedding"] is None:
                chunk["embedding"] = self._encode_text(chunk.get("text", ""))

            graph.add_node(chunk["id"], node_type="chunk", **chunk)

        # 2) 构建 proposition 节点（优先预构建）
        all_propositions: List[Dict] = []
        seen_prop_ids = set()
        for chunk in chunks_coref:
            if "propositions" in chunk and chunk["propositions"]:
                for prop in chunk["propositions"]:
                    pid = prop.get("id")
                    if pid and pid not in seen_prop_ids:
                        all_propositions.append(prop)
                        seen_prop_ids.add(pid)

        if all_propositions:
            logger.info(f"使用预构建的propositions: {len(all_propositions)} 个")
            props = all_propositions
            fact_assignments = {p["id"]: p.get("fact_ids", []) for p in props if p.get("id")}
        else:
            logger.info(f"使用提取的facts构建propositions: {len(all_facts)} 个facts")
            props, fact_assignments = self.build_proposition_nodes(all_facts)

        for prop in props:
            if "embedding" not in prop or prop["embedding"] is None:
                prop["embedding"] = self._encode_text(prop.get("text", ""))
            graph.add_node(prop["id"], node_type="proposition", **prop)

        # 3) C -> P 边
        fact_to_chunk = {f["id"]: f.get("source_chunk_id") for f in all_facts if f.get("id")}
        for prop_id, fact_ids in fact_assignments.items():
            for fact_id in (fact_ids or []):
                chunk_id = fact_to_chunk.get(fact_id)
                if chunk_id:
                    graph.add_edge(chunk_id, prop_id, edge_type="HAS_PROP", weight=1.0)

        # 4) 构建 concept 节点
        concepts = self.build_concept_nodes_from_props(props)
        for concept in concepts:
            graph.add_node(concept["id"], node_type="concept", **concept)

        # 5) C -> Z 边（按实体命中）
        entity_to_concepts: Dict[str, set] = {}
        concept_norm_entities: Dict[str, set] = {}

        for concept in concepts:
            cid = concept["id"]
            norm_set = set()
            for e in (concept.get("entity_ids", []) or []):
                ne = self._norm_entity(e)
                if ne:
                    norm_set.add(ne)
                    entity_to_concepts.setdefault(ne, set()).add(cid)
            concept_norm_entities[cid] = norm_set

        for chunk in chunks_coref:
            chunk_id = chunk["id"]
            raw_entities = graph.nodes[chunk_id].get("entity_ids", []) or []
            chunk_norm_entities = {self._norm_entity(e) for e in raw_entities if self._norm_entity(e)}
            if not chunk_norm_entities:
                continue

            linked_concepts = set()
            for ne in chunk_norm_entities:
                linked_concepts.update(entity_to_concepts.get(ne, set()))

            for cid in linked_concepts:
                overlap = len(chunk_norm_entities & concept_norm_entities.get(cid, set()))
                w = overlap / max(1, len(chunk_norm_entities))
                graph.add_edge(chunk_id, cid, edge_type="HAS_CONCEPT", weight=float(w))

        # 6) 构建 summary 节点
        summary_nodes = self.build_summary_nodes_simple_then_llm(chunks_coref)
        for s in summary_nodes:
            graph.add_node(s["id"], node_type="summary", **s)
            for cid in s.get("source_chunk_ids", []) or []:
                graph.add_edge(cid, s["id"], edge_type="SUMMARIZED_BY", weight=1.0)

        # 7) 添加 SIMILAR_TO 边
        self.add_similarity_edges(graph, "chunk", top_k=5, sim_threshold=0.7)
        self.add_similarity_edges(graph, "proposition", top_k=10, sim_threshold=0.7)
        self.add_similarity_edges(graph, "concept", top_k=5, sim_threshold=0.75)
        self.add_similarity_edges(graph, "summary", top_k=3, sim_threshold=0.7)

        logger.info(f"异构图构建完成: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        return graph
