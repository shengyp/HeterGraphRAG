"""
新图构建模块

构建包含C/P/Z/S四类节点的异构图（不包含关系节点R）
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
import requests

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, config):
        """初始化图构建器"""
        self.config = config

        # 1. 先定义所有“必填配置项”的名字
        required_keys = [
            "entity_similarity_threshold",
            "concept_num_clusters",
            "proposition_similarity_threshold",
            "concept_similarity_threshold",
            "summary_similarity_threshold",
            "ollama_base_url",
            "llm_model",
        ]

        # 2. 检查这些 key 是否都在 config 里
        missing = [k for k in required_keys if k not in config]
        if missing:
            
            raise ValueError(
                f"GraphBuilder 配置缺少必填项: {', '.join(missing)}"
            )

       
        self.entity_similarity_threshold = config["entity_similarity_threshold"]
        self.concept_num_clusters = config["concept_num_clusters"]
        self.proposition_similarity_threshold = config["proposition_similarity_threshold"]
        self.concept_similarity_threshold = config["concept_similarity_threshold"]
        self.summary_similarity_threshold = config["summary_similarity_threshold"]
        self.ollama_base_url = config["ollama_base_url"]
        self.llm_model = config["llm_model"]

        
        logger.info("新图构建器初始化完成")
    
    def _encode_text(self, text) -> np.ndarray:
        """使用BGE-M3编码文本"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": "bge-m3",
                    "prompt": text
                }
            )
            #检查http状态码
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)
        except Exception as e:
            logger.error(f"编码失败: {e}")
            #这里假设 bge-m3 输出维度固定 1024
            return np.zeros(1024)
    
    def extract_facts_and_entities_from_chunk(self, chunk: Dict) -> Tuple[List[Dict], List[str]]:
        """
        使用LLM从chunk中抽取facts和实体
        
        Args:
            chunk: chunk字典
            
        Returns:
            (fact列表, 实体列表)
        """
        text = chunk["text"]
        facts=[] 
        entities=[]
        
        
        
        # 构造prompt
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
                    "options": {
                        "temperature": 0.3
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip()
                
               
                import json
                import re
                
                
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    entities = data.get("entities", [])
                    
                    # 提取facts
                    facts_data = data.get("facts", {})
                    facts = []
                    
                    for fact_id, fact_info in facts_data.items():
                        if isinstance(fact_info, dict) and "fact" in fact_info:
                            fact_text = fact_info["fact"]
                            triplets = fact_info.get("triplets", [])
                            
                            # 提取涉及的实体
                            fact_entities = []
                            for entity in entities:
                                if entity.lower() in fact_text.lower():
                                    fact_entities.append(entity)
                            
                            # 从triplets中提取实体
                            for triplet in triplets:
                                if len(triplet) >= 3:
                                    head, pred, tail = triplet[0], triplet[1], triplet[2]
                                    if head not in fact_entities and head in entities:
                                        fact_entities.append(head)
                                    if tail not in fact_entities and tail in entities:
                                        fact_entities.append(tail)
                            
                            facts.append({
                                "id": f"{chunk['id']}_fact_{len(facts)}",
                                "text": fact_text,
                                "entity_ids": fact_entities,
                                "triplets": triplets,
                                "source_chunk_id": chunk["id"]
                            })
                    
                    if facts and entities:
                        logger.info(f"✓ LLM成功提取 {len(facts)} 个facts, {len(entities)} 个实体")
                        return facts, entities
                    else:
                        logger.warning(f"LLM返回了空的facts或entities列表")
        
      
        
        return facts, entities
    
    
    def build_proposition_nodes(self, facts) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """
        构建命题节点
        
        按实体分桶并聚类，然后按文本去重
        
        Returns:
            (命题节点列表, fact_assignments)
        """
        logger.info(f"构建命题节点: {len(facts)} 个facts")
        
        # 按实体分桶
        entity_buckets = {}
        for fact in facts:
            for entity in fact["entity_ids"]:
                if entity not in entity_buckets:
                    entity_buckets[entity] = []
                entity_buckets[entity].append(fact)
        
        propositions = []
        fact_assignments = {}
        
        # 在每个桶内拼接
        prop_dedup_map = {}  
        
        for entity, bucket_facts in entity_buckets.items():
            if not bucket_facts:
                continue
            
            
            seen_texts = set()
            unique_fact_texts = []
            for f in bucket_facts:
                if f["text"] not in seen_texts:
                    unique_fact_texts.append(f["text"])
                    seen_texts.add(f["text"])
            
            prop_text = " ".join(unique_fact_texts[:10])  # 最多10个不重复的fact
            
            # 收集所有实体
            all_entities = set()
            for f in bucket_facts:
                all_entities.update(f["entity_ids"])
            
            # 创建去重键：只用文本（不考虑实体）
            dedup_key = prop_text
            
            # 如果已存在相同文本的命题，合并实体和fact_ids
            if dedup_key in prop_dedup_map:
                prop_dedup_map[dedup_key]["entity_ids"].update(all_entities)
                prop_dedup_map[dedup_key]["fact_ids"].extend([f["id"] for f in bucket_facts])
            else:
                prop_dedup_map[dedup_key] = {
                    "text": prop_text,
                    "entity_ids": all_entities,  # 使用set
                    "fact_ids": [f["id"] for f in bucket_facts]
                }
        
        # 转换为列表并分配ID
        for i, (text, prop_data) in enumerate(prop_dedup_map.items()):
            prop_id = f"prop_{i}"
            propositions.append({
                "id": prop_id,
                "text": prop_data["text"],
                "entity_ids": list(prop_data["entity_ids"]),  # 转回list
                "embedding": self._encode_text(prop_data["text"])
            })
            fact_assignments[prop_id] = list(set(prop_data["fact_ids"]))  # 去重fact_ids
        
        logger.info(f"创建了 {len(propositions)} 个命题节点（按文本去重后）")
        return propositions, fact_assignments
    
    def build_concept_nodes_from_props(self, props: List[Dict]) -> List[Dict]:
        """
        基于命题的实体集合聚类生成概念节点
        
        策略：
        1. 收集所有命题中的实体并去重
        2. 对实体进行编码和聚类
        3. 将命题分配到最相关的实体簇
        4. 使用LLM为每个概念生成描述性文本
        
        Args:
            props: 命题节点列表
            
        Returns:
            概念节点列表
        """
        if not props:
            return []
        
        logger.info(f"构建概念节点: {len(props)} 个命题")
        
        # 1. 收集所有实体并去重
        all_entities = set()
        for prop in props:
            all_entities.update(prop.get("entity_ids", []))
        
        all_entities = list(all_entities)
        logger.info(f"收集到 {len(all_entities)} 个唯一实体")
        
        if not all_entities:
            return []
        
        # 2. 对实体进行编码
        entity_embeddings = []
        valid_entities = []
        for entity in all_entities:
            try:
                emb = self._encode_text(entity)
                # 转换为numpy数组并确保是1维的
                emb = np.asarray(emb).flatten()
                
                # 检查维度
                if emb.shape[0] == 1024:
                    entity_embeddings.append(emb)
                    valid_entities.append(entity)
                else:
                    logger.warning(f"实体 '{entity}' 的embedding维度错误: {emb.shape}, 期望(1024,)")
            except Exception as e:
                logger.warning(f"实体 '{entity}' 编码失败: {e}")
        
        if not entity_embeddings:
            logger.warning("没有有效的实体embeddings，跳过概念节点构建")
            return []
        
        all_entities = valid_entities
        
        entity_embeddings = np.vstack(entity_embeddings)
        
        # 3. 对实体进行聚类
        n_clusters = min(self.concept_num_clusters, len(all_entities))
        if n_clusters < 2:
            n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        entity_cluster_labels = kmeans.fit_predict(entity_embeddings)
        
        # 4. 为每个实体簇创建概念节点
        concepts = []
        for i in range(n_clusters):
            # 获取该簇的实体
            cluster_entity_indices = np.where(entity_cluster_labels == i)[0]
            cluster_entities = [all_entities[j] for j in cluster_entity_indices]
            
            # 使用LLM生成概念文本（不需要命题信息）
            concept_text = self._generate_concept_text_from_entities(cluster_entities)
            
            concepts.append({
                "id": f"concept_{i}",
                "text": concept_text,
                "entity_ids": cluster_entities,
                "embedding": kmeans.cluster_centers_[i]
            })
        
        logger.info(f"创建了 {len(concepts)} 个概念节点")
        return concepts
    
    def _generate_concept_text_from_entities(self, cluster_entities: List[str]) -> str:
        """
        使用LLM从实体列表生成概念文本
        
        Args:
            cluster_entities: 该概念簇的实体列表
            
        Returns:
            概念描述文本
        """
        if not cluster_entities:
            return "Concept"
        
        # 构造prompt
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
                    "options": {
                        "num_predict": 10,
                        "temperature": 0.3
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                concept_text = response.json().get("response", "").strip()
                if concept_text and len(concept_text) > 0:
                    return concept_text
        
        except Exception as e:
            logger.warning(f"LLM概念生成失败: {e}")
        
        # 降级：使用实体列表作为概念描述
        return f"Concept: {', '.join(cluster_entities[:3])}"
     
    def group_chunks_for_summary(self, chunks: List[Dict], concept_edges: List[Tuple]) -> List[List[Dict]]:
        """
        基于语义相似性和概念连接关系对chunks进行聚类分组
        
        Args:
            chunks: chunk列表
            concept_edges: HAS_CONCEPT边列表
            
        Returns:
            chunk组列表
        """
        if not chunks:
            return []
        
        if len(chunks) == 1:
            return [chunks]
        
        # 1. 构建chunk之间的相似度矩阵
        chunk_ids = [c["id"] for c in chunks]
        n = len(chunks)
        
        # 提取或生成embeddings
        embeddings = []
        for chunk in chunks:
            if "embedding" in chunk:
                embeddings.append(chunk["embedding"])
            else:
                # 如果没有embedding，现场生成
                emb = self._encode_text(chunk["text"])
                embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # 计算语义相似度矩阵
        similarity_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarity_matrix / (norms[:, None] * norms[None, :] + 1e-10)
        
        # 2. 转换为距离矩阵用于聚类
        # 先将相似度矩阵裁剪到[0, 1]范围，避免负距离
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # 3. 使用层次聚类
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # 转换为压缩距离矩阵
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # 层次聚类
        linkage_matrix = linkage(condensed_dist, method='average')
        
        # 动态确定聚类数量（基于距离阈值）
        # 距离阈值：相似度低于0.5时分开（即距离大于0.5）
        distance_threshold = 0.5
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        
        # 4. 根据聚类标签分组
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(chunks[i])
        
        result_groups = list(cluster_groups.values())
        
        logger.info(f"聚类完成: {len(chunks)} 个chunks分为 {len(result_groups)} 组")
        
        return result_groups
    
    def generate_summary_node(self, chunks_group: List[Dict]) -> Dict:
        """
        使用LLM生成摘要节点
        
        Args:
            chunks_group: chunk组
            
        Returns:
            摘要节点
        """
        # 收集文本
        texts = [c["text"] for c in chunks_group[:5]]  # 最多5个chunk
        combined_text = " ".join(texts)
        
        # 使用LLM生成摘要
        summary_text = self._call_llm_for_summary(combined_text)
        
        # 收集实体
        all_entities = set()
        for chunk in chunks_group:
            if "entity_ids" in chunk:
                all_entities.update(chunk["entity_ids"])
        
        return {
            "id": f"summary_{len(chunks_group)}",
            "text": summary_text,
            "entity_ids": list(all_entities),
            "embedding": self._encode_text(summary_text)
        }
    
    def _call_llm_for_summary(self, text: str) -> str:
        """调用LLM生成摘要"""
        prompt = f"""# Role
You are a professional text summarizer who creates concise, informative summaries.

# Task
Summarize the following text in 2-3 sentences, capturing the main ideas and key information.

# Instructions
- Keep the summary between 2-3 sentences
- Focus on the most important information
- Use clear, concise language
- Maintain factual accuracy

# Example
Input: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some of France's leading artists and intellectuals for its design."

Output: "The Eiffel Tower is a wrought-iron lattice tower in Paris, named after engineer Gustave Eiffel. Built between 1887 and 1889, it initially faced criticism from French artists and intellectuals for its design."

# Your Task
Text:
{text[:500]}

Summary:"""
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM摘要生成失败: {e}")
        
        return text[:200]  # 降级：返回前200字符
    
    def add_similarity_edges(self, graph: nx.DiGraph, node_type: str, top_k: int = 5, sim_threshold: float = 0.7):
        """
        为指定类型的节点添加SIMILAR_TO边
        
        Args:
            graph: 图对象
            node_type: 节点类型
            top_k: 每个节点的top-k邻居
            sim_threshold: 相似度阈值
        """
        # 获取指定类型的所有节点
        nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == node_type]
        
        if len(nodes) < 2:
            return
        
        # 提取embedding
        embeddings = []
        for node_id in nodes:
            emb = graph.nodes[node_id].get("embedding")
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(1024))
        
        embeddings = np.array(embeddings)
        
        # 计算相似度矩阵
        similarity_matrix = np.dot(embeddings, embeddings.T)
        norms = np.linalg.norm(embeddings, axis=1)
        similarity_matrix = similarity_matrix / (norms[:, None] * norms[None, :] + 1e-10)
        
        # 为每个节点添加top-k相似边
        for i, node_id in enumerate(nodes):
            similarities = similarity_matrix[i]
            similarities[i] = -1  # 排除自己
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            for j in top_indices:
                if similarities[j] >= sim_threshold:
                    graph.add_edge(node_id, nodes[j],
                                 edge_type="SIMILAR_TO",
                                 weight=float(similarities[j]))
    
    def build_heterogeneous_graph(self, chunks_coref: List[Dict]) -> nx.DiGraph:
        """
        构建完整异构图
        
        Args:
            chunks_coref: 共指消解后的chunk列表
            
        Returns:
            异构图
        """
        logger.info(f"开始构建异构图: {len(chunks_coref)} 个chunks")
        
        graph = nx.Graph()
        
        # 1) 添加Chunk节点并提取facts和实体
        all_facts = []
        for chunk in chunks_coref:
           
            facts, entity_ids = self.extract_facts_and_entities_from_chunk(chunk)
            chunk["entity_ids"] = entity_ids
            chunk["facts"] = facts
            all_facts.extend(facts)
            
            # 确保chunk有embedding
            if "embedding" not in chunk or chunk["embedding"] is None:
                logger.debug(f"为chunk {chunk['id']} 生成embedding")
                chunk["embedding"] = self._encode_text(chunk.get("text", ""))
            
            graph.add_node(chunk["id"], node_type="chunk", **chunk)
        
        # 2) 构建Proposition节点
        # 优先使用预构建的propositions
        all_propositions = []
        
        # 从chunks中收集预构建的propositions
        seen_prop_ids = set()
        for chunk in chunks_coref:
            # 检查chunk所属文档是否有预构建的propositions
            if "propositions" in chunk:
                for prop in chunk["propositions"]:
                    if prop["id"] not in seen_prop_ids:
                        all_propositions.append(prop)
                        seen_prop_ids.add(prop["id"])
        
        if all_propositions:
            logger.info(f"使用预构建的propositions: {len(all_propositions)} 个")
            props = all_propositions
 
            fact_assignments = {p["id"]: p.get("fact_ids", []) for p in props}
        else:
            # 使用从chunks中提取的facts构建propositions
            logger.info(f"使用提取的facts构建propositions: {len(all_facts)} 个facts")
            props, fact_assignments = self.build_proposition_nodes(all_facts)
        
        for prop in props:
            # 如果没有embedding，现场生成
            if "embedding" not in prop:
                prop["embedding"] = self._encode_text(prop["text"])
            graph.add_node(prop["id"], node_type="proposition", **prop)
        
        # 3) 建立C->P边
        fact_to_chunk = {f["id"]: f["source_chunk_id"] for f in all_facts}
        for prop_id, fact_ids in fact_assignments.items():
            for fact_id in fact_ids:
                chunk_id = fact_to_chunk.get(fact_id)
                if chunk_id:
                    graph.add_edge(chunk_id, prop_id, edge_type="HAS_PROP", weight=1.0)
        
        # 4) 构建Concept节点
        concepts = self.build_concept_nodes_from_props(props)
        for concept in concepts:
            graph.add_node(concept["id"], node_type="concept", **concept)
        
        # 5) 建立C->Z边（基于实体重叠度）
        for chunk_id in [c["id"] for c in chunks_coref]:
            chunk_entities = set(graph.nodes[chunk_id].get("entity_ids", []))
            
            for concept in concepts:
                concept_entities = set(concept["entity_ids"])
                
                if chunk_entities and concept_entities:
                    overlap = len(chunk_entities & concept_entities) / len(chunk_entities | concept_entities)
                    
                    if overlap >= 0.1:  # 阈值
                        graph.add_edge(chunk_id, concept["id"], 
                                     edge_type="HAS_CONCEPT", weight=overlap)
        
        # 6) 构建Summary节点
        concept_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                        if d.get("edge_type") == "HAS_CONCEPT"]
        chunk_groups = self.group_chunks_for_summary(chunks_coref, concept_edges)
        
        for i, group in enumerate(chunk_groups):
            if len(group) > 1:  # 至少2个chunk才生成摘要
                summary = self.generate_summary_node(group)
                summary["id"] = f"summary_{i}"
                graph.add_node(summary["id"], node_type="summary", **summary)
                
                # 建立C->S边
                for chunk in group:
                    graph.add_edge(chunk["id"], summary["id"],
                                 edge_type="SUMMARIZED_BY", weight=1.0)
        
        # 7) 添加SIMILAR_TO边
        self.add_similarity_edges(graph, "chunk", top_k=5, sim_threshold=0.7)
        self.add_similarity_edges(graph, "proposition", top_k=10, sim_threshold=0.7)
        self.add_similarity_edges(graph, "concept", top_k=5, sim_threshold=0.75)
        self.add_similarity_edges(graph, "summary", top_k=3, sim_threshold=0.7)
        
        logger.info(f"异构图构建完成: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        
        return graph

