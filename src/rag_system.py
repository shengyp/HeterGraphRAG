"""
RAG系统主类 - 简化版

流程：
1. 离线构建全局图 G_global
2. 粗检得到chunk集合 C_q
3. 从C_q做2-hop扩展得到子图 G_q
4. 在G_q上做LLM语义锚点选择
5. 在G_q上做结构中心度排序
6. 取top节点及1-hop邻居送入LLM
"""

import logging
import pickle
from typing import Dict, List
import networkx as nx

from src.bgem3_retriever import BGEM3Retriever
from src.graph_builder import GraphBuilder
from src.intent_representation import IntentRepresentation
from src.semantic_anchor_selector import SemanticAnchorSelector
from src.structural_centrality_ranker import StructuralCentralityRanker
from src.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG系统 - 基于异构图的多跳问答系统"""
    
    def __init__(self, config: Dict):
        """初始化RAG系统"""
        self.config = config
        self.global_graph = None  # 全局图 G_global
        
        # 初始化模块
        logger.info("初始化RAG系统...")
        
        self.bgem3_retriever = BGEM3Retriever(config.get("coarse_retrieval", {}))
        self.graph_builder = GraphBuilder(config.get("graph_construction", {}))
        self.intent_repr = IntentRepresentation(config.get("intent_representation", {}))
        self.anchor_selector = SemanticAnchorSelector(config.get("semantic_anchor", {}))
        self.centrality_ranker = StructuralCentralityRanker(config.get("structural_centrality", {}))
        self.answer_generator = AnswerGenerator(config.get("answer_generation", {}))
        
        logger.info("RAG系统初始化完成")
    
    def build_global_graph(self, documents: List[Dict]):
        """
        离线构建全局图 G_global
        
        Args:
            documents: 预处理后的文档列表（包含chunks, propositions, concepts, summaries）
        """
        logger.info(f"开始构建全局图: {len(documents)} 个文档")
        
        # 使用graph_builder构建异构图
        self.global_graph = self.graph_builder.build_heterogeneous_graph(documents)
        
        logger.info(f"全局图构建完成: {self.global_graph.number_of_nodes()} 节点, {self.global_graph.number_of_edges()} 边")
    
    def save_global_graph(self, path: str):
        """保存全局图"""
        with open(path, 'wb') as f:
            pickle.dump(self.global_graph, f)
        logger.info(f"全局图已保存: {path}")
    
    def load_global_graph(self, path: str):
        """加载全局图"""
        with open(path, 'rb') as f:
            self.global_graph = pickle.load(f)
        logger.info(f"全局图已加载: {self.global_graph.number_of_nodes()} 节点")
    
    def index_documents(self, documents: List[Dict]):
        """
        为文档建立索引（用于粗检）
        
        Args:
            documents: 文档列表
        """
        logger.info(f"开始索引文档: {len(documents)} 个文档")
        self.bgem3_retriever.build_doc_index(documents)
        logger.info("文档索引完成")
    
    def query(self, query: str) -> Dict:
        """
        执行完整的RAG查询流程
        
        流程：
        1. 粗检：BGE-M3检索得到chunk集合 C_q
        2. 2-hop扩展：从C_q在G_global中扩展得到子图 G_q
        3. 意图表征：构建意图向量
        4. 语义锚点选择：在G_q上选择锚点
        5. 结构中心度排序：在G_q上排序锚点
        6. 答案生成：取top节点及1-hop邻居送入LLM
        
        Args:
            query: 查询文本
            
        Returns:
            {
                "answer": "答案文本",
                "supporting_nodes": {...},
                "metadata": {...}
            }
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始RAG查询: {query}")
        logger.info(f"{'='*60}\n")
        
        # 1. 粗检：BGE-M3检索
        logger.info("步骤1: BGE-M3粗检")
        candidate_docs = self.bgem3_retriever.hybrid_doc_retrieval(query)
        
        # 提取chunk集合 C_q
        chunk_ids = []
        candidate_chunks = []
        for doc in candidate_docs:
            if "chunks" in doc and doc["chunks"]:
                for chunk in doc["chunks"]:
                    chunk_ids.append(chunk["id"])
                    candidate_chunks.append(chunk)
        
        logger.info(f"粗检完成: {len(candidate_docs)} 个文档, {len(chunk_ids)} 个chunks")
        
        # 2. 2-hop扩展：从C_q得到子图 G_q
        logger.info("\n步骤2: 2-hop子图扩展")
        G_q = self._extract_subgraph(chunk_ids, hops=2)
        logger.info(f"子图提取完成: {G_q.number_of_nodes()} 节点, {G_q.number_of_edges()} 边")
        
        # 3. 意图表征
        logger.info("\n步骤3: 意图表征")
        h_intent, intent_type = self.intent_repr.build_intent_vector(query, candidate_chunks)
        logger.info(f"意图表征完成: intent_type={intent_type}")
        
        # 4. 语义锚点选择（在子图G_q上）
        logger.info("\n步骤4: 语义锚点选择")
        s_sem = self.anchor_selector.select_semantic_anchors(G_q, query, h_intent, intent_type)
        
        # 获取超过阈值的锚点
        tau_sem = self.config.get("semantic_anchor", {}).get("tau_sem", 0.25)
        anchor_candidates = [nid for nid, score in s_sem.items() if score >= tau_sem]
        
        # Top-K保底
        if not anchor_candidates:
            top_k = self.config.get("semantic_anchor", {}).get("top_k_fallback", 10)
            sorted_nodes = sorted(s_sem.items(), key=lambda x: x[1], reverse=True)
            anchor_candidates = [nid for nid, score in sorted_nodes[:top_k]]
        
        logger.info(f"语义锚点选择完成: {len(anchor_candidates)} 个候选锚点")
        
        # 5. 结构中心度排序（在全局图G_global上）
        logger.info("\n步骤5: 结构中心度排序（在全局图上）")
        top_anchors = self.centrality_ranker.rank_anchor_nodes(self.global_graph, anchor_candidates, s_sem)
        logger.info(f"锚点排序完成: {len(top_anchors)} 个top锚点")
        
        # 6. 收集节点及1-hop邻居（在全局图上）
        logger.info("\n步骤6: 收集上下文节点（在全局图上）")
        context_nodes = set(top_anchors)
        
        # 添加1-hop邻居（从全局图）
        for node_id in top_anchors:
            if self.global_graph.has_node(node_id):
                neighbors = list(self.global_graph.neighbors(node_id))
                context_nodes.update(neighbors)
        
        logger.info(f"上下文节点: {len(context_nodes)} 个（包含1-hop邻居）")
        
        # 构建ranked_nodes格式（用于答案生成）
        ranked_nodes = {}
        for node_id in context_nodes:
            if self.global_graph.has_node(node_id):
                node_data = self.global_graph.nodes[node_id]
                node_type = node_data.get("node_type", "unknown")
                
                if node_type not in ranked_nodes:
                    ranked_nodes[node_type] = []
                
                ranked_nodes[node_type].append({
                    "id": node_id,
                    "text": node_data.get("text", ""),
                    "score": s_sem.get(node_id, 0.0)
                })
        
        # 7. 答案生成
        logger.info("\n步骤7: 答案生成")
        result = self.answer_generator.generate_answer(query, ranked_nodes, self.global_graph)
        
        # 添加元数据
        result["metadata"] = {
            "intent_type": intent_type,
            "num_candidate_chunks": len(chunk_ids),
            "num_subgraph_nodes": G_q.number_of_nodes(),
            "num_subgraph_edges": G_q.number_of_edges(),
            "num_anchors": len(top_anchors),
            "num_context_nodes": len(context_nodes)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RAG查询完成")
        logger.info(f"{'='*60}\n")
        
        return result
    
    def _extract_subgraph(self, chunk_ids: List[str], hops: int = 2) -> nx.Graph:
        """
        从chunk集合在全局图中做N-hop扩展，得到查询子图
        
        2-hop原因：
        - 第1跳：chunk -> proposition/concept/summary
        - 第2跳：proposition/concept/summary -> 其他相似节点
        
        Args:
            chunk_ids: chunk ID列表
            hops: 扩展跳数
            
        Returns:
            查询子图 G_q
        """
        logger.info(f"从 {len(chunk_ids)} 个chunks开始{hops}-hop扩展")
        
        subgraph_nodes = set()
        
        # 添加起始chunk节点
        for chunk_id in chunk_ids:
            if self.global_graph.has_node(chunk_id):
                subgraph_nodes.add(chunk_id)
        
        # 多跳扩展
        current_nodes = set(chunk_ids)
        for hop in range(hops):
            next_nodes = set()
            
            for node_id in current_nodes:
                if not self.global_graph.has_node(node_id):
                    continue
                
                # 获取邻居
                neighbors = list(self.global_graph.neighbors(node_id))
                next_nodes.update(neighbors)
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            logger.info(f"  第{hop+1}跳: 新增 {len(next_nodes)} 个节点")
        
        # 提取子图
        G_q = self.global_graph.subgraph(subgraph_nodes).copy()
        
        return G_q
