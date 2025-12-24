"""
BGE-M3混合粗检模块

使用BGE-M3的稠密与稀疏混合表示进行文档级检索
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
import requests

logger = logging.getLogger(__name__)


class BGEM3Retriever:
    """BGE-M3混合检索器，支持稠密向量和稀疏表示"""
    
    def __init__(self, config: Dict):
        """
        初始化BGE-M3检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.top_k_dense = config.get("top_k_dense", 200)
        self.top_k_final = config.get("top_k_final", 20)
        self.lambda_dense = config.get("lambda_dense", 0.5)
        self.lambda_sparse = config.get("lambda_sparse", 0.5)
        self.ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        
        # 索引存储
        self.documents = []
        self.doc_dense_embeddings = []
        self.doc_sparse_embeddings = []
        
        logger.info("BGE-M3检索器初始化完成")
    
    def encode_dense(self, text: str) -> np.ndarray:
        """
        使用BGE-M3生成稠密向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            稠密向量 h ∈ R^d
        """
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": "bge-m3",
                    "prompt": text
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding)
        except Exception as e:
            logger.error(f"稠密编码失败: {e}")
            return np.zeros(1024)
    
    def encode_sparse(self, text: str) -> Dict[int, float]:
        """
        使用BGE-M3生成稀疏表示
        
        Args:
            text: 输入文本
            
        Returns:
            稀疏表示 {token_id: weight}
        """
        # 注意：实际的BGE-M3稀疏编码需要特殊的API
        # 这里使用简化的TF-IDF风格实现作为占位符
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # 转换为稀疏表示（使用hash作为token_id）
        sparse = {}
        for word, count in word_counts.items():
            token_id = hash(word) % 100000
            sparse[token_id] = float(count)
        
        return sparse
    
    def dot_sparse(self, sq: Dict[int, float], sd: Dict[int, float]) -> float:
        """
        稀疏向量点积
        
        Args:
            sq: 查询的稀疏表示
            sd: 文档的稀疏表示
            
        Returns:
            点积结果
        """
        acc = 0.0
        for k, v in sq.items():
            if k in sd:
                acc += v * sd[k]
        return acc
    
    def save_index(self, index_path: str) -> None:
        """
        保存索引到文件
        
        Args:
            index_path: 索引文件路径（不含扩展名）
        """
        import pickle
        
        index_data = {
            "documents": self.documents,
            "doc_dense_embeddings": self.doc_dense_embeddings,
            "doc_sparse_embeddings": self.doc_sparse_embeddings
        }
        
        with open(f"{index_path}.pkl", 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"索引已保存到: {index_path}.pkl")
    
    def load_index(self, index_path: str) -> bool:
        """
        从文件加载索引
        
        Args:
            index_path: 索引文件路径（不含扩展名）
            
        Returns:
            是否成功加载
        """
        import pickle
        import os
        
        if not os.path.exists(f"{index_path}.pkl"):
            logger.warning(f"索引文件不存在: {index_path}.pkl")
            return False
        
        try:
            with open(f"{index_path}.pkl", 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data["documents"]
            self.doc_dense_embeddings = index_data["doc_dense_embeddings"]
            self.doc_sparse_embeddings = index_data["doc_sparse_embeddings"]
            
            logger.info(f"索引已加载: {len(self.documents)} 个文档")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False
    
    def build_doc_index(self, docs: List[Dict]) -> None:
        """
        为文档建立稠密和稀疏索引
        
        Args:
            docs: 文档列表
        """
        logger.info(f"开始建立文档索引: {len(docs)} 个文档")
        
        self.documents = docs
        self.doc_dense_embeddings = []
        self.doc_sparse_embeddings = []
        
        for i, doc in enumerate(docs):
            if i % 10 == 0:
                logger.info(f"索引进度: {i}/{len(docs)}")
            
            text = doc.get("text", "")
            
            # 生成稠密表示
            dense = self.encode_dense(text)
            self.doc_dense_embeddings.append(dense)
            
            # 生成稀疏表示
            sparse = self.encode_sparse(text)
            self.doc_sparse_embeddings.append(sparse)
        
        self.doc_dense_embeddings = np.array(self.doc_dense_embeddings)
        logger.info("文档索引建立完成")
    
    def hybrid_doc_retrieval(self, query: str) -> List[Dict]:
        """
        混合文档检索
        
        Args:
            query: 查询文本
            
        Returns:
            候选文档列表 D_top
        """
        logger.info(f"执行混合文档检索: query='{query[:50]}...'")
        
        # 1) 编码查询
        h_q = self.encode_dense(query)
        s_q = self.encode_sparse(query)
        
        # 2) 在稠密索引上检索 top-K_dense
        # 计算余弦相似度
        query_norm = np.linalg.norm(h_q)
        doc_norms = np.linalg.norm(self.doc_dense_embeddings, axis=1)
        
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)
        query_norm = query_norm if query_norm > 0 else 1e-10
        
        dense_similarities = np.dot(self.doc_dense_embeddings, h_q) / (doc_norms * query_norm)
        
        # 获取 top-K_dense 候选
        top_dense_indices = np.argsort(dense_similarities)[::-1][:self.top_k_dense]
        
        # 3) 对候选文档计算混合得分
        scores = []
        for idx in top_dense_indices:
            dense_sim = dense_similarities[idx]
            sparse_sim = self.dot_sparse(s_q, self.doc_sparse_embeddings[idx])
            
            # 混合得分
            score_doc = self.lambda_dense * dense_sim + self.lambda_sparse * sparse_sim
            scores.append((idx, score_doc))
        
        # 4) 排序并返回 top-K_final
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = []
        
        for idx, score in scores[:self.top_k_final]:
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            top_docs.append(doc)
        
        logger.info(f"混合检索完成: 返回 {len(top_docs)} 个文档")
        return top_docs

