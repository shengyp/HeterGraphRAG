"""
新意图表征模块

构造统一查询意图向量: h_intent = [h_t; h_q; h_ctx]
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
import requests

logger = logging.getLogger(__name__)


class IntentRepresentation:
    """意图表征器 - 构造统一查询意图向量"""
    
    def __init__(self, config: Dict):
        """
        初始化意图表征器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        self.llm_model = config.get("llm_model", "qwen2.5:7b")
        
        # 意图类型模板
        self.intent_templates = {
            "FACTUAL": "This is a factual question asking for specific information, events, or attributes.",
            "DEFINITION": "This is a definition question asking for the meaning or explanation of a concept.",
            "CAUSAL": "This is a causal question asking about cause-and-effect relationships or reasons.",
            "COMPARISON": "This is a comparison question asking to compare or contrast multiple objects."
        }
        
        logger.info("新意图表征器初始化完成")
    
    def _encode_text(self, text: str) -> np.ndarray:
        """使用BGE-M3编码文本"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": "bge-m3",
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Embeddings API失败: status={response.status_code}, response={response.text}")
                return np.zeros(1024)
            
            result = response.json()
            embedding = result.get("embedding")
            
            if embedding is None:
                logger.error(f"Embeddings API返回格式错误: {result}")
                return np.zeros(1024)
            
            emb_array = np.array(embedding)
            
            # 检查维度
            if len(emb_array) != 1024:
                logger.error(f"Embedding维度错误: {len(emb_array)}, 预期1024")
                return np.zeros(1024)
            
            # 检查是否是零向量
            if np.linalg.norm(emb_array) < 1e-6:
                logger.warning(f"生成的embedding是零向量")
            
            return emb_array
            
        except Exception as e:
            logger.error(f"编码失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(1024)
    
    def encode_query_and_context(self, query: str, coarse_chunks: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        对查询和精选候选chunk上下文进行编码
        
        注意：这里的coarse_chunks不是第一次粗检索得到的大量文档，
        而是经过初步筛选后的精选候选chunks，数量相对较少且更相关
        
        Args:
            query: 查询文本
            coarse_chunks: 精选候选chunk列表（已经过初步筛选）
            
        Returns:
            (h_q, h_ctx) - 查询向量和上下文向量
        """
        # 编码查询
        h_q = self._encode_text(query)
        
        # 编码chunks并平均池化
        chunk_embeddings = []
        for chunk in coarse_chunks[:20]:  # 限制数量
            if "embedding" in chunk:
                chunk_embeddings.append(chunk["embedding"])
            else:
                chunk_embeddings.append(self._encode_text(chunk["text"]))
        
        if chunk_embeddings:
            h_ctx = np.mean(chunk_embeddings, axis=0)
        else:
            h_ctx = np.zeros_like(h_q)
        
        return h_q, h_ctx
    
    def classify_intent_type(self, query: str, coarse_chunks: List[Dict]) -> str:
        """
        基于查询和精选候选chunks预测查询意图类型
        
        Args:
            query: 查询文本
            coarse_chunks: 精选候选chunk列表（已经过初步筛选）
            
        Returns:
            意图类型字符串
        """
        prompt = f"""# Role
You are a question analysis expert who classifies questions by their intent type.

# Task
Classify the intent type of the given question into one of four categories.

# Intent Types
- FACTUAL: asking for specific facts, events, or attributes
- DEFINITION: asking for definitions or explanations
- CAUSAL: asking about causes, reasons, or effects
- COMPARISON: asking to compare or contrast

# Examples
Question: "Who invented the telephone?"
Answer: FACTUAL

Question: "What is quantum mechanics?"
Answer: DEFINITION

Question: "Why did World War II start?"
Answer: CAUSAL

Question: "What is the difference between Python and Java?"
Answer: COMPARISON

# Your Task
Question: {query}

Answer (one word only):"""
        
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
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip().upper()
                
                for intent_type in self.intent_templates.keys():
                    if intent_type in result:
                        return intent_type
        
        except Exception as e:
            logger.warning(f"意图分类失败: {e}")
        
        # 默认返回FACTUAL
        return "FACTUAL"
    
    def encode_intent_type(self, intent_type: str) -> np.ndarray:
        """
        将意图类型对应的描述性模板编码为向量
        
        Args:
            intent_type: 意图类型
            
        Returns:
            意图类型向量 h_t
        """
        template = self.intent_templates.get(intent_type, self.intent_templates["FACTUAL"])
        return self._encode_text(template)
    
    def build_intent_vector(self, query: str, coarse_chunks: List[Dict]) -> Tuple[np.ndarray, str]:
        """
        构造统一查询意图向量: h_intent = [h_t; h_q; h_ctx]
        
        Args:
            query: 查询文本
            coarse_chunks: 精选候选chunk列表（已经过初步筛选，不是第一次粗检索的大量文档）
            
        Returns:
            (h_intent, intent_type) - 统一意图向量和意图类型
        """
        logger.info(f"构造意图向量: query='{query[:50]}...'")
        
        # 1) 编码查询和上下文
        h_q, h_ctx = self.encode_query_and_context(query, coarse_chunks)
        
        # 2) 分类意图类型
        intent_type = self.classify_intent_type(query, coarse_chunks)
        
        # 3) 编码意图类型
        h_t = self.encode_intent_type(intent_type)
        
        # 4) 拼接为统一意图向量
        h_intent = np.concatenate([h_t, h_q, h_ctx])
        
        logger.info(f"意图向量构造完成: intent_type={intent_type}, dim={h_intent.shape[0]}")
        
        return h_intent, intent_type

