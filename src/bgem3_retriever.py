"""
BGE-M3混合粗检模块

目标：
1) 使用 BGE-M3 的标准稀疏向量（lexical_weights: token_id -> weight
2) 稠密检索（dense）+ 稀疏检索（sparse）混合
3) sparse 用倒排索引（inverted index）实现，查询时累加 w_q * w_d

"""

import logging
import pickle
import os

import numpy as np

logger = logging.getLogger(__name__)


class BGEM3Retriever:
    def __init__(self, config):
 
        self.config = config or {}

        self.model_name = self.config.get("model_name", "BAAI/bge-m3")
        self.use_fp16 = bool(self.config.get("use_fp16", True))
        self.batch_size = int(self.config.get("batch_size", 16))

        self.top_k_dense = int(self.config.get("top_k_dense", 200))
        self.top_k_sparse = int(self.config.get("top_k_sparse", 200))
        self.top_k_final = int(self.config.get("top_k_final", 20))

        self.fusion_method = self.config.get("fusion_method", "rrf")  # "rrf" / "linear"
        self.rrf_k = int(self.config.get("rrf_k", 60))
        self.lambda_dense = float(self.config.get("lambda_dense", 0.5))
        self.lambda_sparse = float(self.config.get("lambda_sparse", 0.5))

        # 文档与索引
        self.documents = []
        self.doc_dense_embeddings = None          # np.ndarray (N, D)
        #
        self.doc_norms = None                     # np.ndarray (N,)
        #存放每篇文档的稀疏权重字典：token_id -> weight
        self.doc_sparse_weights = []     
        #倒排索引：token_id -> [(doc_idx, weight_in_doc), ...]     
        self.inverted_index = {}                  

        # 初始化 BGE-M3 模型（FlagEmbedding）
        self.model = self._load_bge_m3_model()

        logger.info("BGE-M3检索器初始化完成")

    # -------------------------
    # Model
    # -------------------------
    def _load_bge_m3_model(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            raise RuntimeError(
                "缺少依赖 FlagEmbedding。请先安装：pip install -U FlagEmbedding"
            ) from e

        
        return BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def _encode_batch(self, texts, return_dense=True, return_sparse=True):
      
        out = self.model.encode(
            texts,
            return_dense=return_dense,
            return_sparse=return_sparse,
        )
        dense_vecs = out.get("dense_vecs") if return_dense else None
        lexical_weights = out.get("lexical_weights") if return_sparse else None
        return dense_vecs, lexical_weights #token_id -> weight

    # -------------------------
    # Index build
    # -------------------------
    def build_doc_index(self, docs):
        """
        建立索引（dense + sparse）

        """
        if not docs:
            self.documents = []
            self.doc_dense_embeddings = np.zeros((0, 1024))
            self.doc_norms = np.zeros((0,))
            self.doc_sparse_weights = []
            self.inverted_index = {}
            logger.warning("传入docs为空，索引为空")
            return

        logger.info(f"开始建立文档索引: {len(docs)} 个文档")

        self.documents = docs
        self.doc_sparse_weights = []
        dense_list = []

        self.inverted_index = {}

        texts = [(d.get("text") or "") for d in docs]

      
        n = len(texts)
        bs = max(1, self.batch_size)
        #以 batch size 步长遍历所有文档。
        for start in range(0, n, bs):
            end = min(n, start + bs)
            #取当前批次文本
            batch_texts = texts[start:end]

            #dense_vecs：当前批次每篇文档的 dense 向量
            #lexical_list：当前批次每篇文档的 sparse 字典
            dense_vecs, lexical_list = self._encode_batch(
                batch_texts, return_dense=True, return_sparse=True
            )

            # 把 dense 转成 numpy array
            dense_vecs = np.asarray(dense_vecs)
            dense_list.append(dense_vecs)

            #遍历文档的洗漱列表
            for local_i, sparse in enumerate(lexical_list):
                doc_idx = start + local_i
                if sparse is None:
                    sparse = {}
                self.doc_sparse_weights.append(sparse)

                # 构建倒排inverted_index[101] = [(10,0.8), (11,0.6)]
                for tid, wd in sparse.items():
                    self.inverted_index.setdefault(tid, []).append((doc_idx, float(wd)))

            if start % (bs * 10) == 0:
                logger.info(f"索引进度: {start}/{n}")
        #把所有 batch 的 dense 拼成 (N, D) 矩阵
        self.doc_dense_embeddings = np.vstack(dense_list).astype(np.float32)

        #对每一行（每篇文档向量）求 L2 范数
        norms = np.linalg.norm(self.doc_dense_embeddings, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)
        self.doc_norms = norms

        logger.info("文档索引建立完成")


    def save_index(self, index_path):
        """
        保存索引到文件（pickle）

        会保存：
        - documents
        - doc_dense_embeddings
        - doc_norms
        - doc_sparse_weights
        - inverted_index
        """
        index_data = {
            "documents": self.documents,
            "doc_dense_embeddings": self.doc_dense_embeddings,
            "doc_norms": self.doc_norms,
            "doc_sparse_weights": self.doc_sparse_weights,
            "inverted_index": self.inverted_index,
        }
        with open(f"{index_path}.pkl", "wb") as f:
            pickle.dump(index_data, f)
        logger.info(f"索引已保存到: {index_path}.pkl")

    def load_index(self, index_path):
        """
        加载索引（pickle）
        """
        pkl = f"{index_path}.pkl"
        if not os.path.exists(pkl):
            logger.warning(f"索引文件不存在: {pkl}")
            return False

        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.doc_dense_embeddings = data["doc_dense_embeddings"]
            self.doc_norms = data["doc_norms"]
            self.doc_sparse_weights = data["doc_sparse_weights"]
            self.inverted_index = data["inverted_index"]

            logger.info(f"索引已加载: {len(self.documents)} 个文档")
            return True
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return False

    # -------------------------
    # Dense retrieval
    # -------------------------
    def _dense_retrieve(self, h_q):
        """
        dense 召回：余弦相似度 top_k_dense
        返回：list[(doc_idx, score)]
        """
        if self.doc_dense_embeddings is None or len(self.documents) == 0:
            return []
        #转成 float32 numpy 向量
        h_q = np.asarray(h_q).astype(np.float32)
        #求 query 范数，避免除 0
        q_norm = np.linalg.norm(h_q)
        q_norm = q_norm if q_norm > 0 else 1e-10
        #每个文档向量与 query 向量点积/两者范数乘积
        sims = (self.doc_dense_embeddings @ h_q) / (self.doc_norms * q_norm)
        

        k = min(self.top_k_dense, sims.shape[0])
        if k <= 0:
            return []
        #排序取最大 K 个,返回 (doc_idx, score) 列表
        top_idx = np.argsort(sims)[::-1][:k]

        return [(int(i), float(sims[i])) for i in top_idx]

    # -------------------------
    # Sparse retrieval (serious)
    # -------------------------
    def _sparse_retrieve(self, s_q): #query 的 sparse 字典 token_id -> weight
       
        if not self.inverted_index or len(self.documents) == 0:
            return []

        scores = {}  

        # 遍历 query 的 token_id
        for tid, wq in s_q.items():
            postings = self.inverted_index.get(tid)
            if not postings:
                continue

            wq = float(wq)
            #对命中文档累加得分：wq * wd
            for doc_idx, wd in postings:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + wq * float(wd)

        if not scores:
            return []

        #按得分降序，取 top_k_sparse，返回候选列表

        items = list(scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
        k = min(self.top_k_sparse, len(items))
        return [(int(doc_idx), float(score)) for doc_idx, score in items[:k]]

    # -------------------------
    # 融合
    # -------------------------
    def _fuse_scores_rrf(self, dense_list, sparse_list):
       
        rrf_k = max(1, self.rrf_k)
        fused = {}

        for rank, (doc_idx, _) in enumerate(dense_list, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)

        for rank, (doc_idx, _) in enumerate(sparse_list, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)

        items = list(fused.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def _fuse_scores_linear(self, dense_list, sparse_list):
      
        fused = {}
        for doc_idx, s in dense_list:
            fused[doc_idx] = fused.get(doc_idx, 0.0) + self.lambda_dense * float(s)
        for doc_idx, s in sparse_list:
            fused[doc_idx] = fused.get(doc_idx, 0.0) + self.lambda_sparse * float(s)

        items = list(fused.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def hybrid_doc_retrieval(self, query):
        """

        流程：
        1) 对 query 编码：dense_vec + lexical_weights（标准 sparse）
        2) dense top_k_dense 召回
        3) sparse top_k_sparse 召回（倒排索引 + lexical matching）
        4) 融合：默认 RRF（也可 linear）
        5) 返回 top_k_final 文档，并附 score

        """
        if not query:
            return []

        if self.documents is None or len(self.documents) == 0:
            logger.warning("尚未建立/加载索引，documents为空")
            return []

        logger.info(f"执行初始文档检索: query='{query[:50]}...'")

  
        dense_vecs, lexical_list = self._encode_batch([query], return_dense=True, return_sparse=True)
        
        h_q = np.asarray(dense_vecs[0])
        s_q = lexical_list[0] or {}

   
        dense_candidates = self._dense_retrieve(h_q)


        sparse_candidates = self._sparse_retrieve(s_q)


        if self.fusion_method == "linear":
            fused = self._fuse_scores_linear(dense_candidates, sparse_candidates)
        else:
            fused = self._fuse_scores_rrf(dense_candidates, sparse_candidates)

        # 5) top_k_final
        k = min(self.top_k_final, len(fused))
        top_docs = []
        for doc_idx, score in fused[:k]:
            doc = self.documents[doc_idx].copy()
            doc["score"] = float(score)
            top_docs.append(doc)

        logger.info(f"混合检索完成: 返回 {len(top_docs)} 个文档")
        return top_docs
