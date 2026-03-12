# -*- coding: utf-8 -*-
"""
BGE-M3 混合粗检模块（定稿版：只产 seeds=chunk ids，不改变职责）

目标：
1) 使用 BGE-M3 稠密检索（dense）+ 稀疏检索（sparse）混合
2) sparse 用倒排索引（inverted index）实现，查询时累加 w_q * w_d
3) 返回 top_k_final 个“文档对象”，文档里应携带 chunks（供 RAGSystem 生成 seeds）

与新版 RAGSystem 对齐的关键点：
- RAGSystem 从 returned_docs 里读取 doc["chunks"][*]["id"] 作为 seeds
- 因此检索器必须尽量保证返回的 doc 带 chunks 且 chunks 含 id/text
- 索引文本：优先 doc["text"]，若缺失则使用 chunks 拼接得到 doc_text
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BGEM3Retriever:
    def __init__(self, config: Dict[str, Any]):
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

      
        self.expected_dense_dim = self.config.get("expected_dense_dim", None)  # e.g., 1024

        # 文档与索引
        self.documents: List[Dict[str, Any]] = []
        self.doc_dense_embeddings: np.ndarray = None  # (N, D)
        self.doc_norms: np.ndarray = None  # (N,)
        self.doc_sparse_weights: List[Dict[Any, float]] = []
        self.inverted_index: Dict[Any, List[Tuple[int, float]]] = {}

        # 初始化 BGE-M3 模型（FlagEmbedding）
        self.model = self._load_bge_m3_model()
        logger.info("BGE-M3 Retriever initialized | model=%s", self.model_name)

    # -------------------------
    # Model
    # -------------------------
    def _load_bge_m3_model(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            raise RuntimeError("缺少依赖 FlagEmbedding。请先安装：pip install -U FlagEmbedding") from e
        return BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def _encode_batch(self, texts: List[str], return_dense: bool = True, return_sparse: bool = True):
        out = self.model.encode(texts, return_dense=return_dense, return_sparse=return_sparse)
        dense_vecs = out.get("dense_vecs") if return_dense else None
        lexical_weights = out.get("lexical_weights") if return_sparse else None
        return dense_vecs, lexical_weights  # lexical: token_id -> weight

    # -------------------------
    # Index build
    # -------------------------
    @staticmethod
    def _doc_text_from_chunks(doc: Dict[str, Any], max_chunks: int = 12, max_len: int = 2000) -> str:
        """
        当 doc["text"] 缺失时，使用其 chunks 拼接为索引文本（尽量稳定且不爆长）。
        """
        chunks = doc.get("chunks") or []
        parts = []
        for ch in chunks[: max_chunks]:
            t = (ch.get("text") or "").strip()
            if t:
                parts.append(t)
        text = " ".join(parts).strip()
        return text[:max_len]

    def build_doc_index(self, docs: List[Dict[str, Any]]):
        """
        建立索引（dense + sparse）。

        注意：
        - 索引单元仍然是 docs（文档级），RAGSystem 再从 doc["chunks"] 中抽 seeds
        - 如果 doc 缺少 text，则用 chunks 拼接得到 doc_text
        """
        if not docs:
            self.documents = []
            self.doc_dense_embeddings = np.zeros((0, 0), dtype=np.float32)
            self.doc_norms = np.zeros((0,), dtype=np.float32)
            self.doc_sparse_weights = []
            self.inverted_index = {}
            logger.warning("docs 为空：索引为空")
            return

        self.documents = docs
        self.doc_sparse_weights = []
        self.inverted_index = {}
        dense_list = []

        # 索引文本准备
        texts: List[str] = []
        for d in docs:
            txt = (d.get("text") or "").strip()
            if not txt:
                txt = self._doc_text_from_chunks(d)
            if not txt:
                # 严格但不崩：跳过空文档（否则 encode 会失败）
                txt = " "  # 最小占位，避免模型报错
            texts.append(txt)

        n = len(texts)
        bs = max(1, self.batch_size)
        logger.info("开始建立文档索引: N=%d bs=%d", n, bs)

        for start in range(0, n, bs):
            end = min(n, start + bs)
            batch_texts = texts[start:end]

            dense_vecs, lexical_list = self._encode_batch(batch_texts, return_dense=True, return_sparse=True)

            dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
            if dense_vecs.ndim != 2 or dense_vecs.shape[0] != (end - start):
                raise ValueError(f"dense_vecs shape invalid: got={dense_vecs.shape} expected=({end-start}, D)")

            # 维度检查（可选）
            if self.expected_dense_dim is not None:
                exp = int(self.expected_dense_dim)
                if dense_vecs.shape[1] != exp:
                    raise ValueError(f"dense dim mismatch: got={dense_vecs.shape[1]} expected={exp}")

            dense_list.append(dense_vecs)

            # sparse / inverted index
            if lexical_list is None:
                lexical_list = [{} for _ in range(end - start)]

            for local_i, sparse in enumerate(lexical_list):
                doc_idx = start + local_i
                if sparse is None:
                    sparse = {}
                self.doc_sparse_weights.append(sparse)

                for tid, wd in sparse.items():
                    self.inverted_index.setdefault(tid, []).append((doc_idx, float(wd)))

            if start % (bs * 10) == 0:
                logger.info("索引进度: %d/%d", start, n)

        self.doc_dense_embeddings = np.vstack(dense_list).astype(np.float32)

        norms = np.linalg.norm(self.doc_dense_embeddings, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)
        self.doc_norms = norms.astype(np.float32)

        logger.info(
            "文档索引建立完成: docs=%d dense_shape=%s inverted_terms=%d",
            len(self.documents),
            tuple(self.doc_dense_embeddings.shape),
            len(self.inverted_index),
        )

    # -------------------------
    # Save/Load
    # -------------------------
    def save_index(self, index_path: str):
        index_data = {
            "documents": self.documents,
            "doc_dense_embeddings": self.doc_dense_embeddings,
            "doc_norms": self.doc_norms,
            "doc_sparse_weights": self.doc_sparse_weights,
            "inverted_index": self.inverted_index,
        }
        with open(f"{index_path}.pkl", "wb") as f:
            pickle.dump(index_data, f)
        logger.info("索引已保存到: %s.pkl", index_path)

    def load_index(self, index_path: str) -> bool:
        pkl = f"{index_path}.pkl"
        if not os.path.exists(pkl):
            logger.warning("索引文件不存在: %s", pkl)
            return False

        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.doc_dense_embeddings = data["doc_dense_embeddings"]
            self.doc_norms = data["doc_norms"]
            self.doc_sparse_weights = data["doc_sparse_weights"]
            self.inverted_index = data["inverted_index"]

            logger.info("索引已加载: docs=%d", len(self.documents))
            return True
        except Exception as e:
            logger.error("加载索引失败: %s", e)
            return False

    # -------------------------
    # Dense retrieval
    # -------------------------
    def _dense_retrieve(self, h_q: np.ndarray):
        if self.doc_dense_embeddings is None or len(self.documents) == 0:
            return []

        h_q = np.asarray(h_q, dtype=np.float32).flatten()
        if h_q.ndim != 1:
            raise ValueError("h_q must be 1D vector")

        if self.expected_dense_dim is not None:
            exp = int(self.expected_dense_dim)
            if h_q.shape[0] != exp:
                raise ValueError(f"query dense dim mismatch: got={h_q.shape[0]} expected={exp}")

        q_norm = np.linalg.norm(h_q)
        q_norm = q_norm if q_norm > 0 else 1e-10

        sims = (self.doc_dense_embeddings @ h_q) / (self.doc_norms * q_norm)

        k = min(self.top_k_dense, sims.shape[0])
        if k <= 0:
            return []

        top_idx = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top_idx]

    # -------------------------
    # Sparse retrieval
    # -------------------------
    def _sparse_retrieve(self, s_q: Dict[Any, float]):
        if not self.inverted_index or len(self.documents) == 0:
            return []

        scores: Dict[int, float] = {}
        for tid, wq in (s_q or {}).items():
            postings = self.inverted_index.get(tid)
            if not postings:
                continue
            wq = float(wq)
            for doc_idx, wd in postings:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + wq * float(wd)

        if not scores:
            return []

        items = list(scores.items())
        items.sort(key=lambda x: x[1], reverse=True)
        k = min(self.top_k_sparse, len(items))
        return [(int(doc_idx), float(score)) for doc_idx, score in items[:k]]

    # -------------------------
    # Fusion
    # -------------------------
    def _fuse_scores_rrf(self, dense_list, sparse_list):
        rrf_k = max(1, self.rrf_k)
        fused: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(dense_list, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_idx, _) in enumerate(sparse_list, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank)

        items = list(fused.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def _fuse_scores_linear(self, dense_list, sparse_list):
        fused: Dict[int, float] = {}
        for doc_idx, s in dense_list:
            fused[doc_idx] = fused.get(doc_idx, 0.0) + self.lambda_dense * float(s)
        for doc_idx, s in sparse_list:
            fused[doc_idx] = fused.get(doc_idx, 0.0) + self.lambda_sparse * float(s)

        items = list(fused.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    # -------------------------
    # Public retrieval
    # -------------------------
    def hybrid_doc_retrieval(self, query: str, top_k_final: Optional[int] = None):
        """
        返回 top_k_final 个文档（doc dict），并附 score。
        约束（与新版 RAGSystem 对齐）：
        - doc 应该带 chunks，并且 chunk 至少包含 id/text
        - 若某 doc 没有有效 chunks，会被过滤（否则 seeds 为空）
        """
        query = (query or "").strip()
        if not query:
            return []

        if not self.documents:
            logger.warning("尚未建立/加载索引，documents为空")
            return []

        logger.info("执行初始文档检索: query='%s...'", query[:50])

        dense_vecs, lexical_list = self._encode_batch([query], return_dense=True, return_sparse=True)
        h_q = np.asarray(dense_vecs[0], dtype=np.float32)
        s_q = lexical_list[0] if lexical_list else {}
        s_q = s_q or {}

        dense_candidates = self._dense_retrieve(h_q)
        sparse_candidates = self._sparse_retrieve(s_q)

        if self.fusion_method == "linear":
            fused = self._fuse_scores_linear(dense_candidates, sparse_candidates)
        else:
            fused = self._fuse_scores_rrf(dense_candidates, sparse_candidates)

        k_final = int(top_k_final) if top_k_final is not None else self.top_k_final
        k = min(k_final, len(fused))
        top_docs = []
        for doc_idx, score in fused[:k]:
            doc = self.documents[doc_idx].copy()
            doc["score"] = float(score)

            # 过滤/修正 chunks：确保 seeds 可用
            chunks = doc.get("chunks") or []
            cleaned = []
            for ch in chunks:
                cid = ch.get("id")
                txt = ch.get("text")
                if cid and isinstance(txt, str) and txt.strip():
                    cleaned.append(ch)
            if not cleaned:
                # 没有可用 chunk，就不返回给 RAGSystem（否则会导致 seed_chunk_ids 为空）
                continue
            doc["chunks"] = cleaned
            top_docs.append(doc)

        logger.info("混合检索完成: 返回 %d 个文档", len(top_docs))
        return top_docs