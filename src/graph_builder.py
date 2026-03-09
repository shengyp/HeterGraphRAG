

import json
import logging
import re
import hashlib
from typing import Dict, List, Tuple, Set

import networkx as nx
import numpy as np
import requests

logger = logging.getLogger(__name__)


class GraphBuilder:

    # 0) 初始化与配置
    def __init__(self, config: Dict):

        required = ["ollama_base_url", "llm_model", "embedding_dim", "entity_type_labels", "entity_type_cache_path"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"GraphBuilder 配置缺少必填项: {', '.join(missing)}")

        self.ollama_base_url = str(config["ollama_base_url"]).rstrip("/")
        self.llm_model = str(config["llm_model"])
        self.embedding_dim = int(config["embedding_dim"])


        self.embedding_model = str(config.get("embedding_model", "bge-m3:latest"))
        self.flush_type_cache_each_write = bool(config.get("flush_type_cache_each_write", False))


        labels = config["entity_type_labels"]
        if not isinstance(labels, list) or not labels:
            raise ValueError("entity_type_labels 必须是非空 list")
        self.type_labels: List[str] = sorted({str(x).strip().upper() for x in labels if str(x).strip()})
        if not self.type_labels:
            raise ValueError("entity_type_labels 清洗后为空")
        self.type_label_set: Set[str] = set(self.type_labels)


        self.type_cache_path = str(config["entity_type_cache_path"])
        self.type_cache: Dict[str, str] = self._load_type_cache()

        logger.info(
            "GraphBuilder(strict) 初始化完成 | embedding_model=%s dim=%d | labels=%d | cache=%s",
            self.embedding_model,
            self.embedding_dim,
            len(self.type_labels),
            self.type_cache_path,
        )


    # 1) 通用工具
    def _norm_entity(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def _entity_node_id(self, entity_text: str) -> str:
        """实体节点ID（保证同名实体合并）"""
        return f"ent::{self._norm_entity(entity_text)}"

    def _type_node_id(self, type_label: str) -> str:
        """类型节点ID"""
        return f"type::{type_label.strip().upper()}"


    def _stable_prop_id(self, prop_text: str) -> str:

        h = hashlib.sha1(prop_text.encode("utf-8")).hexdigest()[:10]
        return f"prop::{h}"

    def _encode_text(self, text: str) -> List[float]:

        text = text.strip()
        if not text:
            raise ValueError("embedding 输入文本为空（严格模式不允许）")

        url = f"{self.ollama_base_url}/api/embeddings"
        resp = requests.post(url, json={"model": self.embedding_model, "prompt": text}, timeout=60)
        #检查http状态
        resp.raise_for_status()

        emb = (resp.json()).get("embedding", None)
        if emb is None:
            raise ValueError("embeddings 接口未返回 embedding 字段")

        #把embedding压成一维
        vec = np.asarray(emb, dtype=float).flatten()
        if vec.shape[0] != self.embedding_dim:
            raise ValueError(f"embedding 维度错误: got={vec.shape[0]} expected={self.embedding_dim}")
        return vec.tolist()

    def _extract_first_json_object(self, raw: str) -> str:

        if not raw:
            raise ValueError("LLM 输出为空")

        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        start = s.find("{")
        if start < 0:
            raise ValueError("LLM 输出中找不到 '{'")

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

        raise ValueError("JSON 大括号不平衡，无法抽取完整对象")

    # =========================
    # 2) 类型缓存（确定性工具，不是兜底）
    # =========================
    def _load_type_cache(self) -> Dict[str, str]:
        try:
            with open(self.type_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("type_cache 文件必须是 JSON object")

            out: Dict[str, str] = {}
            for k, v in data.items():
                kk = str(k).strip()
                vv = str(v).strip().upper()
                if kk and vv:
                    out[kk] = vv
            return out
        except FileNotFoundError:
            return {}
        except Exception as e:
            raise ValueError(f"加载 type_cache 失败: {e}")

    def _save_type_cache(self) -> None:
        with open(self.type_cache_path, "w", encoding="utf-8") as f:
            json.dump(self.type_cache, f, ensure_ascii=False, indent=2)

    # =========================
    # 3) Step 1: chunk -> facts & entities（英文提示词：Role/Task/Example）
    # =========================
    def extract_facts_and_entities_from_chunk(self, chunk: Dict) -> Tuple[List[Dict], List[str]]:
        """
        从 chunk 文本抽取：
        - entities: 规范化实体列表（字符串）
        - facts: 原子事实列表，每条带 triplets、entity_ids、source_chunk_id
        """
        cid = chunk.get("id")
        text = chunk.get("text")
        if not (isinstance(text, str) and text.strip()):
            raise ValueError(f"chunk 文本为空: chunk_id={cid}")

        prompt = f"""# Role:
        You are a structured information extractor for HotpotQA multi-hop question answering.
        Your task is to convert a given chunk of text into structured facts that can be used for graph construction.

        You must extract information only from the input text. Do not add outside knowledge, do not guess, and do not alter the meaning of the original text.

        ## Task:
        Given one chunk of text, extract:

        1. entities:
        - A list of entities explicitly mentioned in the text
        - Do not output pronouns (such as he, she, it, they, his, her, its, their)
        - Use standard, complete, and uniquely identifiable names whenever possible
        - Do not casually treat pure attribute words, function words, or generic category words as entities (for example, "actor", "city", and "year" are generally not entities)

        2. facts:
        - Extract one or more atomic facts
        - Each fact must express exactly one minimal, independent, and citable fact
        - Do not merge multiple relations into a single fact
        - Every fact must be directly supported by the input text
        - If the text does not contain any clear fact, do not fabricate one; return an empty list

        3. triplets:
        - Each fact must correspond to one or more triplets
        - Each triplet must follow the format: [head, relation, tail]
        - The head and tail should come from entities whenever possible; if not suitable as entities, use the original literal text
        - The relation should be concise, stable, and semantically clear, preferably as a short phrase, such as:
          "is", "was born in", "starred", "directed by", "located in", "part of", "published in", "married to"
        - Triplets must be strictly semantically consistent with their corresponding fact

        ## Important Constraints
        - Never fabricate information that is not explicitly stated in the input text
        - Never use background knowledge outside the input text to fill in missing information
        - Preserve literals such as time, year, number, and title exactly as they appear, for example: "1999", "42", "president"
        - If one sentence contains multiple facts, split them into multiple atomic facts
        - If the relation between two entities is unclear, do not force a relation
        - Do not output duplicate entities
        - Do not output duplicate facts
        - Do not output any schema, explanation, or annotation unrelated to the text

        ## Output Format
        You must output exactly one strictly valid JSON object. Do not output markdown, do not output code fences, and do not output any explanatory text.

        The JSON format is:
        {
        "entities": ["Entity 1", "Entity 2"],
          "facts": [
            {
        "fact": "One atomic fact sentence",
              "triplets": [
                ["Entity 1", "relation", "Entity 2 or literal"]
              ]
            }
          ]
        }

        ## Output Requirements
        - The top level must contain exactly two keys: "entities" and "facts"
        - "entities" must be an array of strings
        - "facts" must be an array
        - Each fact object must contain exactly two keys: "fact" and "triplets"
        - "fact" must be a string
        - "triplets" must be an array
        - Each triplet must have length 3, and all three elements must be strings
        - Even if nothing can be extracted, you must still output valid JSON, for example:
          {"entities": [], "facts": []}

        ## Example
        Input chunk:
        The Newcomers is a 2000 film starring Chris Evans.

        Output:
        {
        "entities": ["The Newcomers", "Chris Evans", "2000"],
          "facts": [
            {
        "fact": "The Newcomers is a 2000 film.",
              "triplets": [
                ["The Newcomers", "is", "2000 film"]
              ]
            },
            {
        "fact": "The Newcomers starred Chris Evans.",
              "triplets": [
                ["The Newcomers", "starred", "Chris Evans"]
              ]
            }
          ]
        }

        ## Input chunk
        {text}

        ## Output
        Output JSON only.
        """

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        resp.raise_for_status()

        raw = (resp.json() or {}).get("response", "").strip()
        js = self._extract_first_json_object(raw)
        data = json.loads(js)

        entities = data.get("entities")
        facts_list = data.get("facts")

        if not isinstance(entities, list) or not entities:
            raise ValueError(f"LLM 抽取 entities 失败或为空: chunk_id={cid}")
        if not all(isinstance(x, str) and x.strip() for x in entities):
            raise ValueError(f"LLM 返回 entities 格式不合法: chunk_id={cid}")

        if not isinstance(facts_list, list) or not facts_list:
            raise ValueError(f"LLM 抽取 facts 失败或为空: chunk_id={cid}")

        facts: List[Dict] = []
        for idx, item in enumerate(facts_list):
            if not isinstance(item, dict):
                raise ValueError(f"facts[{idx}] 不是 dict: chunk_id={cid}")

            fact_text = item.get("fact")
            triplets = item.get("triplets")

            if not (isinstance(fact_text, str) and fact_text.strip()):
                raise ValueError(f"facts[{idx}].fact 为空: chunk_id={cid}")
            if not isinstance(triplets, list) or not triplets:
                raise ValueError(f"facts[{idx}].triplets 为空或不合法: chunk_id={cid}")


            ft_low = fact_text.lower()
            fact_entities: List[str] = []
            for e in entities:
                #用匹配来进行绑定
                if e.lower() in ft_low:
                    fact_entities.append(e)

            for tri in triplets:
                if not (isinstance(tri, list) and len(tri) >= 3):
                    raise ValueError(f"facts[{idx}].triplets 存在非法三元组: chunk_id={cid}")
                head, _, tail = tri[0], tri[1], tri[2]
                if head in entities and head not in fact_entities:
                    fact_entities.append(head)
                if tail in entities and tail not in fact_entities:
                    fact_entities.append(tail)

            if not fact_entities:
                raise ValueError(f"facts[{idx}] 未能绑定任何实体（严格模式不允许）: chunk_id={cid}")

            facts.append(
                {
                    "id": f"{cid}_fact_{idx}",
                    "text": fact_text.strip(),
                    "triplets": triplets,
                    "entity_ids": fact_entities,
                    "source_chunk_id": cid,
                }
            )

        return facts, entities


    # 4) Step 2: facts -> propositions

    def build_proposition_nodes(self, facts: List[Dict]) -> Tuple[List[Dict], Dict[str, List[str]]]:
        if not facts:
            raise ValueError("facts 为空")

        buckets: Dict[str, List[Dict]] = {}
        for f in facts:
            eids = f.get("entity_ids")
            if not eids:
                raise ValueError(f"fact 缺少 entity_ids: fact_id={f.get('id')}")
            for e in eids:
                #如果 e 这个键已经存在，就返回它对应的值，如果不存在，就先创建
                buckets.setdefault(e, []).append(f)

        if not buckets:
            raise ValueError("按实体分桶后 buckets 为空（严格模式）")
        #开始去重
        prop_dedup: Dict[str, Dict] = {}

        for _, bucket_facts in buckets.items():
            seen = set()
            uniq_texts: List[str] = []
            all_entities: Set[str] = set()
            fact_ids: List[str] = []
            source_chunks: Set[str] = set()

            for f in bucket_facts:
                t = (f.get("text") or "").strip()
                if not t:
                    raise ValueError(f"fact text 为空: fact_id={f.get('id')}")

                if t not in seen:
                    uniq_texts.append(t)
                    seen.add(t)

                all_entities.update(f.get("entity_ids") or [])
                if not f.get("id"):
                    raise ValueError("fact 缺少 id")
                fact_ids.append(f["id"])

                scid = f.get("source_chunk_id")
                if not scid:
                    raise ValueError(f"fact 缺少 source_chunk_id: fact_id={f.get('id')}")
                source_chunks.add(scid)

            prop_text = " ".join(uniq_texts[:10]).strip()
            if not prop_text:
                raise ValueError("prop_text 为空（严格模式）")

            dedup_key = prop_text
            #如果这个 proposition 还没出现过就新建
            if dedup_key not in prop_dedup:
                prop_dedup[dedup_key] = {
                    "text": prop_text,
                    "entity_ids": set(all_entities),
                    "fact_ids": list(fact_ids),
                    "source_chunk_ids": set(source_chunks),
                }
            #如果出现过就合并
            else:
                prop_dedup[dedup_key]["entity_ids"].update(all_entities)
                prop_dedup[dedup_key]["fact_ids"].extend(fact_ids)
                prop_dedup[dedup_key]["source_chunk_ids"].update(source_chunks)

        propositions: List[Dict] = []
        fact_assignments: Dict[str, List[str]] = {}

        for prop_text, pd in prop_dedup.items():
            #根据 prop_text 生成一个稳定 id
            pid = self._stable_prop_id(prop_text)

            entity_ids = sorted({x for x in pd["entity_ids"] if isinstance(x, str) and x.strip()})
            if not entity_ids:
                raise ValueError(f"P 缺少 entity_ids: prop_id={pid}")

            fact_ids = sorted(set(pd["fact_ids"]))
            if not fact_ids:
                raise ValueError(f"P 缺少 fact_ids: prop_id={pid}")

            source_chunk_ids = sorted(set(pd["source_chunk_ids"]))
            if not source_chunk_ids:
                raise ValueError(f"P 缺少 source_chunk_ids: prop_id={pid}")

            propositions.append(
                {
                    "id": pid,
                    "text": prop_text,
                    "entity_ids": entity_ids,
                    "fact_ids": fact_ids,
                    "source_chunk_ids": source_chunk_ids,
                    "embedding": self._encode_text(prop_text),
                }
            )
            fact_assignments[pid] = fact_ids

        if not propositions:
            raise ValueError("propositions 为空")
        return propositions, fact_assignments


    # 5) Step 3: entity -> type

    def infer_entity_type(self, entity: str, context: str) -> str:
        canon = self._norm_entity(entity)

        if not canon:
            raise ValueError("entity为空")

        if canon in self.type_cache:

            t = str(self.type_cache[canon]).strip().upper()
            if t not in self.type_label_set:
                raise ValueError(f"type_cache 中存在非法标签: {t} for entity={entity}")
            return t

        #把所有可选标签拼成一个字符串，中间用 | 连接
        labels_str = " | ".join(self.type_labels)
        ctx = context.strip()
        if len(ctx) > 220:
            ctx = ctx[:220]

        prompt = f"""# Role
        You are an entity type classifier for evidence-graph construction.

        # Task
        Given:
        1. one entity string
        2. one short context snippet

        choose exactly one label for the entity from the allowed labels.

        # Allowed labels
        {labels_str}

        # Labeling rules
        1. You must classify ONLY based on the given entity string and the given context.
        2. Do NOT use outside world knowledge unless it is explicitly supported by the context.
        3. If the entity refers to a human, real person, fictional character, or named individual, choose "PERSON".
        4. If the entity refers to a geographic location, place, region, country, city, state, province, continent, or other physical location, choose "LOCATION".
        5. Otherwise, choose "OTHER".
        6. If the context is insufficient or ambiguous, choose the most conservative label.
        7. Do NOT explain your reasoning.
        8. Output exactly one JSON object and nothing else.

        # Important boundary cases
        - film / book / organization / company / award / event / brand / school / universe / series / concept -> OTHER
        - nationality / language / date / year / occupation / role / genre -> OTHER
        - fictional people with personal names -> PERSON
        - named places in fictional or real settings -> LOCATION

        # Output format
        {{"type":"<ONE_LABEL>"}}

        # Valid output examples
        {{"type":"PERSON"}}
        {{"type":"LOCATION"}}
        {{"type":"OTHER"}}

        # Input
        Entity: "{entity}"
        Context: "{ctx}"

        # Output
        Return JSON only.
        """
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 40},
            },
            timeout=90,
        )
        resp.raise_for_status()

        raw = (resp.json() or {}).get("response", "").strip()
        js = self._extract_first_json_object(raw)
        data = json.loads(js)

        if "type" not in data:
            raise ValueError(f"LLM type 输出缺少 'type' 字段: entity={entity}")

        t = str(data["type"]).strip().upper()
        if t not in self.type_label_set:
            raise ValueError(f"LLM type 输出不在允许集合内: type={t} entity={entity}")

        # 写缓存
        self.type_cache[canon] = t
        if self.flush_type_cache_each_write:
            self._save_type_cache()
        return t


    # 6) 主入口：构建 E–P–E 图

    def build_heterogeneous_graph(self, chunks: List[Dict]) -> nx.DiGraph:

        if not chunks:
            raise ValueError("chunks 为空")

        G = nx.DiGraph()

        # ---------- (A) 加入 C 节点（证据容器）----------
        for c in chunks:
            cid = c.get("id")
            if not cid:
                raise ValueError("chunk 缺少 id")
            if not (isinstance(c.get("text"), str) and c["text"].strip()):
                raise ValueError(f"chunk 缺少 text: chunk_id={cid}")

            if "embedding" in c and c["embedding"] is not None:
                vec = np.asarray(c["embedding"], dtype=float).flatten()
                if vec.shape[0] != self.embedding_dim:
                    raise ValueError(f"chunk embedding 维度错误: chunk_id={cid}")
                emb = vec.tolist()
            else:
                emb = self._encode_text(c["text"])

            G.add_node(
                cid,
                node_type="chunk",
                text=c["text"],
                embedding=emb,
            )

        # ---------- (B) 抽取 facts/entities，并准备 entity 的轻上下文 ----------
        all_facts: List[Dict] = []
        #实体识别的上下文（避免歧义）
        entity_ctx: Dict[str, str] = {}

        for c in chunks:
            cid = c["id"]
            facts, entities = self.extract_facts_and_entities_from_chunk(c)

            if not entities:
                raise ValueError(f"chunk entities 为空: chunk_id={cid}")

            # 同一实体首次出现的 chunk 作为上下文
            for e in entities:
                canon = self._norm_entity(e)
                if canon and canon not in entity_ctx:
                    entity_ctx[canon] = c["text"]

            all_facts.extend(facts)


            G.nodes[cid]["entity_mentions"] = entities
            G.nodes[cid]["facts_extracted"] = facts

        if not all_facts:
            raise ValueError("全局 all_facts 为空")

        # ---------- (C) 构建 P 节点（推理核心）----------
        props, _ = self.build_proposition_nodes(all_facts)

        for p in props:
            pid = p["id"]
            G.add_node(
                pid,
                node_type="proposition",
                text=p["text"],
                embedding=p["embedding"],
                entity_ids=p["entity_ids"],
                fact_ids=p["fact_ids"],
                source_chunk_ids=p["source_chunk_ids"],
            )

            # P -> C（
            for cid in p["source_chunk_ids"]:
                if not G.has_node(cid):
                    raise ValueError(f"P 引用了不存在的 chunk: prop_id={pid} chunk_id={cid}")
                G.add_edge(pid, cid, edge_type="SUPPORTED_BY", weight=1.0)

        # ---------- (D) 构建 E 节点、Z 节点，并连 EP / EZ ----------
        created_entities: Set[str] = set()

        for p in props:
            pid = p["id"]
            ent_node_ids: List[str] = []

            for e in p["entity_ids"]:
                canon = self._norm_entity(e)
                if not canon:
                    raise ValueError(f"实体 canonicalize 为空: entity={e} prop_id={pid}")

                eid = self._entity_node_id(e)
                ent_node_ids.append(eid)

                if not G.has_node(eid):

                    G.add_node(
                        eid,
                        node_type="entity",
                        text=e,
                        canonical=canon,
                        embedding=self._encode_text(canon),
                    )

                # 每个实体只做一次 typing
                if eid not in created_entities:
                    created_entities.add(eid)

                    ctx = entity_ctx.get(canon, "")
                    t = self.infer_entity_type(e, ctx)

                    zid = self._type_node_id(t)
                    if not G.has_node(zid):
                        G.add_node(zid, node_type="type", text=t)

                    # E -> Z
                    G.add_edge(eid, zid, edge_type="IS_A", weight=1.0)

                # EP 双向边
                G.add_edge(eid, pid, edge_type="INVOLVED_IN", weight=1.0)
                G.add_edge(pid, eid, edge_type="ASSERTS", weight=1.0)

            # 写回 proposition 的 entity_node_ids
            G.nodes[pid]["entity_node_ids"] = ent_node_ids

        # (D) 离线：结构先验（只算一次）
        from src.offline_struct_prior import compute_and_write_struct_prior

        sc_cfg = (self.config.get("structural_centrality", {}) or {})
        compute_and_write_struct_prior(
            G,
            alpha_core=float(sc_cfg.get("alpha_core", 0.5)),
            alpha_betw=float(sc_cfg.get("alpha_betw", 0.5)),
            attr="s_struct_global",
        )


        # (E) 落盘 cache
        self._save_type_cache()

        logger.info("构图完成: nodes=%d edges=%d", G.number_of_nodes(), G.number_of_edges())
        return G