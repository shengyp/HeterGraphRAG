# KSEM-RAG 完整方法论

## 系统概述

KSEM-RAG（Knowledge-aware Semantic Reasoning Architecture for Retrieval-Augmented Generation）是一个**多跳问答检索增强系统**，将语义理解、知识图谱推理和LLM结合，用于复杂多跳问题的答案生成。

### 核心设计理念

1. **意图控制**（Intent-Controlled）：根据问题的推理类型（BRIDGE/COMPARISON/DIRECT）动态调整检索和扩展策略
2. **混合检索**（Hybrid Retrieval）：BGE-M3稠密+稀疏混合检索提高召回率
3. **异构图推理**（Heterogeneous Graph Reasoning）：将chunks、propositions、entities整合成知识图，支持图上推理
4. **语义+结构融合**（Semantic+Structural Fusion）：LLM语义评分与图结构先验相结合

---

## 方法论工作流（8大核心步骤）

### **步骤1：意图识别（Intent Representation）**

**目标**：理解问题的推理结构和答案类型

**模块**：`IntentRepresentation`

**输入**：
- 用户查询（query）
- 粗检索的种子chunks（可选）

**输出**：
- `reasoning_type`：推理类型，取值为：
  - `BRIDGE`：顺序多跳（e.g., "A的父亲是谁，他来自哪里"）
  - `COMPARISON`：平行比较（e.g., "哪个更大/更有名"）
  - `DIRECT`：直接查询（e.g., "谁是..."）

- `answer_type`：答案类型，取值包括：
  - `PERSON`, `LOCATION`, `ORGANIZATION`, `WORK`, `EVENT`, `TIME`, `NUMBER`, `CONCEPT`, `BOOLEAN`, `DESCRIPTION`, `OTHER`

**实现方式**：
1. 从粗检索的chunks中提取最多4个上下文片段（max_ctx_snippets=4）
2. 调用Qwen2.5:14b LLM，输入包含query + 上下文片段
3. LLM返回JSON格式的意图信息
4. 解析并验证是否属于定义的REASONING_SPACE和ANSWER_TYPE_SPACE

**关键参数**：=
- `max_ctx_snippets`：最多取多少个seed chunk（默认4）
- `max_snippet_len`：每个snippet最大长度（默认260）
- `strict_methodology`：是否严格遵循方法论验证

---

### **步骤2：粗检索（Coarse Retrieval）**

**目标**：快速召回相关文档，生成seed chunks

**模块**：`BGEM3Retriever`

**输入**：
- 原始query
- 意图识别的推理类型（reasoning_type）
- 相关参数（top_k_dense, top_k_sparse等）

**输出**：
- 返回文档列表，每个文档包含chunks列表

**实现方式**：

#### **2.1 两种检索方式（基于推理类型）**

**平行推理（COMPARISON/PARALLEL）**：
```
query → BGE-M3混合检索 → top_k_final_parallel=40 文档
```

**顺序推理（BRIDGE/DIRECT）**：
```
第一轮: query → BGE-M3混合检索 → top_k_final_seq1=10 文档
第二轮: query + 第一轮top-2文档的evidence → BGE-M3混合检索 → top_k_final_seq2=20 文档
合并: 去重后保留所有文档
```
**first-hop evidence选择**：
- 从第一轮top-N文档（默认`feedback_docs=5`）中收集候选chunk（默认每篇最多`chunks_per_doc=6`）
- 对候选chunk与原始query重新计算BGE-M3稠密相似度和稀疏匹配分数
- 综合`dense_score`、`sparse_score`、第一轮文档分数、query词项覆盖率进行加权重排
- 去除重复片段后保留top evidence snippets（默认`max_snippets=4`），拼接到第二轮query
- 若重排失败或没有有效候选，回退到第一轮top-2文档的前几个chunk，保证流程可运行


#### **2.2 BGE-M3混合检索细节**

**稠密检索（Dense）**：
- 使用BGE-M3生成query和document的向量表示
- 计算余弦相似度，取top_k_dense=200

**稀疏检索（Sparse）**：
- 构建BM25/倒排索引
- 计算稀疏分数，取top_k_sparse=200

**融合策略**：
```
final_score = RRF(dense_rank, sparse_rank, rrf_k=60)
     或
final_score = lambda_dense × dense_score + lambda_sparse × sparse_score
```

**截断**：取top_k_final=20个文档作为检索结果

**关键参数**：
- `top_k_dense`：稠密检索初始候选（默认200）
- `top_k_sparse`：稀疏检索初始候选（默认200）
- `top_k_final`：最终返回文档数（默认20）
- `rrf_k`：RRF融合参数（默认60）
- `lambda_dense`/`lambda_sparse`：线性融合权重

---

### **步骤3：全局知识图构建（Graph Construction）**

**目标**：将所有chunks转化为异构知识图，支持后续的图上推理

**模块**：`GraphBuilder`

**输入**：
- 所有chunks（每个chunk包含text等信息）

**输出**：
- 异构有向图 `nx.DiGraph`，包含三类节点和多类边

**异构图结构**：

```
节点类型：
├── Chunk (C)：原始文本片段
├── Proposition (P)：原子陈述/事实
└── Entity (E)：实体/概念

边类型：
├── C → P：SUPPORTED_BY（chunk支撑proposition）
├── P → E：ASSERTS（proposition断言entity）
└── E → E：LINKED（entity与entity关联）

节点属性：
├── node_type：节点类型
├── text：节点文本内容
├── entity_type：实体类型（仅E节点，如PERSON/LOCATION等）
├── embedding：向量表示
├── s_struct_global：离线计算的结构中心性
├── s_final_q：查询时最终分数（query-time，在查询步骤中计算）
└── ... (其他元数据)
```

**构建过程**：

1. **Chunk节点创建**：直接使用chunk的id和text

2. **Proposition提取**（LLM-based）：
   - 调用Qwen2.5:14b LLM，输入chunk text
   - LLM提取原子陈述（e.g., "Albert Einstein was born in Germany"）
   - 为每个proposition创建节点

3. **Entity提取和类型标注**（LLM-based）：
   - 从proposition中提取entity
   - 调用LLM进行entity typing，标注为9类之一：
     - `PERSON`, `LOCATION`, `ORGANIZATION`, `WORK`, `EVENT`, `TIME`, `NUMBER`, `CONCEPT`, `OTHER`
   - 支持entity_type_cache.json缓存以加速

4. **边创建**：
   - `C → P`：每个proposition使用"SUPPORTED_BY"边连接到其源chunk
   - `P → E`：每个proposition中提到的entity使用"ASSERTS"边连接
   - `E → E`：同一proposition或相邻chunk中提到的entities之间创建"LINKED"边

5. **离线结构先验计算**（OfflineStructPrior）：
   - 计算全局betweenness centrality和closeness centrality
   - 组合为 `s_struct_global = alpha_core × core + alpha_betw × betw`
   - 存储到每个节点属性中

**关键参数**：
- `entity_type_labels`：9种entity类型
- `embedding_dim`：BGE-M3向量维度（1024）
- `flush_type_cache_each_write`：是否每次写入后刷新entity type缓存
- `offline_struct_prior.alpha_core`/`alpha_betw`：结构先验权重

---

### **步骤4：种子子图提取（Query Subgraph）**

**目标**：从粗检索的seed chunks出发，按固定拓扑扩展，生成与query相关的子图

**模块**：在`RAGSystem._build_query_subgraph_from_seeds()`

**输入**：
- 粗检索返回的seed_chunk_ids（列表）
- 全局图（global_graph）

**输出**：
- 查询子图 `G_q`：nx.DiGraph
- 绑定到seed chunks的proposition列表 `p_seed`

**扩展拓扑**：固定**两跳P-E-P**

```
L0: seed_chunks (C)
  ↓ SUPPORTED_BY (反向)
L1: p1 = propositions directly supported by seed chunks
  ↓ ASSERTS (正向)
L2: e = entities asserted by p1
  ↓ ASSERTS (反向)
L3: p2 = propositions asserting entities in e
  
子图 = {C, p1, e, p2} ∪ {C→p1→e→p2 的所有边}
```

**关键算法**：

```python
# Step 1: seed chunks → p1 (L1)
for cid in seed_chunk_ids:
    for p in predecessors(cid, edge_type="SUPPORTED_BY"):
        p1.add(p)

# Step 2: p1 → e (L2)
for pid in p1:
    for e in successors(pid, edge_type="ASSERTS"):
        e.add(e)

# Step 3: e → p2 (L3)
for eid in e:
    for p in successors(eid, edge_type="ASSERTS"):
        p2.add(p)

# 返回子图包含: {C, p1, e, p2}
```

**关键参数**：
- 固定拓扑：不可配置，严格为P-E-P两跳

**错误处理**：
- 若seed_chunk_ids为空，报错："粗检索未命中任何chunk"
- 若p_seed为空，报错："seed chunks未绑定到任何proposition"

---

### **步骤5：语义锚点评分（Semantic Anchor Selection）**

**目标**：用LLM对子图中的每个proposition进行语义评分，衡量其作为"语义锚点"（bridge/direct evidence）的价值

**模块**：`SemanticAnchorSelector`

**输入**：
- 查询子图 `G_q`
- query
- reasoning_type（BRIDGE/COMPARISON/DIRECT）
- answer_type（PERSON/LOCATION等）

**输出**：
- `scores`：Dict[prop_id, float]，分数范围[0, 1]
  - 0.80-1.00：非常重要（直接支持答案或关键bridge）
  - 0.50-0.79：清晰相关但不是核心
  - 0.20-0.49：背景信息，帮助有限
  - 0.00-0.19：基本无关

**评分逻辑**：

对每个proposition，调用LLM评分，包含以下关键context：

1. **语义类型提示（Semantic Hint）**：
   ```
   如果answer_type=PERSON：
   "Hint: Prefer facts about people, actors, authors..."
   
   如果answer_type=LOCATION：
   "Hint: Prefer facts about locations, cities, countries..."
   ```

2. **推理类型指导**：
   ```
   若reasoning_type=BRIDGE/SEQUENTIAL：
   "prioritize propositions that introduce next-hop entities"
   
   若reasoning_type=COMPARISON/PARALLEL：
   "prioritize propositions supporting comparison/alignment"
   
   若reasoning_type=DIRECT：
   "prioritize propositions directly supporting answer"
   ```

3. **LLM Prompt框架**：
   ```
   # Role: semantic anchor scorer in multi-hop QA
   # Task: evaluate proposition value for this question [0, 1]
   # Scoring Criteria: [0.80-1.00] → 0.00-0.19 (详细标准)
   # Reasoning Guidance: (推理类型指导)
   # Answer-Type Guidance: (软偏好，不硬拒)
   # Examples: (参考例子)
   
   Question: {query}
   Proposition: {prop_text}
   → Output JSON with score and reasoning
   ```

**关键参数**：
- `max_node_text_len`：proposition文本最大长度（默认420）
- `llm_score_timeout`：LLM超时（默认20秒）

---

### **步骤6：融合排序与锚点选择（Structural Centrality Ranking & Anchor Selection）**

**目标**：融合离线结构先验（s_struct_global）和查询时语义分（s_sem），选出top-k anchor propositions

**模块**：`StructuralCentralityRanker` + `RAGSystem._select_anchor_props()`

**输入**：
- 查询子图 `G_q`
- 语义分 `s_sem`（来自步骤5）

**输出**：
- anchor_props：选中的top-8个proposition的列表

**融合公式**：

对每个proposition `pid`：
$$s_{final} = \beta_{struct} \times s_{struct\_global}(pid) + \gamma_{sem} \times s_{sem}(pid)$$

其中：
- `s_struct_global(pid)`：离线计算的结构中心性（存储在节点属性）
- `s_sem(pid)`：LLM语义评分（来自步骤5）
- `beta_struct`：结构权重（默认1.0）
- `gamma_sem`：语义权重（默认0.5）

**选择策略**：
1. 计算子图中所有propositions的融合分
2. 按分数降序排序
3. 取top-k_prop=8个作为锚点

**关键参数**：
- `gamma_sem`：语义分权重（默认0.5）
- `beta_struct`：结构分权重（默认1.0）
- `k_prop`：保留的锚点数（默认12，但通常被上层覆盖为8）

---

### **步骤7：意图控制的两跳扩展（Intent-Controlled Expansion）**

**目标**：基于推理类型，从anchor propositions出发进行受控扩展，获得更多相关propositions

**模块**：`RAGSystem._intent_controlled_expand()`

**输入**：
- anchor_props：来自步骤6的锚点propositions
- answer_type：来自步骤1
- 全局图

**输出**：
- expanded_props：新增的propositions（不包括anchor_props）

**扩展拓扑**：与步骤4相同，P-E-P两跳

```
P_anchor
  ↓ ASSERTS (正向)
E: 由anchor提到的entities
  ↓ ASSERTS (反向)
P_expand: 由这些entities提到的其他propositions
```

**意图控制的约束**：

**如果answer_type属于{PERSON, LOCATION, ORGANIZATION, TIME, NUMBER, WORK, EVENT}**：
- Layer2强约束：只保留entity_type与answer_type相匹配的propositions
- 理由：防止无关实体的扩展
- Fallback：若无匹配propositions，使用原始candidates

**如果answer_type属于{CONCEPT, BOOLEAN, DESCRIPTION, OTHER}**：
- Layer2宽松约束：保留所有propositions
- 理由：概念/描述类答案可从多角度支持

**分数传播**：
```
对expanded propositions中的每个p：
s_final_q(p) = expand_struct_mix × s_struct_global(p) 
              + (1 - expand_struct_mix) × decay × s_final_q(parent_anchor)
```

其中：
- `expand_struct_mix`：权重比（默认0.3，给结构分30%权重）
- `expand_decay`：衰减因子（默认0.8，两跳扩展的score乘以0.8）

**关键参数**：
- `top_entity_expand`：Layer2保留的entities数（默认6）
- `expand_decay`：衰减因子（默认0.8）
- `expand_struct_mix`：结构分权重（默认0.3）

---

### **步骤8.1：最终Proposition合并与截断**

**目标**：合并anchor + expanded propositions，再次排序，截断至max_final_p

**模块**：在`RAGSystem.query()` → 最终merge逻辑

**输入**：
- anchor_props（来自步骤6）
- expanded_props（来自步骤7）

**输出**：
- final_props：最多18个propositions

**处理流程**：

1. **去重**：
   ```python
   merged_props = list(dict.fromkeys(anchor_props + expanded_props))
   ```

2. **重排**：
   ```
   按 s_final_q 降序排序
   （该分数在步骤7中已为扩展propositions计算，
    anchor propositions使用步骤6的融合分）
   ```

3. **截断**：
   ```python
   final_props = merged_props[:max_final_p]  # max_final_p=18
   ```

**关键参数**：
- `max_final_p`：最终保留的propositions数（默认18）

---

### **步骤8.2：证据聚合（Evidence Aggregation）**

**目标**：从final propositions反向聚合chunks和entities，用于答案生成

**模块**：`RAGSystem._build_evidence_buckets()`

**输入**：
- final_props（来自步骤8.1）
- 全局图

**输出**：
- 证据buckets，包含：
  - `chunk`：按相关性排序的chunks
  - `proposition`：proposition列表及其分数
  - `entity`：实体及其类型
  - `type`：实体类型列表（用于消歧）

**聚合逻辑**：

1. **从final_props反向查找chunks**：
   ```
   对每个 prop_id in final_props:
       对每个 chunk_id in predecessors(prop_id, edge_type="SUPPORTED_BY"):
           chunks.append(chunk_id)
   截断：max_chunks_per_p × len(final_props) = 3×18 = 54
   ```

2. **从final_props正向查找entities**：
   ```
   对每个 prop_id in final_props:
       对每个 entity_id in successors(prop_id, edge_type="ASSERTS"):
           entities.append((entity_id, entity_type))
   截断：k_e = 12
   ```

3. **去重并排序**：
   - Chunks按出现频次/分数排序
   - Entities按类型和分数排序

**关键参数**：
- `max_chunks_per_p`：每个proposition最多关联chunks数（默认3）
- `k_e`：保留的entity数（默认12）

---

### **步骤8.3：答案生成（Answer Generation）**

**目标**：基于聚合的证据，用LLM生成自然语言答案

**模块**：`AnswerGenerator.generate_answer()`

**输入**：
- query：原始问题
- ranked_nodes：证据buckets
  - propositions（最多18个）
  - chunks（最多54个）
  - entities（最多12个）
  - types（实体类型列表）
- 全局图
- intent_topo：推理类型（BRIDGE/COMPARISON/DIRECT）
- intent_sem：答案类型（PERSON/LOCATION等）

**输出**：
- 包含字段的字典：
  - `answer`：生成的答案文本
  - `reasoning`：推理过程（可选）
  - `sources`：证据来源的proposition/chunk ids
  - `metadata`：调试信息

**生成过程**：

1. **证据准备**（文本截断）：
   ```
   k_p = 5: 最多5个top propositions
   k_c = 5: 最多5个top chunks
   k_e = 5: 最多5个top entities
   k_z = 5: 最多5个实体类型
   
   max_len_prop = 520: 每个proposition最多520字符
   max_len_chunk = 650: 每个chunk最多650字符
   max_len_ent = 120: 每个entity最多120字符
   max_len_type = 40: 每个类型最多40字符
   ```

2. **LLM Prompt构建**：
   ```
   # Role: Answer generator in multi-hop QA
   # Task: Generate answer based on evidence
   # Reasoning Type: {intent_topo}
   # Answer Type: {intent_sem}
   
   Evidence:
   Propositions: {top_props}
   Chunks: {top_chunks}
   Entities: {top_entities}
   Types: {entity_types}
   
   Question: {query}
   → Output: {"answer": "...", "reasoning": "..."}
   ```

3. **LLM生成**：
   - 模型：Qwen2.5:14b
   - temperature：0.2（低随机性）
   - num_predict：240（最多生成240 tokens）
   - timeout：90秒

**关键参数**：
- `k_per_type`：默认值5（控制p/c/e/z）
- `temperature`：0.2（确定性）
- `num_predict`：240（答案长度）

---

## 核心配置参数总览

```yaml
strict_methodology: true  # 是否严格遵循方法论验证

coarse_retrieval:
  model_name: /path/to/bge-m3
  top_k_dense: 200        # 稠密检索初始候选数
  top_k_sparse: 200       # 稀疏检索初始候选数
  top_k_final: 20         # 混合检索最终返回数
  fusion_method: rrf      # 融合方法: rrf 或 linear
  rrf_k: 60               # RRF参数
  lambda_dense: 0.5       # 线性融合的稠密权重
  lambda_sparse: 0.5      # 线性融合的稀疏权重
  top_k_final_parallel: 40    # 平行推理的返回数
  top_k_final_seq1: 10        # 顺序推理第一轮数
  top_k_final_seq2: 20        # 顺序推理第二轮数

graph_construction:
  ollama_base_url: http://localhost:11434
  llm_model: qwen2.5:14b
  embedding_model: bge-m3:latest
  embedding_dim: 1024
  entity_type_labels: [PERSON, LOCATION, ORGANIZATION, ...]
  entity_type_cache_path: /path/to/cache.json
  strict_methodology: true
  offline_struct_prior:
    alpha_core: 0.5       # core centrality权重
    alpha_betw: 0.5       # betweenness centrality权重
    attr: s_struct_global # 存储属性名

intent_representation:
  ollama_base_url: http://localhost:11434
  llm_model: qwen2.5:14b
  max_ctx_snippets: 4     # seed chunks的上下文数
  max_snippet_len: 260    # 每个snippet的最大长度
  strict_methodology: true

semantic_anchor:
  llm_model: qwen2.5:14b
  llm_score_timeout: 20   # LLM超时（秒）
  max_node_text_len: 420  # proposition文本截断

structural_centrality:
  gamma_sem: 0.5          # 语义分权重
  beta_struct: 1.0        # 结构分权重
  k_prop: 12              # 保留的proposition数
  k_ent: 12               # 保留的entity数

rerank:
  top_anchor_p: 8         # 锚点proposition数（覆盖k_prop）
  top_entity_expand: 6    # 扩展时保留的entities
  max_final_p: 18         # 最终proposition截断数
  max_chunks_per_p: 3     # 每个proposition关联的chunk数
  expand_decay: 0.8       # 扩展proposition的衰减因子
  expand_struct_mix: 0.3  # 扩展时的结构分权重

answer_generation:
  llm_model: qwen2.5:14b
  k_per_type: 5           # 每类证据的默认数（p/c/e/z）
  max_len_prop: 520       # proposition截断长度
  max_len_chunk: 650      # chunk截断长度
  max_len_ent: 120        # entity截断长度
  max_len_type: 40        # 类型截断长度
  temperature: 0.2        # LLM生成温度
  num_predict: 240        # LLM最大生成token数
  timeout: 90             # LLM超时（秒）
```

---

## 关键创新点

### 1. **意图控制的推理路由**
- 不同推理类型（BRIDGE/COMPARISON/DIRECT）采用不同的检索策略
- BRIDGE：两轮顺序检索（第一轮10个文档，第二轮基于evidence的20个文档）
- COMPARISON：一次平行检索（40个文档）
- DIRECT：逻辑同BRIDGE

### 2. **LLM-in-the-loop的细粒度评分**
- 不仅评分直接答案，也评分"关键bridge propositions"
- 根据推理类型和答案类型提供动态提示（semantic hints）
- 分数范围[0, 1]，包含详细的分数标准

### 3. **异构图上的受控扩展**
- 固定P-E-P两跳拓扑，确保可解释性
- 意图控制的entity过滤（按answer_type）
- 分数传播机制（衰减+权重混合）

### 4. **结构先验与查询时语义的融合**
$$s_{final} = \beta_{struct} \times s_{struct\_global} + \gamma_{sem} \times s_{sem}$$
- 离线计算结构中心性（betweenness + closeness）
- 查询时计算LLM语义分
- 加权融合，支持参数调节

### 5. **多跳问答的端到端管道**
- 从粗检索的chunks → 知识图 → 子图 → 锚点 → 扩展 → 证据聚合 → 答案生成
- 每个步骤都有明确的输入输出和可验证的约束

---

## 方法论验证与调试

### 严格模式（strict_methodology=true）

启用以下验证：
- 每个步骤的输入/输出不能为空，否则抛出RuntimeError
- Entity type filter必须有匹配结果，否则报警
- Intent representation必须返回定义的reasoning_type和answer_type

### 元数据追踪

每个query的结果中包含metadata，记录：
- `reasoning_type`/`answer_type`：识别结果
- `num_seed_chunks`：粗检索的chunks数
- `num_q_nodes`/`num_q_edges`：子图大小
- `num_anchor_props`/`num_final_props`：锚点和最终propositions数
- `gamma_sem`/`beta_struct`：融合参数
- 其他调试信息（entity type filter匹配情况等）

---

## 总结：8步工作流

```
1. 意图识别（query → reasoning_type, answer_type）
         ↓
2. 粗检索（query, reasoning_type → seed_chunks）
         ↓
3. 知识图构建（所有chunks → 全局异构图）
         ↓
4. 种子子图提取（seed_chunks → G_q）
         ↓
5. 语义锚点评分（G_q, query, reasoning_type, answer_type → s_sem）
         ↓
6. 融合排序与锚点选择（s_struct_global, s_sem → anchor_props）
         ↓
7. 意图控制的两跳扩展（anchor_props, answer_type → expanded_props）
         ↓
8. 最终合并、证据聚合与答案生成（anchor+expanded → final_props → answer）
```

---

## 环境与依赖

**Python版本**：3.10 / 3.11

**核心库**：
- `FlagEmbedding`（BGE-M3混合检索）
- `networkx`（异构图表示与操作）
- `requests`（Ollama API调用）
- `numpy`（向量计算）

**推理引擎**：
- Ollama（本地LLM推理服务）
- Qwen2.5:14b（LLM主模型）
- bge-m3:latest（嵌入模型）

**注意**：
- Ollama需要单独启动（`ollama serve`）
- BGE-M3嵌入维度固定为1024
- 所有LLM调用通过HTTP到Ollama（默认11434端口）

