# HeterGraphRAG
# 项目检查说明与启动 README

## 1. 项目概览

这个项目是一个 **面向 HotpotQA 风格多跳问答的异构图 RAG 系统**。整体流程可以概括为：

1. 将 HotpotQA 样本整理为 `documents / chunks`
2. 用 BGE-M3 做粗检索，召回候选文档
3. 从 chunk 构图，构造 `Chunk / Proposition / Entity / Type` 异构图
4. 用意图识别控制粗检索和图扩展方式
5. 用语义锚点 + 结构先验对 proposition 重排
6. 聚合证据并调用 LLM 生成最终答案

---
## 4. 目录结构与每个文件的作用

### 4.1 顶层目录

#### `config.yaml`
项目总配置文件。定义粗检索、构图、意图识别、语义锚点、答案生成、结构重排等模块参数。

#### `requirements.txt`
Python 依赖清单。现在已经补充了源码真实会用到的关键依赖，适合 `pip install -r requirements.txt`。

#### `environment.yml`
Conda 环境文件。适合在新机器上快速创建可运行环境。

#### `run_main_hotpot_chunks.py`
用途：项目统一主入口。

输入：
- `config.yaml`
- `data/hotpotqa/hotpot_chunks.json`
- `data/hotpotqa/hotpot_docs.json`
- 用户指定的 `--num-questions`
- 可选 `--query`，或 `--use-first-question-query`

输出：
- `results/run_main_hotpot_chunks_q{N}.json`

特点：
- 支持只用前 N 个问题对应的 chunk/doc 子集做实验
- 不需要手动改源码里的路径
- 兼容当前项目里 `embedding_model / model_name` 的字段差异

#### `eval_intent_coarse_chunk_only.py`
批量评测脚本。功能是：
- 用 `RAGSystem` 的前半段能力
- 只做 **intent prediction + coarse retrieval**
- 然后在 **chunk 级别** 评估召回情况
- 输出详细结果和汇总结果

#### `graph_build_experiment (1).py`
图构建调试/统计脚本。用于：
- 从 coarse retrieval 的结果里选 case
- 解析对应 chunk
- 调用 `GraphBuilder` 构图
- 输出每个 case 的构图统计、图对齐信息、聚合指标

#### `offline_struct`
空文件。当前没有明确实际用途，像是遗留文件。

#### `.idea/`
IDE 配置目录，供 PyCharm / IntelliJ 使用，不是项目运行所需。

#### `.venv/`
本地虚拟环境目录，不可移植，不应视为项目源代码组成部分。

---

### 4.2 `src/` 源代码目录

#### `src/__init__.py`
包初始化文件，定义项目版本号。

#### `src/bgem3_retriever.py`
BGE-M3 混合粗检索模块。

主要职责：
- 加载 `BGEM3FlagModel`
- 对文档建立 dense + sparse 索引
- 做混合召回
- 返回 top-k 文档对象

这个模块是整个系统的**粗检索入口**。

#### `src/graph_builder.py`
异构图构建核心模块。

主要职责：
- 从 chunk 文本中抽取 entities 和 facts
- 将 facts 聚合成 propositions
- 构建 `chunk / proposition / entity / type` 图
- 计算并写入离线结构先验 `s_struct_global`
- 维护 entity type cache

这个模块是整个系统最核心的结构化建图模块。

#### `src/hotpot_utils.py`
HotpotQA 数据格式转换工具。

主要职责：
- `sample_to_documents`：把单个 HotpotQA 样本转成 documents
- `documents_to_chunks`：把 documents 展开成 chunks
- `dataset_to_documents`：批量转 documents
- `dataset_to_chunks`：批量转 chunks

它负责把原始 QA 数据转成项目内部统一使用的文档/块格式。

#### `src/intent_representation.py`
意图识别模块。

主要职责：
- 接收 question
- 调用 LLM 输出 `topology` 与 `semantic`
- 把问题分类为：
  - `Sequential / Parallel`
  - `HUM / LOC / OTHER`

这个结果会直接控制粗检索和图扩展策略。

#### `src/semantic_anchor_selector.py`
语义锚点评分模块。

主要职责：
- 对查询子图中的 proposition 逐个打分
- 判断哪个 proposition 更适合作为 multi-hop QA 的“语义锚点”
- 输出 proposition → semantic score

#### `src/structural_centrality_ranker.py`
结构中心性排序模块。

主要职责：
- 读取图中离线写入的结构先验分数
- 可与语义分数融合
- 对 proposition / entity 做 query-time 排序

#### `src/offline_struct_prior.py`
离线结构先验计算模块。

主要职责：
- 仅在 `P-E` 子图上计算结构先验
- 结合 core number 和 betweenness centrality
- 写入节点属性，例如 `s_struct_global`

#### `src/answer_generator.py`
答案生成模块。

主要职责：
- 对 proposition / entity / chunk / type 证据桶做归一化、截断、去重
- 拼 prompt
- 调用 LLM 生成最终答案

#### `src/rag_system.py`
系统总控模块。

主要职责：
- 串联 coarse retrieval、graph build、intent、semantic anchor、centrality ranking、answer generation
- 对外提供统一的系统接口

这是整个项目的主 orchestrator。



#### `src/run_coarse_single.py`
粗检索单样例测试脚本。

主要用途：
- 读取文档与 QA 数据
- 构建 BGEM3Retriever 索引
- 随机取第一条 question 做粗检索测试
- 打印 top 文档标题和 chunk 预览

但当前内部路径写错，不能直接运行成功。

#### `src/run_hotpot_preprocess.py`
HotpotQA 预处理脚本。

主要用途：
- 将原始 HotpotQA 数据转成 `documents` 和 `chunks`
- 输出到 JSON 文件

---

### 4.3 `data/` 数据目录


#### `data/hotpotqa/hotpotqa.json`
HotpotQA 样本数据。当前压缩包中是一个列表，包含约 1000 条样本。

#### `data/hotpotqa/hotpot_docs.json`
由 HotpotQA 转换得到的 document 级语料。

#### `data/hotpotqa/hotpot_chunks.json`
由 HotpotQA 转换得到的 chunk 级语料，是构图和评测的重要输入。

#### `data/hotpotqa/hotpotqa_corpus.json`
HotpotQA 相关的语料文件，供后续检索或实验使用。

#### `data/hotpotqa/hotpot_dev_distractor_v1.json`
HotpotQA 的原始/标准 dev distractor 数据文件。

#### `data/musique/musique_full_v1.0_test.jsonl`
MuSiQue 测试集文件。

---

## 5. 推荐启动顺序



### 第一步：创建虚拟环境

Windows：

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 第二步：安装依赖

先安装已有 requirements：

```bash
pip install -r requirements.txt
```


### 第三步：准备 Ollama

需要本机启动 Ollama，并准备至少两个模型：

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3:latest
```

然后确保服务可用，默认地址应与 `config.yaml` 一致：

```bash
http://localhost:11434
```

### 第四步：准备数据

当前项目压缩包里已经有以下数据：

- `data/hotpotqa/hotpotqa.json`
- `data/hotpotqa/hotpot_docs.json`
- `data/hotpotqa/hotpot_chunks.json`


### 第五步：运行 coarse retrieval 评测

推荐使用：

```bash
python eval_intent_coarse_chunk_only.py \
  --config config.yaml \
  --qa_path data/hotpotqa/hotpotqa.json \
  --docs_path data/hotpotqa/hotpot_docs.json \
  --chunks_path data/hotpotqa/hotpot_chunks.json \
  --index_prefix data/hotpotqa/hotpot_bgem3_index \
  --output_dir outputs/intent_coarse_chunk_eval
```

这个脚本会做：
1. 加载配置和数据
2. 初始化 `RAGSystem`
3. 构建或加载 BGEM3 索引
4. 对每个样本做 `intent + coarse retrieval`
5. 统计 chunk-level recall
6. 输出详细结果和 summary

### 第六步：运行图构建实验

推荐这样传参运行：

```bash
python "graph_build_experiment (1).py" \
  --repo_root . \
  --config config.yaml \
  --chunks data/hotpotqa/hotpot_chunks.json \
  --coarse outputs/intent_coarse_chunk_eval/intent_coarse_chunk_details.json \
  --mode stats \
  --output_dir outputs/graph_build_experiment
```

如果你想看单 case 的详细构图过程，可以把 `--mode` 改为：

```bash
--mode debug
```

### 第七步：单样例粗检索测试

`src/run_coarse_single.py` 的用途是单测粗检索。

但是它当前默认路径不对，所以按现状需要你先改里面的路径，或者自行重写一个更通用的单测脚本再运行。

---



