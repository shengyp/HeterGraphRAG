# 环境配置

本项目使用：

- `FlagEmbedding`：用于 BGE-M3 稠密+稀疏混合检索。
- `Ollama`：用于基于 LLM 的意图分类、图构建、实体类型标注、语义锚点评分和答案生成。
- `networkx`、`numpy`、`requests`：用于图处理、排序和运行时工具。

## 1. 创建 Python 环境

推荐 Python 版本：`3.10` 或 `3.11`。

```bash
conda create -n ksem-rag python=3.10 -y
conda activate ksem-rag
python -m pip install --upgrade pip setuptools wheel
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你的机器有 CUDA，建议先安装匹配的 PyTorch 版本，或者在安装 `requirements.txt` 后重新安装 PyTorch。
CUDA 12.1 示例：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

仅 CPU 环境：

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

## 2. 验证 Python 包

```bash
python - <<'PY'
import numpy
import networkx
import requests
from FlagEmbedding import BGEM3FlagModel
print("numpy", numpy.__version__)
print("networkx", networkx.__version__)
print("FlagEmbedding OK")
PY
```

## 3. 准备 Ollama

根据你的平台安装并启动 Ollama：

```bash
ollama serve
```

在另一个终端中拉取默认实验配置使用的模型：

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3:latest
```

验证 Ollama 服务：

```bash
curl http://localhost:11434/api/tags
```

验证生成接口：

```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"Return JSON only: {\"ok\":true}","stream":false}'
```

验证 embedding 接口：

```bash
curl http://localhost:11434/api/embed \
  -d '{"model":"bge-m3:latest","input":"test embedding"}'
```

当前图构建模块要求 `bge-m3:latest` 的 `embedding_dim = 1024`。

## 4. 按从小到大的顺序运行实验

下面的命令默认你已经进入项目根目录：

```bash
cd /home/featurize/KSEM/src
```

检查数据：

```bash
python experiments/01_probe_data.py
```

构建 5 个 QA 的检索索引：

```bash
python experiments/02_build_index.py \
  --num-qa 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index
```

评估粗检索：

```bash
python experiments/03_eval_retrieval.py \
  --num-queries 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index
```

构建 5 个 QA 的图：

```bash
python experiments/04_build_graph_subset.py \
  --num-qa 5 \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

运行一次端到端查询：

```bash
python experiments/05_query_one.py \
  --qa-index 0 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

运行一个小批量实验：

```bash
python experiments/06_run_batch.py \
  --num-queries 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

## 5. 扩展实验规模

5 个 QA 的流程正常后，再逐步扩大规模：

```bash
# 10 QA
python experiments/02_build_index.py --num-qa 10 --index-prefix experiments/artifacts/hotpot_10_bgem3_index
python experiments/04_build_graph_subset.py --num-qa 10 --graph-path experiments/artifacts/hotpot_10_graph.pkl
python experiments/06_run_batch.py --num-queries 10 --index-prefix experiments/artifacts/hotpot_10_bgem3_index --graph-path experiments/artifacts/hotpot_10_graph.pkl

# 50 QA
python experiments/02_build_index.py --num-qa 50 --index-prefix experiments/artifacts/hotpot_50_bgem3_index
python experiments/04_build_graph_subset.py --num-qa 50 --graph-path experiments/artifacts/hotpot_50_graph.pkl
python experiments/06_run_batch.py --num-queries 50 --index-prefix experiments/artifacts/hotpot_50_bgem3_index --graph-path experiments/artifacts/hotpot_50_graph.pkl
```

只有在严格方法论元数据健康时，才建议继续扩展到 1000 QA。

## 6. 方法论健康字段

论文实验中建议检查这些元数据字段：

- `strict_methodology`
- `entity_type_filter_candidate_count`
- `entity_type_filter_matched_count`
- `entity_type_filter_fallback_used`
- `followup_used_firsthop_evidence`
- `followup_firsthop_snippet_count`
- `legacy_intent_schema_used`
- `intent_schema`
- `type_cache_hit`
- `type_cache_label`
- `type_cache_observed_entity_count`
- `type_cache_missing_entity_count`

严格模式下的健康状态通常应满足：

- `legacy_intent_schema_used = False`
- `followup_used_firsthop_evidence = True`
- `entity_type_filter_matched_count > 0`
- `entity_type_filter_fallback_used = False`
- `type_cache_missing_entity_count = 0`

## 7. 常见问题

`ModuleNotFoundError: No module named 'FlagEmbedding'`

```bash
pip install -U FlagEmbedding
```

Ollama 连接被拒绝：

```bash
ollama serve
```

缺少 Ollama 模型：

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3:latest
```

旧的实体类型标签缺失导致严格图构建失败：

确保 `entity_type_labels` 包含：

```text
PERSON, LOCATION, ORGANIZATION, WORK, EVENT, TIME, NUMBER, CONCEPT, OTHER
```

`type_cache_missing_entity_count > 0` 导致严格查询失败：

你很可能加载了旧图。请用 `04_build_graph_subset.py` 重新构建图。
