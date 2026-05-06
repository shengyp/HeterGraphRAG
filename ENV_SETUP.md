# Environment Setup

This project uses:

- `FlagEmbedding` for BGE-M3 dense+sparse retrieval.
- `Ollama` for LLM-based intent classification, graph construction, entity typing, semantic anchor scoring, and answer generation.
- `networkx`, `numpy`, and `requests` for graph/ranking/runtime utilities.

## 1. Create Python Environment

Recommended Python version: `3.10` or `3.11`.

```bash
conda create -n ksem-rag python=3.10 -y
conda activate ksem-rag
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If your machine has CUDA, install the matching PyTorch build first or reinstall it after `requirements.txt`.
Example for CUDA 12.1:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

CPU-only fallback:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

## 2. Verify Python Packages

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

## 3. Prepare Ollama

Install and start Ollama according to your platform:

```bash
ollama serve
```

In another shell, pull the models used by the default experiment config:

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3:latest
```

Verify Ollama:

```bash
curl http://localhost:11434/api/tags
```

Verify generation:

```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:14b","prompt":"Return JSON only: {\"ok\":true}","stream":false}'
```

Verify embedding:

```bash
curl http://localhost:11434/api/embed \
  -d '{"model":"bge-m3:latest","input":"test embedding"}'
```

The graph builder currently expects `embedding_dim = 1024` for `bge-m3:latest`.

## 4. Run Experiments From Small To Large

All commands below assume you are in:

```bash
cd /home/featurize/KSEM/src
```

Probe data:

```bash
python experiments/01_probe_data.py
```

Build a 5-QA retrieval index:

```bash
python experiments/02_build_index.py \
  --num-qa 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index
```

Evaluate coarse retrieval:

```bash
python experiments/03_eval_retrieval.py \
  --num-queries 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index
```

Build a 5-QA graph:

```bash
python experiments/04_build_graph_subset.py \
  --num-qa 5 \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

Run one end-to-end query:

```bash
python experiments/05_query_one.py \
  --qa-index 0 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

Run a small batch:

```bash
python experiments/06_run_batch.py \
  --num-queries 5 \
  --index-prefix experiments/artifacts/hotpot_subset_bgem3_index \
  --graph-path experiments/artifacts/hotpot_subset_graph.pkl
```

## 5. Scaling Plan

After the 5-QA run is healthy:

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

Only move to 1000 QA after strict-methodology metadata is healthy.

## 6. Methodology Health Fields

For paper experiments, inspect these metadata fields:

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

Expected healthy strict run:

- `legacy_intent_schema_used = False`
- `followup_used_firsthop_evidence = True`
- `entity_type_filter_matched_count > 0`
- `entity_type_filter_fallback_used = False`
- `type_cache_missing_entity_count = 0`

## 7. Common Failures

`ModuleNotFoundError: No module named 'FlagEmbedding'`

```bash
pip install -U FlagEmbedding
```

Ollama connection refused:

```bash
ollama serve
```

Missing Ollama model:

```bash
ollama pull qwen2.5:14b
ollama pull bge-m3:latest
```

Strict graph failure because old type labels are missing:

Ensure `entity_type_labels` includes:

```text
PERSON, LOCATION, ORGANIZATION, WORK, EVENT, TIME, NUMBER, CONCEPT, OTHER
```

Strict query failure because `type_cache_missing_entity_count > 0`:

You are likely loading an old graph. Rebuild the graph with `04_build_graph_subset.py`.
