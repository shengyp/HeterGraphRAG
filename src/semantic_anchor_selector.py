"""
LLM语义锚点选择模块

打分：
s_total = s_llm

"""

import logging
import re
import requests

logger = logging.getLogger(__name__)


class SemanticAnchorSelector:


    def __init__(self, config):
        cfg = config or {}
        self.config = cfg
        self.ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")
        self.llm_model = cfg.get("llm_model", "qwen2.5:14b")


        self.tau_sem = float(cfg.get("tau_sem", 0.25))
        self.top_k_fallback = int(cfg.get("top_k_fallback", 5))

        self.llm_score_timeout = int(cfg.get("llm_score_timeout", 20))

        logger.info(
            "SemanticAnchorSelector initialized: llm_model=%s, llm_score_timeout=%ds",
            self.llm_model,
            self.llm_score_timeout,
        )

    def llm_score_node_as_anchor(self, query, intent_type, node_type, node_text):
      
        node_text = (node_text or "")[:450]

        prompt = f"""# Role
You are a strict relevance rater for retrieval-augmented question answering.

# Task
Given a Query and one Node, output ONE float in [0.0, 1.0] indicating how useful this node would be for answering the query.
This is a CONTINUOUS score (any value like 0.13, 0.67, 0.92 is valid), not a small set of fixed choices.
Score meaning: "Would I directly cite or strongly rely on this node when answering?"

# Output Format (STRICT)
- Output ONLY a single number (e.g., 0.73)
- No extra words, no JSON, no punctuation.
- Keep 2 decimal places when possible.

# Scoring Rubric (continuous ranges + examples)

## [0.90, 1.00] Direct answer / decisive evidence
Example A
Query: "What port does SkyWalking OAP use by default?"
Node: "The OAP server listens on 12800 (HTTP) and 11800 (gRPC)."
Output: 0.96

Example B
Query: "Who developed the theory of relativity?"
Node: "Albert Einstein developed the theory of relativity in 1905."
Output: 0.97

## [0.70, 0.89] Strongly supportive, key details, close to answering
Example A
Query: "What port does SkyWalking OAP use by default?"
Node: "Configuration examples often use http://127.0.0.1:12800 as the OAP endpoint."
Output: 0.82

Example B
Query: "Why did the service time out?"
Node: "Logs show timeouts happen under high concurrency due to downstream latency spikes."
Output: 0.79

## [0.40, 0.69] Useful context, partially relevant, but not sufficient alone
Example A
Query: "What port does SkyWalking OAP use by default?"
Node: "OAP is the backend that receives traces and metrics from agents."
Output: 0.55

Example B
Query: "What is circuit breaker?"
Node: "A circuit breaker is a resilience pattern often used with fallbacks and timeouts."
Output: 0.62

## [0.10, 0.39] Weakly related, tangential, unlikely to be cited
Example A
Query: "What port does SkyWalking OAP use by default?"
Node: "SkyWalking UI is a web dashboard for observability."
Output: 0.22

Example B
Query: "Compare BFS and DFS."
Node: "Graph traversal is used in search problems."
Output: 0.30

## [0.00, 0.09] Not relevant
Example A
Query: "What port does SkyWalking OAP use by default?"
Node: "RabbitMQ is a message broker."
Output: 0.01

Example B
Query: "What is the time complexity of Dijkstra?"
Node: "Paris is the capital of France."
Output: 0.00

# Your Input
Query: {query}
Intent: {intent_type}
Node Type: {node_type}
Node Content:
{node_text}

Score (single number only):"""

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 8, "temperature": 0.0},
            },
            timeout=self.llm_score_timeout,
        )
        resp.raise_for_status()

        out = (resp.json().get("response") or "").strip()


        if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", out):
            raise ValueError(f"LLM score parse failed. raw='{out[:120]}'")

        v = float(out)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"LLM score out of range: {v}")

        return v

    def select_semantic_anchors(self, graph, query, h_intent, intent_type):
     
        n = graph.number_of_nodes()
        logger.info("开始全量LLM锚点评分：nodes=%d intent_type=%s", n, intent_type)

        s_scores = {}
        for i, (node_id, node_data) in enumerate(graph.nodes(data=True), start=1):
            if i % 10 == 0:
                logger.info("评分进度：%d/%d", i, n)

            node_type = node_data.get("node_type", "unknown")
            node_text = node_data.get("text", "")

            s_llm = self.llm_score_node_as_anchor(query, intent_type, node_type, node_text)
            s_scores[node_id] = float(s_llm)

            if i <= 5:
                logger.info("node=%s type=%s llm=%.3f", str(node_id)[:30], node_type, s_llm)

        logger.info("LLM打分完成：scored_nodes=%d", len(s_scores))
        return s_scores
