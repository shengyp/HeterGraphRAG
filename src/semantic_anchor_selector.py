

import logging
import re
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class SemanticAnchorSelector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config
        self.config = cfg
        self.ollama_base_url = str(cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.llm_model = str(cfg.get("llm_model", "qwen2.5:14b"))
        self.timeout = int(cfg.get("llm_score_timeout", 20))
        self.max_prop_len = int(cfg.get("max_node_text_len", 420))
        logger.info("SemanticAnchorSelector init | model=%s", self.llm_model)

    @staticmethod
    def _sem_hint(sem: str) -> str:
        sem = sem.strip().upper()
        if sem == "HUM":
            return "Hint: Prefer facts about people, actors, authors, or a specific person."
        if sem == "LOC":
            return "Hint: Prefer facts about locations, cities, countries, where something is located, or birthplace."
        return "Hint: No type bias."

    def _score_one(self, query: str, topo: str, sem: str, prop_text: str) -> float:
        prop_text = (prop_text or "")[: self.max_prop_len]
        hint = self._sem_hint(sem)

        prompt = f"""# Role
        You are a "semantic anchor scorer" in multi-hop question answering retrieval.

        # Task
        Given a question and a proposition, please evaluate the value of this proposition as a "semantic anchor" and output a score in the range [0,1].

        Here, "semantic anchor value" refers to whether the proposition can significantly help retrieval and reasoning in multi-hop question answering, including but not limited to:
        - directly supporting the final answer
        - providing a key intermediate entity
        - connecting adjacent steps in the reasoning chain
        - narrowing the subsequent search space
        - providing key attributes for parallel comparison or alignment

        A high score should be assigned not only to "direct answer propositions," but also to "key bridge propositions."

        {hint}

        # Topology Guidance
        - If the question topology is Sequential: prioritize propositions that can introduce the next-hop intermediate entity or connect earlier and later reasoning steps.
        - If the question topology is Parallel: prioritize propositions that can support comparison, alignment, intersection, or aggregation across multiple branches.

        # Semantic Guidance
        - HUM: prioritize information related to people, actors, authors, identities, and person attributes.
        - LOC: prioritize information related to places, cities, countries, geographic affiliation, birthplace, and where something is located.
        - OTHER: apply no additional type bias.

        # Scoring Criteria
        - 0.80 - 1.00: the proposition is very important for answering the question; it either directly supports the answer or is an indispensable bridge fact.
        - 0.50 - 0.79: the proposition is clearly relevant to the question and provides some help, but it is not a core bridge or direct evidence.
        - 0.20 - 0.49: the proposition provides only weakly relevant background information and has limited usefulness.
        - 0.00 - 0.19: the proposition is basically irrelevant to the question, or although it is topically related, it provides no substantial help to the reasoning chain.

        # Examples (HotpotQA style)
        Question: "what is one of the stars of The Newcomers known for"
        Proposition: "The Newcomers starred Chris Evans."
        Output: 0.88

        Question: "what is one of the stars of The Newcomers known for"
        Proposition: "Chris Evans is known for his superhero roles as the Marvel Comics character Captain America."
        Output: 0.95

        Question: "what is one of the stars of The Newcomers known for"
        Proposition: "The Newcomers was filmed in Vermont." 
        Output: 0.12

        # Input
        Question: {query}
        Topology: {topo}
        Semantic: {sem}
        Proposition: {prop_text}

        Score:"""

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 8, "temperature": 0.0},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

        out = (resp.json().get("response")).strip()
        if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", out):
            raise ValueError(f"LLM score parse failed. raw='{out[:120]}'")
        v = float(out)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"LLM score out of range: {v}")
        return v

    def select_semantic_anchors(self, graph, query: str, intent_topo: str, intent_sem: str) -> Dict[str, Any]:
        scored = 0
        s_scores: Dict[str, float] = {}

        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("node_type") != "proposition":
                continue
            prop_text = node_data.get("text")
            s_scores[node_id] = float(self._score_one(query, intent_topo, intent_sem, prop_text))
            scored += 1

        logger.info("Semantic scoring done: scored=%d", scored)
        return {"scores": s_scores, "metadata": {"scored": scored}}
