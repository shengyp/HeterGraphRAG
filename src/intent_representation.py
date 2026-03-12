

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class IntentRepresentation:
    TOPO_SPACE = ["Sequential", "Parallel"]
    SEM_SPACE = ["HUM", "LOC", "OTHER"]
    TOPO_SET = set(TOPO_SPACE)
    SEM_SET = set(SEM_SPACE)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config
        self.config = cfg
        self.ollama_base_url = str(cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.llm_model = str(cfg.get("llm_model", "qwen2.5:14b"))
        self.timeout = int(cfg.get("timeout", 12))
        #最多取多少个 seed chunk 片段放进 prompt。
        self.max_ctx_snippets = int(cfg.get("max_ctx_snippets", 4))
        self.max_snippet_len = int(cfg.get("max_snippet_len", 260))

        logger.info("IntentRepresentation init | model=%s", self.llm_model)

    @staticmethod
    def _extract_first_json_object(raw: str) -> str:
        if not raw:
            raise ValueError("LLM output empty")
        s = raw.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)

        start = s.find("{")
        if start < 0:
            raise ValueError("No '{' found in LLM output")

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
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        raise ValueError("JSON braces not balanced")

    def predict_intent(
            self,
            query: str,
            seed_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], str, str]:

        query = query.strip()
        if not query:
            raise ValueError("query 为空")

        seed_chunks = seed_chunks or []

        snippets: List[str] = []
        for c in seed_chunks[: self.max_ctx_snippets]:
            t = (c.get("text") or "").strip().replace("\n", " ")
            if t:
                snippets.append(t[: self.max_snippet_len])

        ctx_part = ""
        if snippets:
            ctx_block = "\n".join([f"- {s}" for s in snippets])
            ctx_part = f"\n# Optional context (seed chunks; may be noisy)\n{ctx_block}\n"

        prompt = f"""# Role
        You are an intent classifier for a HotpotQA-style multi-hop question answering system.

        # Task
        Given the [Question] and optional [Seed Chunks], predict the retrieval intent and output a STRICT JSON object.
        Output JSON only. Do NOT add explanations. Do NOT wrap in Markdown fences.

        You must output two fields:
        1) topology: Sequential | Parallel
        2) semantic: HUM | LOC | OTHER

        # Definitions

        ## topology
        - Sequential:
          The question needs step-by-step multi-hop reasoning, usually by finding an intermediate entity/fact first, then reaching the final answer.
          Typical form: A -> B -> Answer

        - Parallel:
          The question needs two or more branches to be retrieved in parallel and then aligned/compared/merged/intersected.
          Typical form: Branch1 + Branch2 -> merge/compare -> Answer

        ## semantic
        - HUM:
          The question is ultimately centered on a person (identity, attributes, roles, biography, actions).
        - LOC:
          The question is ultimately centered on a location (birthplace, place of occurrence, city/country/region).
        - OTHER:
          Anything else (works, organizations, events, years, numbers, objects, concepts).

        # HotpotQA-style Examples

        Example 1
        Question: "what is one of the stars of The Newcomers known for"
        Seed Chunks:
        - "The Newcomers ... starring ... Chris Evans."
        - "Evans is known for his superhero roles ..."
        Output: {{"topology":"Sequential","semantic":"HUM"}}

        Example 2
        Question: "Where was the actor who played Character X born?"
        Seed Chunks:
        - "Character X was played by Person A."
        - "Person A was born in City B."
        Output: {{"topology":"Sequential","semantic":"LOC"}}

        Example 3
        Question: "Which two films were released in 2010 and directed by the same person?"
        Seed Chunks:
        - "Film A ... directed by Director D ... released in 2010."
        - "Film B ... directed by Director D ... released in 2010."
        Output: {{"topology":"Parallel","semantic":"OTHER"}}

        # Input
        Question: {query}
        {ctx_part}
        # Output (STRICT JSON only)
        """

        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 120},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

        raw = (resp.json() or {}).get("response").strip()
        js = self._extract_first_json_object(raw)
        data = json.loads(js)

        topo = str(data.get("topology")).strip()
        sem = str(data.get("semantic")).strip().upper()

        if topo not in self.TOPO_SET:
            raise ValueError(f"Invalid topology='{topo}'. Must be one of: {self.TOPO_SPACE}")
        if sem not in self.SEM_SET:
            raise ValueError(f"Invalid semantic='{sem}'. Must be one of: {self.SEM_SPACE}")

        intent = {"topology": topo, "semantic": sem}
        logger.info("Intent predicted: topo=%s sem=%s", topo, sem)
        return intent, topo, sem
