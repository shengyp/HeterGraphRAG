

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class IntentRepresentation:
    REASONING_SPACE = ["BRIDGE", "COMPARISON", "DIRECT"]
    ANSWER_TYPE_SPACE = [
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
        "TIME",
        "NUMBER",
        "WORK",
        "EVENT",
        "CONCEPT",
        "BOOLEAN",
        "DESCRIPTION",
        "OTHER",
    ]
    REASONING_SET = set(REASONING_SPACE)
    ANSWER_TYPE_SET = set(ANSWER_TYPE_SPACE)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.config = cfg
        self.ollama_base_url = str(cfg.get("ollama_base_url", "http://localhost:11434")).rstrip("/")
        self.llm_model = str(cfg.get("llm_model", "qwen2.5:14b"))
        self.timeout = int(cfg.get("timeout", 90))
        self.strict_methodology = bool(cfg.get("strict_methodology", False))
        self.last_metadata: Dict[str, Any] = {}
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
        1) reasoning_type: BRIDGE | COMPARISON | DIRECT
        2) answer_type: PERSON | LOCATION | ORGANIZATION | TIME | NUMBER | WORK | EVENT | CONCEPT | BOOLEAN | DESCRIPTION | OTHER

        # Definitions

        ## reasoning_type
        - BRIDGE:
          The question needs step-by-step multi-hop reasoning by finding an intermediate entity/fact first, then reaching the final answer.
          Typical form: A -> B -> Answer

        - COMPARISON:
          The question needs two or more entities/branches to be retrieved in parallel and compared on the same attribute.
          Typical form: A.attribute vs B.attribute -> Answer

        - DIRECT:
          The answer can usually be obtained from one evidence path without an explicit bridge or comparison.

        ## answer_type
        This is the expected final answer type, not the type of every entity in the question.
        - PERSON: a person, human, named individual, or character.
        - LOCATION: a place, city, country, region, facility, or geographic target.
        - ORGANIZATION: a company, school, agency, team, party, or institution.
        - TIME: a date, year, period, or time expression.
        - NUMBER: a count, rank, age, distance, money amount, percentage, or other numeric value.
        - WORK: a book, film, paper, song, law text, software project, or other named work.
        - EVENT: a war, award, election, competition, conference, disaster, case, or named event.
        - CONCEPT: a concept, method, technology, disease, category, or abstract term.
        - BOOLEAN: yes/no or true/false answer.
        - DESCRIPTION: explanation, reason, process, difference, or descriptive answer.
        - OTHER: use only when none of the above is appropriate.

        # HotpotQA-style Examples

        Example 1
        Question: "what is one of the stars of The Newcomers known for"
        Seed Chunks:
        - "The Newcomers ... starring ... Chris Evans."
        - "Evans is known for his superhero roles ..."
        Output: {{"reasoning_type":"BRIDGE","answer_type":"DESCRIPTION"}}

        Example 2
        Question: "Where was the actor who played Character X born?"
        Seed Chunks:
        - "Character X was played by Person A."
        - "Person A was born in City B."
        Output: {{"reasoning_type":"BRIDGE","answer_type":"LOCATION"}}

        Example 3
        Question: "Which two films were released in 2010 and directed by the same person?"
        Seed Chunks:
        - "Film A ... directed by Director D ... released in 2010."
        - "Film B ... directed by Director D ... released in 2010."
        Output: {{"reasoning_type":"COMPARISON","answer_type":"WORK"}}

        Example 4
        Question: "Who founded SpaceX?"
        Output: {{"reasoning_type":"DIRECT","answer_type":"PERSON"}}

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
                "options": {"temperature": 0.0, "num_predict": 160},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

        raw = (resp.json() or {}).get("response").strip()
        js = self._extract_first_json_object(raw)
        data = json.loads(js)

        # Backward-compatible parsing while the prompt/configs are being migrated.
        has_new_schema = "reasoning_type" in data and "answer_type" in data
        has_legacy_schema = "topology" in data or "semantic" in data
        legacy_intent_schema_used = (not has_new_schema) and has_legacy_schema

        if self.strict_methodology and legacy_intent_schema_used:
            raise ValueError("strict_methodology=True but intent classifier returned legacy topology/semantic schema")

        reasoning_type = str(data.get("reasoning_type") or data.get("topology") or "").strip().upper()
        answer_type = str(data.get("answer_type") or data.get("semantic") or "").strip().upper()

        legacy_reasoning = {"SEQUENTIAL": "BRIDGE", "PARALLEL": "COMPARISON"}
        legacy_answer = {"HUM": "PERSON", "LOC": "LOCATION"}
        reasoning_type = legacy_reasoning.get(reasoning_type, reasoning_type)
        answer_type = legacy_answer.get(answer_type, answer_type)

        if reasoning_type not in self.REASONING_SET:
            raise ValueError(f"Invalid reasoning_type='{reasoning_type}'. Must be one of: {self.REASONING_SPACE}")
        if answer_type not in self.ANSWER_TYPE_SET:
            raise ValueError(f"Invalid answer_type='{answer_type}'. Must be one of: {self.ANSWER_TYPE_SPACE}")

        intent = {"reasoning_type": reasoning_type, "answer_type": answer_type}
        self.last_metadata = {
            "legacy_intent_schema_used": bool(legacy_intent_schema_used),
            "intent_schema": "legacy" if legacy_intent_schema_used else "current",
        }
        logger.info("Intent predicted: reasoning_type=%s answer_type=%s", reasoning_type, answer_type)
        return intent, reasoning_type, answer_type
