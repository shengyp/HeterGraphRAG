


import logging
import requests

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """答案生成器 - 基于多粒度证据选择生成答案"""

    def __init__(self, config):
        cfg = config or {}
        self.config = cfg

        self.k_c = int(cfg.get("k_c", 5))
        self.k_p = int(cfg.get("k_p", 3))
        self.k_z = int(cfg.get("k_z", 2))
        self.k_s = int(cfg.get("k_s", 2))

        self.llm_model = cfg.get("llm_model", "qwen2.5:7b")
        self.ollama_base_url = cfg.get("ollama_base_url", "http://localhost:11434")

        logger.info("AnswerGenerator initialized")

    def select_evidence_nodes(self, ranked_nodes_by_type):
   
        buckets = ranked_nodes_by_type or {}

        evidence = {
            "chunk": [],
            "prop": [],
            "concept": [],
            "summary": [],
        }

        # chunk
        for item in (buckets.get("chunk") or [])[: self.k_c]:
            evidence["chunk"].append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )

        # 
        for item in (buckets.get("proposition") or [])[: self.k_p]:
            evidence["prop"].append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )

        # concept
        for item in (buckets.get("concept") or [])[: self.k_z]:
            evidence["concept"].append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )

        # summary
        for item in (buckets.get("summary") or [])[: self.k_s]:
            evidence["summary"].append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "score": float(item.get("score", 0.0)),
                }
            )

        logger.info(
            "证据选择完成: C=%d, P=%d, Z=%d, S=%d",
            len(evidence["chunk"]),
            len(evidence["prop"]),
            len(evidence["concept"]),
            len(evidence["summary"]),
        )
        return evidence

    def build_rag_prompt(self, query, evidence):
        """
        构造结构化 RAG Prompt
        """
        prompt_parts = []

        # -----------------------------
        # 1. ROLE
        # -----------------------------
        prompt_parts.append("# Role")
        prompt_parts.append(
            "You are an evidence-grounded question-answering assistant. "
            "Your answers must be derived solely from the provided evidence. "
            "Your goal is to give a precise, minimal answer that directly satisfies the question."
        )

        # -----------------------------
        # 2. TASK
        # -----------------------------
        prompt_parts.append("\n# Task")
        prompt_parts.append(
            "Carefully read the question and the evidence. Your job is not to repeat the evidence, "
            "but to determine exactly what information the question seeks. "
            "To do this reliably, always:\n"
            "1) Identify what entity or subject the question is ultimately referring to, "
            "   even if the question refers to it indirectly or through a descriptive phrase.\n"
            "2) Identify the type of information being requested about that subject "
            "   (such as a name, description, date, role, or other specific attribute).\n"
            "3) Find the information in the evidence that directly provides that attribute.\n"
            "Provide only the requested information, nothing more."
        )

        # -----------------------------
        # 3. QUESTION
        # -----------------------------
        prompt_parts.append("\n# Question")
        prompt_parts.append(query)

        # -----------------------------
        # 4. EVIDENCE BLOCKS
        # -----------------------------
        # Key Facts / Propositions
        if evidence["prop"]:
            prompt_parts.append("\n# Key Facts")
            for i, prop in enumerate(evidence["prop"], 1):
                text = prop.get("text", "")[:500]
                prompt_parts.append(f"{i}. {text}")

        # Summaries
        if evidence["summary"]:
            prompt_parts.append("\n# Summaries")
            for i, summary in enumerate(evidence["summary"], 1):
                text = summary.get("text", "")[:400]
                prompt_parts.append(f"{i}. {text}")

        # Concepts
        if evidence["concept"]:
            prompt_parts.append("\n# Related Information")
            for i, concept in enumerate(evidence["concept"], 1):
                text = concept.get("text", "")[:300]
                prompt_parts.append(f"{i}. {text}")

        # Chunks
        if evidence["chunk"]:
            prompt_parts.append("\n# Detailed Evidence")
            for i, chunk in enumerate(evidence["chunk"], 1):
                text = chunk.get("text", "")[:500]
                prompt_parts.append(f"[{i}] {text}")

        # -----------------------------
        # 5. REASONING GUIDELINES
        # -----------------------------
        prompt_parts.append("\n# Reasoning Guidelines")
        prompt_parts.append(
            "- Interpret the question precisely before looking for an answer.\n"
            "- Determine what entity the question is ultimately referring to, even when that entity is "
            "mentioned indirectly or through a descriptive phrase.\n"
            "- Then determine the exact aspect or attribute of that entity that the question is asking for.\n"
            "- Prefer evidence that directly supports the requested attribute of the correct subject.\n"
            "- Avoid answers that merely restate evidence or mention related entities not asked about.\n"
            "- Base your answer strictly on the provided evidence, not on external knowledge."
        )

        # -----------------------------
        # 6. ANSWER REQUIREMENTS
        # -----------------------------
        prompt_parts.append("\n# Answer Requirements")
        prompt_parts.append(
            "- Provide only the specific information that answers the question.\n"
            "- Keep the answer brief (usually a few words).\n"
            "- Do NOT restate the question.\n"
            "- Do NOT provide explanations.\n"
            "- Do NOT give multiple answers—only one.\n"
            "- Ensure your answer is fully supported by the evidence."
        )

        # -----------------------------
        # 7. FINAL ANSWER OUTPUT
        # -----------------------------
        prompt_parts.append("\n# Final Answer:")

        return "\n".join(prompt_parts)

    def generate_answer_with_rag(self, prompt):
        """
        调用LLM生成答案
        """
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": float(self.config.get("temperature", 0.3)),
                    "num_predict": int(self.config.get("num_predict", 200)),
                },
            },
            timeout=int(self.config.get("timeout", 60)),
        )
        resp.raise_for_status()
        return (resp.json().get("response") or "").strip()

    def generate_answer(self, query, ranked_nodes_by_type, graph):
   
        logger.info("开始生成答案: query='%s...'", (query or "")[:50])

        evidence = self.select_evidence_nodes(ranked_nodes_by_type)
        prompt = self.build_rag_prompt(query, evidence)
        answer = self.generate_answer_with_rag(prompt)

        logger.info("答案生成完成")
        return {"answer": answer, "supporting_nodes": evidence}
