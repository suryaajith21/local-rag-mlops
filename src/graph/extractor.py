"""LLM-powered named entity and relationship extraction from chunk text.

Extracts entities of types MODEL, TECHNIQUE, DATASET, CONCEPT, ORGANIZATION
and the relationships between them. Results feed the knowledge graph built
during ingestion.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import CONFIG

_EXTRACT_PROMPT = """\
Extract named entities and relationships from the text below.

Entity types to extract (ignore everything else):
- MODEL: neural network models (e.g. BERT, GPT-4, LLaMA, T5, ResNet)
- TECHNIQUE: methods, algorithms, training procedures (e.g. fine-tuning, RLHF, \
RAG, attention, backpropagation, dropout)
- DATASET: benchmarks and datasets (e.g. MMLU, HumanEval, ImageNet, CIFAR-10)
- CONCEPT: abstract ideas (e.g. hallucination, alignment, tokenization, \
overfitting, mode collapse)
- ORGANIZATION: companies, labs, institutions (e.g. OpenAI, Google DeepMind, \
Anthropic, Meta AI)

Rules:
- Extract at most {max_entities} entities total.
- Normalize names: title case for MODEL and ORGANIZATION \
(e.g. "Bert" → "BERT", "openai" → "OpenAI"), lowercase for TECHNIQUE and \
CONCEPT (e.g. "Fine-Tuning" → "fine-tuning").
- Only extract relationships between entities you already listed.
- If the text contains no clear entities, return empty lists.

Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation:
{{
  "entities": [
    {{"name": "BERT", "type": "MODEL"}},
    {{"name": "fine-tuning", "type": "TECHNIQUE"}}
  ],
  "relationships": [
    {{"source": "BERT", "relation": "trained with", "target": "fine-tuning"}}
  ]
}}

Text:
{text}"""


class EntityExtractor:
    """Extracts named entities and relationships from text chunks using an LLM.

    Calls the Ollama LLM once per chunk with a structured prompt that enforces
    a fixed entity taxonomy. JSON parse failures return empty results so a
    single bad LLM response never blocks ingestion.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=CONFIG.ollama_model,
            temperature=0.0,
            base_url=CONFIG.ollama_base_url,
        )

    def extract(self, chunk: dict) -> dict:
        """Extract entities and relationships from a single chunk.

        Args:
            chunk: A chunk dict with ``"content"`` (str) and ``"metadata"``
                (dict) keys, as produced by the ingestion pipeline.

        Returns:
            A dict with keys:
                - ``"entities"`` (list[dict]): Each entry has ``"name"`` and
                  ``"type"`` keys.
                - ``"relationships"`` (list[dict]): Each entry has
                  ``"source"``, ``"relation"``, and ``"target"`` keys.
                - ``"chunk_content"`` (str): The chunk's text content.
                - ``"chunk_metadata"`` (dict): The chunk's metadata.

            Returns empty entity/relationship lists on any parse failure
            so ingestion is never blocked by a single bad LLM response.
        """
        empty = {
            "entities": [],
            "relationships": [],
            "chunk_content": chunk.get("content", ""),
            "chunk_metadata": chunk.get("metadata", {}),
        }

        text = chunk.get("content", "").strip()
        if not text:
            return empty

        try:
            prompt = _EXTRACT_PROMPT.format(
                max_entities=CONFIG.graph_max_entities_per_chunk,
                text=text,
            )
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Strip markdown fences if the model disobeys
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(l for l in lines if not l.startswith("```")).strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return empty

            data = json.loads(raw[start:end])

            entities = [
                e for e in data.get("entities", [])
                if isinstance(e, dict)
                and e.get("name")
                and e.get("type") in CONFIG.graph_entity_types
            ]
            relationships = [
                r for r in data.get("relationships", [])
                if isinstance(r, dict)
                and r.get("source")
                and r.get("relation")
                and r.get("target")
            ]

            return {
                "entities": entities[: CONFIG.graph_max_entities_per_chunk],
                "relationships": relationships,
                "chunk_content": chunk.get("content", ""),
                "chunk_metadata": chunk.get("metadata", {}),
            }

        except Exception:
            return empty
