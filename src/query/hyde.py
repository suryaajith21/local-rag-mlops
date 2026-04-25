"""HyDE — Hypothetical Document Embeddings.

Instead of embedding the raw user query for dense retrieval, ask the LLM to
generate a short hypothetical passage that would answer the question, then
embed that passage. The hypothetical uses domain vocabulary and sentence
structure that is much closer to real document chunks, substantially improving
cosine similarity matching.

HyDE is applied only to dense (vector) retrieval. BM25 uses the original
query because the hypothetical text introduces vocabulary not in the index and
degrades keyword recall.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.config import CONFIG

_HYDE_PROMPT = """\
Write a short passage of 3-5 sentences that directly answers the following \
question. Write it as if it is a paragraph from an academic paper or technical \
document — factual, precise, third-person. Do not begin with "I", "The answer \
is", or any preamble. Do not use bullet points. Just write the passage.

Question: {query}

Passage:"""


class HyDEExpander:
    """Generates a hypothetical document passage for improved dense retrieval.

    The LLM is prompted to write a short, academic-style passage that would
    directly answer the user's query. That passage is then used in place of
    the raw query for dense embedding lookup.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=CONFIG.ollama_model,
            temperature=0.5,
            base_url=CONFIG.ollama_base_url,
        )

    def expand(self, query: str) -> str:
        """Generate a hypothetical passage that would answer the query.

        The generated passage uses the same vocabulary and sentence structure
        as real document chunks, which improves cosine similarity matching
        during dense retrieval.

        Args:
            query: The user's natural-language question.

        Returns:
            A 3-5 sentence hypothetical passage written in academic style.
            Falls back to the original query string if the LLM call fails,
            so retrieval continues without HyDE rather than crashing.
        """
        try:
            prompt = _HYDE_PROMPT.format(query=query)
            response = self._llm.invoke([HumanMessage(content=prompt)])
            passage = response.content.strip()
            # Strip any accidental leading labels the model might add
            for prefix in ("Passage:", "Answer:", "Response:"):
                if passage.startswith(prefix):
                    passage = passage[len(prefix):].strip()
            return passage if passage else query
        except Exception:
            return query
