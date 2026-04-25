"""LLM answer synthesis with grounding and source citation.

Takes the final ranked chunks from the retrieval pipeline and generates a
concise, grounded answer. Instructs the LLM to cite sources in a parseable
format so callers can surface provenance to the end user.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import CONFIG

_GENERATE_PROMPT = """\
Answer the question using ONLY the provided context passages below.
If the context does not contain enough information to answer, say exactly:
"The document does not contain sufficient information to answer this question."
Do not hallucinate or add information not present in the passages.
Be concise — 3-5 sentences unless a list is clearly better.
After your answer, on a new line list the sources you used in this format:
Sources: [filename, page X], [filename, page Y]

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    """LLM answer synthesis with grounded context and source citation.

    Formats retrieved chunks as numbered passages, instructs the LLM to answer
    only from those passages and to list sources, then parses the citation line
    to extract provenance metadata.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=CONFIG.ollama_model,
            temperature=CONFIG.ollama_temperature,
            base_url=CONFIG.ollama_base_url,
        )

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate a grounded answer from retrieved chunks.

        Formats chunks as numbered passages with source attribution, sends them
        to the LLM with strict grounding instructions, and parses the response
        for inline source citations.

        Args:
            query: The user's original question.
            chunks: Retrieved chunk dicts, each with ``"content"`` and
                ``"metadata"`` (containing ``source`` and ``page`` keys).

        Returns:
            A dict with keys:
                - ``"answer"`` (str): Generated answer text, with the
                  "Sources:" line stripped out.
                - ``"sources"`` (list[str]): Unique source filenames cited.
                - ``"chunk_count"`` (int): Number of context passages used.
        """
        if not chunks:
            return {"answer": "No relevant context found.", "sources": [], "chunk_count": 0}

        # Build numbered context block
        passage_lines: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            src = meta.get("source", "unknown")
            page = meta.get("page", "?")
            passage_lines.append(f"[{i}] {chunk['content']} (Source: {src}, p.{page})")
        context = "\n\n".join(passage_lines)

        prompt = _GENERATE_PROMPT.format(context=context, question=query)

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            full_text = response.content.strip()
        except Exception:
            return {
                "answer": "Answer generation failed.",
                "sources": [],
                "chunk_count": len(chunks),
            }

        # Split answer body from sources line
        answer_body, sources = self._parse_response(full_text, chunks)

        return {
            "answer": answer_body,
            "sources": sources,
            "chunk_count": len(chunks),
        }

    @staticmethod
    def _parse_response(text: str, chunks: list[dict]) -> tuple[str, list[str]]:
        """Split the LLM response into answer body and source list.

        Args:
            text: Full LLM response string.
            chunks: Original chunks used, for metadata fallback.

        Returns:
            Tuple of (answer_body, unique_source_filenames).
        """
        # Locate "Sources:" line (case-insensitive, may be mid-text)
        sources_match = re.search(r"(?im)^sources?:\s*(.*)$", text)

        if sources_match:
            # Everything before the Sources line is the answer
            answer_body = text[: sources_match.start()].strip()
            sources_raw = sources_match.group(1)
            # Extract bracket groups: [filename, page X]
            bracket_items = re.findall(r"\[([^\]]+)\]", sources_raw)
            seen: set[str] = set()
            sources: list[str] = []
            for item in bracket_items:
                token = item.split(",")[0].strip()
                if not token:
                    continue
                if token.isdigit():
                    # LLM cited a passage index like [1] — map it to filename
                    idx = int(token) - 1  # convert 1-indexed to 0-indexed
                    if 0 <= idx < len(chunks):
                        filename = chunks[idx].get("metadata", {}).get("source", "")
                    else:
                        filename = ""
                else:
                    filename = token
                if filename and filename not in seen:
                    seen.add(filename)
                    sources.append(filename)
        else:
            answer_body = text
            sources = []

        # Fallback: collect unique source filenames from chunk metadata
        if not sources:
            seen = set()
            for chunk in chunks:
                src = chunk.get("metadata", {}).get("source", "")
                if src and src not in seen:
                    seen.add(src)
                    sources.append(src)

        return answer_body, sources
