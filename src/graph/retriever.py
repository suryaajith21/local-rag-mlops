"""Graph retriever — entity extraction from query + knowledge graph lookup.

Extracts entity names from a user query using the same LLM-based extractor
used at ingest time, then looks up those entities in the graph to find
associated chunk content. This surfaces chunks that are guaranteed to
mention the queried entity, complementing semantic similarity retrieval.
"""

from __future__ import annotations

from src.graph.engine import GraphEngine
from src.graph.extractor import EntityExtractor


class GraphRetriever:
    """Retrieves chunks from the knowledge graph via entity matching.

    At query time, entity names are extracted from the user's question and
    matched (case-insensitively) against graph nodes. The chunks stored on
    those nodes are returned as additional retrieval context.

    Args:
        engine: A loaded ``GraphEngine`` instance.
    """

    def __init__(self, engine: GraphEngine) -> None:
        self._engine = engine
        self._extractor = EntityExtractor()

    def retrieve(self, query: str) -> list[dict]:
        """Extract entities from the query and return their associated chunks.

        Treats the query string as a mini-chunk for entity extraction. Only
        entity *names* are used for the graph lookup; relationships extracted
        from the query are ignored.

        Args:
            query: The user's natural-language question.

        Returns:
            List of chunk dicts from the graph (may be empty if no matching
            entities are found or the graph is empty). Each dict has
            ``"content"``, ``"metadata"``, ``"rerank_score"``,
            ``"rrf_score"``, and ``"source": "graph"`` fields.
        """
        extraction = self._extractor.extract({"content": query, "metadata": {}})
        entity_names = [e["name"] for e in extraction.get("entities", [])]

        if not entity_names:
            return []

        return self._engine.get_entity_chunks(entity_names)
