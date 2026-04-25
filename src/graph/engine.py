"""NetworkX knowledge graph with pickle persistence.

Nodes represent named entities (MODEL, TECHNIQUE, etc.). Each node stores
the list of chunk content strings where the entity was mentioned. Edges
represent directed relationships between entities with a weight that
reinforces on repeated co-occurrence.

Graph retrieval works by resolving query entity names to nodes, then
returning the chunks stored on those nodes — giving the LLM context that
is guaranteed to mention the entity being asked about.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from src.config import CONFIG


class GraphEngine:
    """Knowledge graph built on NetworkX DiGraph with pickle persistence.

    Nodes carry:
        - ``type`` (str): Entity type from ``CONFIG.graph_entity_types``.
        - ``chunks`` (list[str]): Chunk content strings where this entity
          was mentioned.
        - ``chunk_metadatas`` (list[dict]): Metadata dicts parallel to
          ``chunks`` — index i of ``chunk_metadatas`` corresponds to index i
          of ``chunks``.
        - ``mention_count`` (int): Number of chunks that mention this entity.

    Edges carry:
        - ``relation`` (str): Relationship label (e.g. ``"uses"``).
        - ``weight`` (float): Starts at 1.0, incremented by 0.5 each time
          the same relationship appears in a new chunk.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._path: Path = CONFIG.graph_path

    def clear(self) -> None:
        """Reset to an empty directed graph."""
        self._graph = nx.DiGraph()

    def add_extraction(self, extraction: dict) -> None:
        """Integrate one chunk's extraction result into the graph.

        Creates new nodes and edges if they do not exist; reinforces existing
        ones by appending the chunk content and incrementing counts/weights.

        Args:
            extraction: Dict returned by ``EntityExtractor.extract()``, with
                keys ``"entities"``, ``"relationships"``, ``"chunk_content"``,
                and ``"chunk_metadata"``.
        """
        chunk_content: str = extraction.get("chunk_content", "")
        chunk_metadata: dict = extraction.get("chunk_metadata", {})
        entities: list[dict] = extraction.get("entities", [])
        relationships: list[dict] = extraction.get("relationships", [])

        for entity in entities:
            name: str = entity.get("name", "").strip()
            etype: str = entity.get("type", "CONCEPT")
            if not name:
                continue

            if self._graph.has_node(name):
                node = self._graph.nodes[name]
                if chunk_content and chunk_content not in node["chunks"]:
                    node["chunks"].append(chunk_content)
                    node["chunk_metadatas"].append(chunk_metadata)
                node["mention_count"] += 1
            else:
                self._graph.add_node(
                    name,
                    type=etype,
                    chunks=[chunk_content] if chunk_content else [],
                    chunk_metadatas=[chunk_metadata] if chunk_content else [],
                    mention_count=1,
                )

        for rel in relationships:
            src: str = rel.get("source", "").strip()
            tgt: str = rel.get("target", "").strip()
            relation: str = rel.get("relation", "related_to")
            if not src or not tgt:
                continue
            # Only add edges between nodes we actually have
            if not self._graph.has_node(src) or not self._graph.has_node(tgt):
                continue

            if self._graph.has_edge(src, tgt):
                self._graph[src][tgt]["weight"] += 0.5
            else:
                self._graph.add_edge(src, tgt, relation=relation, weight=1.0)

    def save(self) -> None:
        """Persist the graph to ``CONFIG.graph_path`` as a pickle file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump(self._graph, f)

    def load(self) -> bool:
        """Load the graph from disk.

        Returns:
            ``True`` if the file existed and was loaded successfully,
            ``False`` if the file does not exist.

        Raises:
            RuntimeError: If the file exists but cannot be unpickled.
        """
        if not self._path.exists():
            return False
        try:
            with open(self._path, "rb") as f:
                self._graph = pickle.load(f)
            return True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load entity graph from {self._path}: {exc}"
            ) from exc

    def get_entity_chunks(
        self,
        entity_names: list[str],
        k: int | None = None,
    ) -> list[dict]:
        """Return chunks associated with entities matching the given names.

        Performs a case-insensitive substring match so ``"bert"`` matches
        nodes named ``"BERT"`` and ``"RoBERTa"``. Deduplicates by content
        string. Results are ordered by node ``mention_count`` descending so
        chunks from more central entities appear first.

        Args:
            entity_names: List of entity name strings extracted from the
                user query.
            k: Maximum number of chunks to return. Defaults to
                ``CONFIG.graph_retrieval_top_k``.

        Returns:
            List of chunk dicts, each with ``"content"``, ``"metadata"``,
            ``"rerank_score"``, ``"rrf_score"``, and ``"source"`` keys.
        """
        k = k or CONFIG.graph_retrieval_top_k
        all_node_names = list(self._graph.nodes())

        seen_content: set[str] = set()
        # (mention_count, is_body_text, content, metadata)
        # is_body_text=1 sorts before heading/unknown (higher = better)
        candidates: list[tuple[int, int, str, dict]] = []

        for query_name in entity_names:
            q_lower = query_name.lower()
            matched_nodes = [
                n for n in all_node_names if q_lower in n.lower()
            ]
            for node_name in matched_nodes:
                node_data = self._graph.nodes[node_name]
                mention_count: int = node_data.get("mention_count", 1)
                chunks: list[str] = node_data.get("chunks", [])
                metadatas: list[dict] = node_data.get("chunk_metadatas", [])
                # Pad metadatas if missing (backwards compat with old graphs)
                if len(metadatas) < len(chunks):
                    metadatas = metadatas + [{}] * (len(chunks) - len(metadatas))
                for content, meta in zip(chunks, metadatas):
                    if content and content not in seen_content:
                        seen_content.add(content)
                        is_body = 1 if meta.get("block_type") == "text" else 0
                        candidates.append((mention_count, is_body, content, meta))

        # Primary: body text over headings; secondary: mention_count descending
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

        return [
            {
                "content": content,
                "metadata": meta,
                "rerank_score": 0.0,
                "rrf_score": 0.0,
                "source": "graph",
            }
            for _, _, content, meta in candidates[:k]
        ]

    @property
    def stats(self) -> dict:
        """Return a summary of graph statistics.

        Returns:
            Dict with keys ``"nodes"`` (int), ``"edges"`` (int), and
            ``"total_mentions"`` (int — sum of all node mention counts).
        """
        total_mentions = sum(
            data.get("mention_count", 0)
            for _, data in self._graph.nodes(data=True)
        )
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "total_mentions": total_mentions,
        }
