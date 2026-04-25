"""ChromaDB wrapper for dense vector retrieval.

Uses the modern PersistentClient API with a SentenceTransformer embedding
function. IDs are content-derived SHA256 hashes to make ingestion idempotent.
"""

from __future__ import annotations

import hashlib

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import CONFIG

_BATCH_SIZE = 100


def _chunk_id(content: str, source: str = "", page: int = 0) -> str:
    """Return first 16 hex chars of SHA256(source+page+content) as a stable chunk ID."""
    key = f"{source}:{page}:{content}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class VectorStore:
    """ChromaDB wrapper for dense vector retrieval.

    Manages a single named collection backed by a local PersistentClient.
    Embedding is handled inside ChromaDB via SentenceTransformerEmbeddingFunction
    so embeddings are computed on add and on query consistently.
    """

    COLLECTION_NAME = "rag_chunks"

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(CONFIG.vector_db_dir))
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=CONFIG.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        """Delete and recreate the collection, removing all stored chunks.

        Safe to call even when the collection is empty.
        """
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """Add chunks to the ChromaDB collection.

        IDs are derived from SHA256(content) to prevent duplicates on
        re-ingestion. Documents are batched in groups of ``_BATCH_SIZE`` to
        avoid memory pressure during large ingestion runs.

        Args:
            chunks: List of chunk dicts, each with ``"content"`` (str) and
                ``"metadata"`` (dict) keys.

        Returns:
            Number of chunks successfully submitted to ChromaDB.
        """
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            content = chunk["content"]
            meta = chunk.get("metadata", {})
            # ChromaDB requires all metadata values to be str/int/float/bool
            safe_meta = {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                         for k, v in meta.items()}
            ids.append(_chunk_id(content, str(meta.get("source", "")), int(meta.get("page", 0))))
            documents.append(content)
            metadatas.append(safe_meta)

        added = 0
        for start in range(0, len(ids), _BATCH_SIZE):
            end = start + _BATCH_SIZE
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            added += end - start if end <= len(ids) else len(ids) - start

        return added

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """Dense cosine-similarity search against the collection.

        Args:
            query: Natural-language query string.
            k: Number of results to return. Defaults to
                ``CONFIG.retrieval_k_candidate``.

        Returns:
            List of result dicts sorted by similarity descending, each with:
                - ``"content"`` (str): Chunk text.
                - ``"metadata"`` (dict): Source metadata.
                - ``"score"`` (float): Similarity score in [0, 1] where
                  1 is most similar (converted from cosine distance).
                - ``"rank"`` (int): 1-indexed position in this result list.
        """
        k = k or CONFIG.retrieval_k_candidate
        n_results = min(k, self.count) if self.count > 0 else 0
        if n_results == 0:
            return []

        response = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        results: list[dict] = []
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        for rank, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            # ChromaDB returns cosine distance ∈ [0, 2]; convert to similarity
            results.append(
                {
                    "content": doc,
                    "metadata": meta,
                    "score": float(1.0 - dist),
                    "rank": rank,
                }
            )

        return results

    @property
    def count(self) -> int:
        """Return number of chunks currently in the collection."""
        return self._collection.count()
