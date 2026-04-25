"""BM25 sparse keyword retrieval index, persisted to disk as a pickle file.

Wraps rank-bm25's BM25Okapi with simple lowercase whitespace tokenization.
The chunk list is stored alongside the index so rank-to-document lookup is
O(1) after load.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import CONFIG


class BM25Store:
    """BM25 sparse retrieval index.

    Stores both the BM25Okapi object and the source chunk list as a pickle
    so the index can be rebuilt once during ingestion and loaded cheaply on
    every subsequent process start.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        self._index_path: Path = CONFIG.bm25_index_path

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenization.

        Strips punctuation at word boundaries and splits on any whitespace.
        The same function is used for both index construction and query
        tokenization to guarantee consistent vocabulary matching.

        Args:
            text: Input string to tokenize.

        Returns:
            List of lowercase token strings.
        """
        text = text.lower()
        # Replace punctuation with spaces, then split
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks and persist to disk.

        Tokenizes the ``"content"`` field of each chunk, constructs a
        ``BM25Okapi`` index, and saves both the index and the original chunk
        list to ``CONFIG.bm25_index_path`` as a single pickle.

        Args:
            chunks: List of chunk dicts with ``"content"`` and ``"metadata"``
                keys, as produced by the ingestion pipeline.
        """
        self._chunks = list(chunks)
        corpus = [self._tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

        # Ensure the parent directory exists
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"bm25": self._bm25, "chunks": self._chunks}
        with open(self._index_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self) -> bool:
        """Load the BM25 index and chunk list from disk.

        Args: None

        Returns:
            ``True`` if the index file exists and was loaded successfully,
            ``False`` if the file does not exist.

        Raises:
            RuntimeError: If the file exists but cannot be unpickled.
        """
        if not self._index_path.exists():
            return False
        try:
            with open(self._index_path, "rb") as f:
                payload = pickle.load(f)
            self._bm25 = payload["bm25"]
            self._chunks = payload["chunks"]
            return True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BM25 index from {self._index_path}: {exc}"
            ) from exc

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """BM25 keyword search over the indexed chunks.

        Args:
            query: Natural-language query string.
            k: Number of results to return. Defaults to
                ``CONFIG.retrieval_k_candidate``.

        Returns:
            List of result dicts sorted by BM25 score descending, each with:
                - ``"content"`` (str): Chunk text.
                - ``"metadata"`` (dict): Source metadata.
                - ``"score"`` (float): Raw BM25 score (higher is better).
                - ``"rank"`` (int): 1-indexed position in this result list.

        Raises:
            RuntimeError: If the index has not been built or loaded.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index is not ready. Call build() or load() first."
            )

        k = k or CONFIG.retrieval_k_candidate
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each chunk with its score, sort descending, take top-k
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:k]

        results: list[dict] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append(
                {
                    "content": self._chunks[idx]["content"],
                    "metadata": self._chunks[idx].get("metadata", {}),
                    "score": float(score),
                    "rank": rank,
                }
            )

        return results

    @property
    def is_ready(self) -> bool:
        """Return True if the index has been built or loaded successfully."""
        return self._bm25 is not None
