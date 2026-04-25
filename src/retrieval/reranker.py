"""CrossEncoder reranker — final relevance scoring before the LLM sees context.

The CrossEncoder jointly encodes (query, document) pairs, giving much more
accurate relevance scores than the bi-encoder used for initial retrieval, at
the cost of being too slow to run over the full corpus. It is therefore
applied only to the small candidate set produced by RRF fusion.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import CONFIG


class Reranker:
    """CrossEncoder reranker for final relevance scoring.

    Loads ``CONFIG.reranker_model`` once at construction time and scores
    (query, chunk) pairs on every call to ``rerank()``.
    """

    def __init__(self) -> None:
        self._model: CrossEncoder = CrossEncoder(CONFIG.reranker_model)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        k: int | None = None,
    ) -> list[dict]:
        """Score (query, chunk) pairs and return top-k by CrossEncoder score.

        Args:
            query: The user's query string.
            candidates: List of candidate chunk dicts, each with at least
                ``"content"`` and ``"metadata"`` fields. Typically the output
                of ``reciprocal_rank_fusion()``, which also carries
                ``"rrf_score"``.
            k: Number of chunks to return after reranking. Defaults to
                ``CONFIG.retrieval_k_final``.

        Returns:
            Top-k chunk dicts sorted by CrossEncoder score descending, each
            containing all original fields plus:
                - ``"rerank_score"`` (float): Raw CrossEncoder logit score.
        """
        if not candidates:
            return []

        k = k or CONFIG.retrieval_k_final

        pairs = [(query, c["content"]) for c in candidates]
        scores = self._model.predict(pairs)

        scored: list[dict] = []
        for chunk, score in zip(candidates, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:k]
