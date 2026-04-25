"""Reciprocal Rank Fusion for merging dense and sparse retrieval results.

RRF is a rank-based fusion method that is robust to score-scale differences
between retrieval systems. A document ranked highly in either list receives a
strong boost; one appearing in both lists is boosted from both sides.
"""

from __future__ import annotations

from src.config import CONFIG


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int | None = None,
) -> list[dict]:
    """Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

    The RRF score for a document is the sum over all ranked lists of
    ``1 / (rrf_k + rank)``, where ``rank`` is the 1-indexed position in
    that list. Documents appearing in only one list are still scored (one
    term); documents in both lists receive contributions from both (two
    terms summed), yielding a natural boost for consistently-ranked results.

    Deduplication uses the chunk ``"content"`` string as the key, which is
    guaranteed to match between dense and sparse results because both systems
    index the same chunks produced by the ingestion pipeline.

    Args:
        dense_results: Output of ``VectorStore.search()`` — list of dicts
            with ``"content"``, ``"metadata"``, ``"score"``, and ``"rank"``
            fields. Ranks are 1-indexed.
        sparse_results: Output of ``BM25Store.search()`` — same structure.
        k: Number of top results to return. Defaults to
            ``CONFIG.retrieval_k_candidate``.

    Returns:
        List of merged result dicts sorted by RRF score descending, each
        containing:
            - ``"content"`` (str): Chunk text.
            - ``"metadata"`` (dict): Source metadata (from whichever list
              first contained this document).
            - ``"rrf_score"`` (float): Combined RRF score.
    """
    k = k or CONFIG.retrieval_k_candidate
    rrf_k = CONFIG.rrf_k

    # content → {"metadata": ..., "rrf_score": float}
    scores: dict[str, dict] = {}

    for result_list in (dense_results, sparse_results):
        for item in result_list:
            content = item["content"]
            rank = item["rank"]
            contribution = 1.0 / (rrf_k + rank)

            if content not in scores:
                scores[content] = {
                    "metadata": item.get("metadata", {}),
                    "rrf_score": 0.0,
                }
            scores[content]["rrf_score"] += contribution

    merged = [
        {
            "content": content,
            "metadata": data["metadata"],
            "rrf_score": data["rrf_score"],
        }
        for content, data in scores.items()
    ]

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged[:k]
