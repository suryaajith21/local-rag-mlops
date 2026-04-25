"""Query understanding pipeline.

Public interface: ``process_query()`` orchestrates routing → HyDE expansion
→ retrieval → graph augmentation into a single call. The API layer calls
only this function.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.config import CONFIG
from src.query.hyde import HyDEExpander
from src.query.router import QueryRoute, QueryRouter, RouterDecision, web_search_fallback
from src.retrieval import HybridRetriever

if TYPE_CHECKING:
    from src.graph.retriever import GraphRetriever


def process_query(
    query: str,
    retriever: HybridRetriever,
    graph_retriever: GraphRetriever | None = None,
) -> dict:
    """Full query understanding pipeline: route → expand → retrieve → augment.

    Args:
        query: The user's natural-language question.
        retriever: A live ``HybridRetriever`` instance (passed in so BM25 and
            CrossEncoder are not reloaded on every call).
        graph_retriever: Optional ``GraphRetriever`` for entity-graph
            augmentation. When ``None``, graph augmentation is skipped.

    Returns:
        A dict with keys:
            - ``"query"`` (str): Original query.
            - ``"route"`` (str): ``"direct"``, ``"multihop"``, or ``"web"``.
            - ``"reasoning"`` (str): Router's one-sentence explanation.
            - ``"hyde_passage"`` (str | None): Generated hypothetical passage,
              or ``None`` when HyDE was not used.
            - ``"chunks"`` (list[dict]): Final retrieved chunks.
            - ``"sub_questions"`` (list[str]): Populated for MULTIHOP only.
            - ``"graph_chunks_added"`` (int): Number of graph chunks merged in.
            - ``"time_seconds"`` (float): Wall-clock time for the full call.
    """
    start = time.perf_counter()

    router = QueryRouter()
    hyde = HyDEExpander()
    decision: RouterDecision = router.route(query)

    result: dict = {
        "query": query,
        "route": decision.route.value,
        "reasoning": decision.reasoning,
        "hyde_passage": None,
        "chunks": [],
        "sub_questions": decision.sub_questions,
        "graph_chunks_added": 0,
        "time_seconds": 0.0,
    }

    # ------------------------------------------------------------------
    # WEB route
    # ------------------------------------------------------------------
    if decision.route == QueryRoute.WEB:
        if CONFIG.router_web_fallback_enabled:
            result["chunks"] = web_search_fallback(query)
            result["time_seconds"] = round(time.perf_counter() - start, 2)
            return result
        else:
            # Web disabled — fall back to direct
            decision = RouterDecision(
                route=QueryRoute.DIRECT,
                reasoning="web fallback disabled — falling back to direct",
                sub_questions=[],
            )
            result["route"] = QueryRoute.DIRECT.value

    # ------------------------------------------------------------------
    # DIRECT route
    # ------------------------------------------------------------------
    if decision.route == QueryRoute.DIRECT:
        hyde_passage: str | None = None
        if CONFIG.hyde_enabled:
            hyde_passage = hyde.expand(query)
            result["hyde_passage"] = hyde_passage
        chunks = retriever.retrieve(query, hyde_query=hyde_passage)

    # ------------------------------------------------------------------
    # MULTIHOP route
    # ------------------------------------------------------------------
    elif decision.route == QueryRoute.MULTIHOP:
        sub_questions = decision.sub_questions or [query]
        seen_content: set[str] = set()
        all_chunks: list[dict] = []

        for sub_q in sub_questions[: CONFIG.max_multihop_steps]:
            hyde_passage = None
            if CONFIG.hyde_enabled:
                hyde_passage = hyde.expand(sub_q)

            sub_chunks = retriever.retrieve(sub_q, hyde_query=hyde_passage)
            for chunk in sub_chunks:
                if chunk["content"] not in seen_content:
                    seen_content.add(chunk["content"])
                    all_chunks.append(chunk)

        all_chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        chunks = all_chunks[: CONFIG.retrieval_k_final]

        if CONFIG.hyde_enabled and sub_questions:
            result["hyde_passage"] = hyde.expand(sub_questions[0])

    else:
        chunks = []

    # ------------------------------------------------------------------
    # Graph augmentation (DIRECT and MULTIHOP only)
    # ------------------------------------------------------------------
    if graph_retriever is not None:
        graph_chunks = graph_retriever.retrieve(query)
        if graph_chunks:
            # Rerank graph chunks so they compete on score, not position
            from src.retrieval.reranker import Reranker
            graph_chunks = Reranker().rerank(query, graph_chunks, k=len(graph_chunks))

        existing_contents = {c["content"] for c in chunks}
        for gc in graph_chunks:
            if gc["content"] not in existing_contents:
                chunks.append(gc)
                existing_contents.add(gc["content"])

        # Sort the full candidate pool by rerank_score, then clip
        chunks.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        chunks = chunks[: CONFIG.retrieval_k_final]

        # Count graph chunks that survived the final clip
        graph_contents = {gc["content"] for gc in graph_chunks}
        result["graph_chunks_added"] = sum(
            1 for c in chunks if c["content"] in graph_contents
        )

    result["chunks"] = chunks
    result["time_seconds"] = round(time.perf_counter() - start, 2)
    return result
