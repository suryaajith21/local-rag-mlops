"""Query router — classifies queries and dispatches to the appropriate path.

Three routes:
- DIRECT:   single-pass hybrid retrieval; sufficient for factual lookups.
- MULTIHOP: answer requires synthesising across multiple chunks; decompose
            the question into sub-questions before retrieving.
- WEB:      confidence in the local corpus is too low; fall back to a live
            DuckDuckGo search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import CONFIG

_ROUTER_PROMPT = """\
You are a query router for a document-based question-answering system.

Classify the user query into exactly one of three routes and respond ONLY with
valid JSON — no markdown fences, no preamble, no trailing text.

Routes:
- "direct": A single factual question answerable from one passage in a static
  document corpus. Examples: "What is RAG?", "What does the paper say about RLHF?"
  IMPORTANT — "direct" also covers questions that ask for a list or summary of
  a SINGLE topic: "What are the risks of X?", "What does the paper say about Y?",
  "List the main approaches to Z?" These are DIRECT even though the answer has
  multiple parts, because they are asking about ONE subject.
  BAD (wrongly classified as multihop): "What are the main risks of generative AI?"
  → DIRECT — single topic, even though the answer has multiple parts.
- "multihop": ONLY for questions that require retrieving information about TWO OR
  MORE DISTINCT TOPICS and synthesizing them together. Triggered by explicit
  comparison language: "compare", "difference between", "X vs Y", "how does X
  relate to Y". For multihop, decompose into 2-3 independent sub_questions that
  can each be answered by a single passage.
  GOOD example: "Compare GPT-4 risks vs open source LLM risks" → MULTIHOP
  (two distinct entities being compared) →
  sub_questions: ["What risks does GPT-4 pose?",
                  "What risks do open source LLMs pose?"]
  GOOD example: "Compare GAN training instability with VAE posterior collapse" →
  sub_questions: ["What causes GAN training instability?",
                  "What is VAE posterior collapse?"]
- "web": The query is about current events, very recent publications by name,
  live pricing, or anything clearly outside a static document corpus.
  Examples: "What did OpenAI announce this week?", "Current price of GPT-4 API"

Response format (JSON only, no other text):
{{
  "route": "direct" | "multihop" | "web",
  "reasoning": "one sentence explaining the classification",
  "sub_questions": []
}}

For "multihop", populate sub_questions with 2-3 strings.
For "direct" and "web", sub_questions must be an empty list [].

User query: {query}"""


class QueryRoute(Enum):
    DIRECT = "direct"
    MULTIHOP = "multihop"
    WEB = "web"


@dataclass
class RouterDecision:
    """Result of routing a user query.

    Attributes:
        route: The selected retrieval path.
        reasoning: One-sentence explanation from the LLM.
        sub_questions: Decomposed sub-questions (MULTIHOP only; empty otherwise).
    """

    route: QueryRoute
    reasoning: str
    sub_questions: list[str] = field(default_factory=list)


class QueryRouter:
    """Classifies queries and returns a routing decision.

    Uses the configured Ollama LLM with temperature=0 for deterministic
    classification. JSON parsing failures default to DIRECT so retrieval
    always proceeds.
    """

    def __init__(self) -> None:
        self._llm = ChatOllama(
            model=CONFIG.ollama_model,
            temperature=0.0,
            base_url=CONFIG.ollama_base_url,
        )

    def route(self, query: str) -> RouterDecision:
        """Classify a query and return the routing decision.

        Prompts the LLM to respond in a strict JSON format, then parses the
        response. Any parse failure or unexpected route value falls back to
        ``DIRECT`` so retrieval is never blocked.

        Args:
            query: The user's natural-language question.

        Returns:
            A ``RouterDecision`` with route, reasoning, and (for MULTIHOP)
            sub_questions populated.
        """
        _default = RouterDecision(
            route=QueryRoute.DIRECT,
            reasoning="parse failed — defaulting to direct",
            sub_questions=[],
        )

        try:
            prompt = _ROUTER_PROMPT.format(query=query)
            response = self._llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Strip markdown fences if the model disobeys the prompt
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    l for l in lines if not l.startswith("```")
                ).strip()

            # Find first JSON object in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return _default

            data = json.loads(raw[start:end])

            route_str = data.get("route", "direct").lower()
            try:
                route = QueryRoute(route_str)
            except ValueError:
                route = QueryRoute.DIRECT

            return RouterDecision(
                route=route,
                reasoning=str(data.get("reasoning", "")),
                sub_questions=list(data.get("sub_questions", [])),
            )

        except Exception:
            return _default


def web_search_fallback(query: str) -> list[dict]:
    """Run a DuckDuckGo search for queries outside the local corpus.

    Args:
        query: The user's natural-language question.

    Returns:
        Up to ``CONFIG.web_search_max_results`` result dicts. Each dict has:
            - ``"content"`` (str): Title and body snippet joined by newline.
            - ``"metadata"`` (dict): ``source`` (URL), ``block_type``
              ``"web_result"``, ``section`` ``""``.
            - ``"rerank_score"`` (float): 0.0 — web results bypass reranking.
            - ``"rrf_score"`` (float): 0.0.

        Returns an empty list if the search fails or returns no results.
    """
    from ddgs import DDGS

    try:
        results = []
        with DDGS() as ddgs:
            hits = ddgs.text(query, max_results=CONFIG.web_search_max_results)
            for hit in hits:
                title = hit.get("title", "")
                body = hit.get("body", "")
                url = hit.get("href", "")
                results.append(
                    {
                        "content": f"{title}\n{body}",
                        "metadata": {
                            "source": url,
                            "block_type": "web_result",
                            "section": "",
                        },
                        "rerank_score": 0.0,
                        "rrf_score": 0.0,
                    }
                )
        return results
    except Exception:
        return []
