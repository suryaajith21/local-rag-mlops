"""FastAPI application — main entry point for the RAG engine.

Exposes /health, /ingest, /query, and /stats endpoints. All heavyweight
components (embeddings, BM25, CrossEncoder, Ollama client) are loaded once
at startup via the lifespan context manager and stored on app.state so every
request reuses them without reload overhead.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import CONFIG


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    chunking_mode: str = "template"
    clear_existing: bool = True


class IngestResponse(BaseModel):
    files_processed: int
    chunks_created: int
    graph_nodes: int
    graph_edges: int
    time_seconds: float


class QueryRequest(BaseModel):
    query: str
    use_graph: bool = True


class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    sources: list[str]
    chunks_used: int
    graph_chunks_added: int
    hyde_passage: str | None
    time_seconds: float


class StatsResponse(BaseModel):
    vector_chunks: int
    bm25_ready: bool
    graph_nodes: int
    graph_edges: int
    embedding_model: str
    reranker_model: str
    ollama_model: str


# ---------------------------------------------------------------------------
# Lifespan — load all models once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavyweight components once at startup, reuse on every request."""
    from src.generation.generator import Generator
    from src.graph.engine import GraphEngine
    from src.graph.retriever import GraphRetriever
    from src.retrieval import HybridRetriever

    t0 = time.time()
    app.state.retriever = HybridRetriever()
    app.state.engine = GraphEngine()
    app.state.engine.load()
    app.state.graph_retriever = GraphRetriever(app.state.engine)
    app.state.generator = Generator()
    print(f"All components loaded in {time.time() - t0:.1f}s")
    yield
    # No teardown needed for this stack


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Engine API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Lightweight health check with component status."""
    return {
        "status": "ok",
        "vector_chunks": app.state.retriever.vector_store.count,
        "bm25_ready": app.state.retriever.bm25_store.is_ready,
        "graph_nodes": app.state.engine.stats["nodes"],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Trigger a full ingestion pipeline run.

    Offloaded to a thread pool so the async event loop is not blocked during
    the multi-minute PDF parsing and graph extraction process. After completion
    all in-memory components are reloaded from the freshly written indexes.
    """
    from src.ingestion.pipeline import run_ingestion_pipeline

    # run_ingestion_pipeline is itself async; run it in a thread with its own
    # event loop so it doesn't conflict with the server's running loop.
    summary = await asyncio.to_thread(
        lambda: asyncio.run(
            run_ingestion_pipeline(
                chunking_mode=req.chunking_mode,
                clear_existing=req.clear_existing,
            )
        )
    )

    # Reload all components with fresh data
    from src.retrieval import HybridRetriever
    from src.graph.retriever import GraphRetriever

    app.state.retriever = HybridRetriever()
    app.state.engine.load()
    app.state.graph_retriever = GraphRetriever(app.state.engine)

    return IngestResponse(
        **{k: summary[k] for k in IngestResponse.model_fields if k in summary}
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Full query pipeline: route → HyDE → hybrid retrieve → graph augment → generate."""
    from src.query import process_query

    t0 = time.time()
    gr = app.state.graph_retriever if req.use_graph else None

    result = await asyncio.to_thread(
        process_query, req.query, app.state.retriever, gr
    )

    gen_result = await asyncio.to_thread(
        app.state.generator.generate, req.query, result["chunks"]
    )

    return QueryResponse(
        query=req.query,
        route=result["route"],
        answer=gen_result["answer"],
        sources=gen_result["sources"],
        chunks_used=gen_result["chunk_count"],
        graph_chunks_added=result.get("graph_chunks_added", 0),
        hyde_passage=result.get("hyde_passage"),
        time_seconds=round(time.time() - t0, 2),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return current system statistics."""
    return StatsResponse(
        vector_chunks=app.state.retriever.vector_store.count,
        bm25_ready=app.state.retriever.bm25_store.is_ready,
        graph_nodes=app.state.engine.stats["nodes"],
        graph_edges=app.state.engine.stats["edges"],
        embedding_model=CONFIG.embedding_model,
        reranker_model=CONFIG.reranker_model,
        ollama_model=CONFIG.ollama_model,
    )
