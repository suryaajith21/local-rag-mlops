# Production RAG Engine
[Updated: 4/25/2026]
A self-hosted, privacy-first Retrieval-Augmented Generation system
built as an upgrade over the original RAG pipeline. All inference runs
locally.

## Architecture

```
PDF Documents
     |
     v
+------------------------------------------+
|           Ingestion Pipeline             |
|  pdfplumber -> layout-aware parsing      |
|  Template or semantic chunking           |
|  Heading/bibliography filtering          |
+-------------+----------------+-----------+
              |                |
              v                v
       ChromaDB           BM25 Index        Entity Graph
    (dense vectors)    (sparse keyword)   (NetworkX + LLM
                                           extraction)
              |                |                  |
              +--------+-------+                  |
                       |                          |
                       v                          |
              RRF Fusion (k=20)                   |
                       |                          |
                       v                          |
           CrossEncoder Reranker                  |
           (ms-marco-MiniLM-L-6-v2)               |
                       |                          |
                       +-----------+--------------+
                                   |
                                   v
                       Query Router + HyDE
                     (direct / multihop / web)
                                   |
                                   v
                       llama3.2 Generator
                       (Ollama, local only)
                                   |
                                   v
                       FastAPI + MCP Server
```

## V1 to V2 Improvements

| Component | V1 | V2 |
|---|---|---|
| PDF parsing | PyPDFLoader (page-level, no structure) | pdfplumber (layout-aware, table preservation) |
| Chunking | Fixed 500-char split | Template or semantic mode, configurable |
| Retrieval | Dense vector only (k=3) | BM25 + dense + RRF fusion (k=20 candidates) |
| Reranking | None | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Query understanding | None | HyDE + agentic router (direct/multihop/web) |
| Knowledge graph | None | Entity extraction + NetworkX graph |
| API | CLI loop only | FastAPI REST + MCP server |
| Corpus | 1 document | 4 documents, multi-domain |
| Evaluation | 4 questions, 2 metrics | 20 questions, 4 metrics |
| Judge model | llama3.2 (self-grading) | Mistral-7B (independent judge) |
| Chunk deduplication | None | SHA256(source:page:content) |
| Config | Hardcoded, inconsistent | Single config.py dataclass |
| CI artifact | Ephemeral (not saved) | Uploaded to GitHub Actions artifacts |

## Evaluation Results

Evaluated on 20 questions across 4 categories (DIRECT, ENTITY,
MULTIHOP, ADVERSARIAL) using Mistral-7B as independent judge.

| Metric | V1 | V2 | Notes |
|---|---|---|---|
| Faithfulness | 0.89* | 0.94 | *V1 used self-grading (inflated) |
| Answer Relevancy | 0.86* | 0.46 | *V1 self-graded; excludes correct refusals |
| Context Precision | not measured | 0.80 | Not measured in V1 |
| Context Recall | not measured | 0.87 | Not measured in V1 |

Answer relevancy is suppressed by the ADVERSARIAL category where
correct refusals ("The document does not contain sufficient information")
score 0.0 under Ragas' metric design.

V1 faithfulness and answer_relevancy were measured with llama3.2
grading its own outputs, introducing self-consistency bias. V2 uses
an independent Mistral-7B judge for all metrics.

## Key Design Decisions

**Why hybrid retrieval?** Dense retrieval misses exact keyword matches
for named entities. In testing, BM25 found 40% of relevant chunks
that dense retrieval missed entirely on entity-centric queries like
"Thompson sampling AutoHarness". RRF fusion captures both signals.

**Why a CrossEncoder reranker?** Bi-encoders embed query and document
independently. CrossEncoders see the full (query, document) pair and
score relevance jointly. It is more accurate but too slow to
run on all chunks. The two-stage pipeline (fast retrieval + accurate
reranking) is the production standard.

**Why HyDE?** Raw queries often use different vocabulary than document
chunks. A query like "transformer problems" matches poorly against
chunks discussing "limitations of self-attention for long sequences."
HyDE generates a hypothetical answer passage and embeds that instead,
bridging the vocabulary gap. Measured to help most on short/vague
queries; minimal lift on fully-formed technical questions.

**Why an entity graph?** Pure vector search retrieves semantically
similar chunks but misses entity-relationship context. Graph traversal
finds entity-linked chunks that compete against hybrid results via the
CrossEncoder — graph chunks only survive to the generator if they
outscore a hybrid result, preventing low-quality graph noise from
polluting context.

**Why local-only?** Privacy. No document content or queries leave
the machine. The MCP server exposes the engine to AI assistants
without any cloud dependency.

## Setup

### Local (development)

```bash
# 1. Create environment
conda create -n rag-ops python=3.10
conda activate rag-ops
pip install -r requirements.txt

# 2. Start Ollama and pull models
ollama serve
ollama pull llama3.2
ollama pull mistral

# 3. Add documents to data/

# 4. Run ingestion
python -c "
import asyncio
from src.ingestion.pipeline import run_ingestion_pipeline
asyncio.run(run_ingestion_pipeline(clear_existing=True))
"

# 5. Start API server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the risks of large language models?"}'
```

### Docker

```bash
docker compose up --build
# Then trigger ingestion:
curl -X POST http://localhost:8000/ingest
```

### MCP (Claude Desktop / Cursor)

Add to your MCP config:

```json
{
  "mcpServers": {
    "rag-engine": {
      "command": "python",
      "args": ["-m", "src.api.mcp_server"],
      "cwd": "/path/to/mlops-rag-pipeline"
    }
  }
}
```

Available MCP tools: query, ingest, get_stats

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Component status |
| /stats | GET | Chunk counts, graph size, models |
| /query | POST | Full RAG query pipeline |
| /ingest | POST | Trigger ingestion pipeline |

### Query request

```json
{
  "query": "What are transformer attention heads?",
  "use_graph": true
}
```

### Query response

```json
{
  "query": "...",
  "route": "direct",
  "answer": "...",
  "sources": ["1706.03762v7.pdf"],
  "chunks_used": 5,
  "graph_chunks_added": 1,
  "hyde_passage": "...",
  "time_seconds": 8.4
}
```

## Project Structure

```
mlops-rag-pipeline/
├── src/
│   ├── config.py              # Single source of truth for all config
│   ├── ingestion/
│   │   ├── parser.py          # pdfplumber layout-aware extraction
│   │   ├── chunker.py         # Template + semantic chunking
│   │   └── pipeline.py        # Async ingestion orchestrator
│   ├── retrieval/
│   │   ├── vector_store.py    # ChromaDB dense retrieval
│   │   ├── bm25_store.py      # BM25 sparse retrieval
│   │   ├── hybrid.py          # RRF fusion
│   │   └── reranker.py        # CrossEncoder reranking
│   ├── query/
│   │   ├── router.py          # Direct/multihop/web routing
│   │   └── hyde.py            # Hypothetical document expansion
│   ├── graph/
│   │   ├── extractor.py       # LLM entity extraction
│   │   ├── engine.py          # NetworkX graph store
│   │   └── retriever.py       # Graph-augmented retrieval
│   ├── generation/
│   │   └── generator.py       # LLM synthesis with citation
│   └── api/
│       ├── server.py          # FastAPI application
│       └── mcp_server.py      # MCP server (3 tools)
├── evaluation/
│   ├── v1/                    # Original baseline results
│   │   ├── test_dataset.json
│   │   └── evaluation_results.csv
│   ├── v2/                    # Production evaluation
│   │   ├── test_dataset.json
│   │   ├── evaluation_results.csv
│   │   └── summary.json
│   └── eval_v2.py
├── data/                      # PDF documents
├── vector_db/                 # ChromaDB + BM25 index (gitignored)
├── docker-compose.yml
├── Dockerfile
└── .github/workflows/
    └── eval_pipeline.yml
```

## Corpus

| Document | Domain | Pages |
|---|---|---|
| genai_review.pdf | Generative AI survey (manuscript under review) | 109 |
| [1706.03762v7.pdf](https://arxiv.org/abs/1706.03762) | Transformer / Attention | ~15 |
| [2603.03329v1.pdf](https://arxiv.org/abs/2603.03329) | AutoHarness / LLM agents | ~10 |
| [2602.02276v1.pdf](https://arxiv.org/abs/2602.02276) | Kimi K2.5 / Multimodal agents | ~20 |


## Sources for data:


## Known Limitations

- **Table detection**: pdfplumber detects ruling-line tables only. LaTeX booktabs-style tables in academic papers are captured as body text. Fix: camelot-py lattice/stream detection or a vision model pass for image-rendered tables.
- **ENTITY answer relevancy**: Queries about entities with sparse corpus coverage return adjacent context rather than precise passages. Fix: expand corpus or add entity-focused documents.
- **Ollama concurrency**: llama3.2 serves one request at a time. Concurrent users queue linearly. Fix: vLLM or TGI for batched inference.
