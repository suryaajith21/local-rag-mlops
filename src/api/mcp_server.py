"""MCP server — exposes the RAG engine as tools to MCP-compatible clients.

Implements three tools (query, ingest, get_stats) that proxy to the FastAPI
server over HTTP. The FastAPI server must be running at BASE_URL before any
tool call is made.

Run with:
    python -m src.api.mcp_server
"""

from __future__ import annotations

import asyncio

import httpx
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

BASE_URL = "http://localhost:8000"

server = Server("rag-engine")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare the three tools this server exposes."""
    return [
        types.Tool(
            name="query",
            description=(
                "Query the RAG knowledge base with a natural language question. "
                "Returns a grounded answer with source citations. "
                "Uses hybrid retrieval (BM25 + dense vector), CrossEncoder "
                "reranking, and an entity knowledge graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from the knowledge base",
                    },
                    "use_graph": {
                        "type": "boolean",
                        "description": "Whether to augment with entity graph retrieval",
                        "default": True,
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="ingest",
            description=(
                "Trigger document ingestion pipeline. Parses all PDFs in the "
                "data/ directory, chunks them, builds vector index, BM25 index, "
                "and entity knowledge graph. Use when documents have been added "
                "or updated."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "chunking_mode": {
                        "type": "string",
                        "enum": ["template", "semantic"],
                        "description": "Chunking strategy to use",
                        "default": "template",
                    },
                },
            },
        ),
        types.Tool(
            name="get_stats",
            description=(
                "Return current system statistics: chunk counts, graph size, "
                "models in use."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch tool calls to the FastAPI server and format the response."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        if name == "query":
            resp = await client.post(
                f"{BASE_URL}/query",
                json={
                    "query": arguments["question"],
                    "use_graph": arguments.get("use_graph", True),
                },
            )
            data = resp.json()
            text = (
                f"Answer: {data['answer']}\n\n"
                f"Route: {data['route']} | "
                f"Chunks used: {data['chunks_used']} | "
                f"Graph chunks: {data['graph_chunks_added']} | "
                f"Time: {data['time_seconds']}s\n"
                f"Sources: {', '.join(data['sources']) if data['sources'] else 'none'}"
            )

        elif name == "ingest":
            resp = await client.post(
                f"{BASE_URL}/ingest",
                json={"chunking_mode": arguments.get("chunking_mode", "template")},
            )
            data = resp.json()
            text = (
                f"Ingestion complete.\n"
                f"Files: {data['files_processed']} | "
                f"Chunks: {data['chunks_created']} | "
                f"Graph nodes: {data['graph_nodes']} | "
                f"Graph edges: {data['graph_edges']} | "
                f"Time: {data['time_seconds']}s"
            )

        elif name == "get_stats":
            resp = await client.get(f"{BASE_URL}/stats")
            data = resp.json()
            text = "\n".join(f"{k}: {v}" for k, v in data.items())

        else:
            text = f"Unknown tool: {name}"

    return [types.TextContent(type="text", text=text)]


async def main() -> None:
    """Entry point: run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
