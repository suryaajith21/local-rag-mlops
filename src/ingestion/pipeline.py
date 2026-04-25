"""Ingestion pipeline: parse → chunk → embed → store.

Orchestrates the full ingestion flow from raw PDFs to stored chunks.
Entity graph extraction is stubbed and will be filled in Phase 4.
"""

import asyncio
import time
from pathlib import Path
from typing import Literal

from src.config import CONFIG
from src.graph.engine import GraphEngine
from src.graph.extractor import EntityExtractor
from src.ingestion.chunker import chunk_blocks
from src.ingestion.parser import parse_directory
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore


async def run_ingestion_pipeline(
    data_dir: Path = CONFIG.data_dir,
    chunking_mode: Literal["template", "semantic"] = "template",
    clear_existing: bool = True,
) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → store.

    Parses all PDF files found under ``data_dir``, chunks them using
    ``chunking_mode``, then writes to the vector store, BM25 index, and
    entity graph (currently stubbed).

    Args:
        data_dir: Directory to search recursively for PDF files.
        chunking_mode: ``"template"`` for fixed-size chunking or
            ``"semantic"`` for embedding-based chunking.
        clear_existing: When ``True``, existing vector store and BM25 index
            are cleared before writing (prevents the duplicate-chunk
            corruption bug from the original rebuild_db.py).

    Returns:
        Summary dict with keys:
            - ``files_processed`` (int)
            - ``blocks_extracted`` (int)
            - ``chunks_created`` (int)
            - ``tables_preserved`` (int)
            - ``vector_store_count`` (int): chunks confirmed in ChromaDB
            - ``bm25_ready`` (bool): whether BM25 index was built
            - ``graph_nodes`` (int): entity nodes in the knowledge graph
            - ``graph_edges`` (int): relationships in the knowledge graph
            - ``entities_extracted`` (int): total entity mentions extracted
            - ``time_seconds`` (float)
    """
    start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"Ingestion Pipeline Starting")
    print(f"  data_dir      : {data_dir}")
    print(f"  chunking_mode : {chunking_mode}")
    print(f"  clear_existing: {clear_existing}")
    print(f"{'='*60}")

    # --- Stage 0: Initialise stores ---
    vector_store = VectorStore()
    bm25_store = BM25Store()

    if clear_existing:
        print("\n[Stage 0] Clearing existing stores...")
        vector_store.clear()
        print(f"  Cleared ChromaDB collection '{VectorStore.COLLECTION_NAME}'")
        if CONFIG.bm25_index_path.exists():
            CONFIG.bm25_index_path.unlink()
            print(f"  Cleared BM25 index at {CONFIG.bm25_index_path}")

    # --- Stage 1: Parse PDFs ---
    print(f"\n[Stage 1] Parsing PDFs in {data_dir} ...")
    blocks = parse_directory(data_dir)
    files_processed = len({b["metadata"]["source"] for b in blocks})
    blocks_extracted = len(blocks)

    block_type_counts: dict[str, int] = {}
    for b in blocks:
        bt = b["metadata"].get("block_type", "unknown")
        block_type_counts[bt] = block_type_counts.get(bt, 0) + 1

    print(f"\n  Files processed : {files_processed}")
    print(f"  Blocks extracted: {blocks_extracted}")
    for btype, count in sorted(block_type_counts.items()):
        print(f"    {btype:<10}: {count}")

    if blocks_extracted == 0:
        print("\nNo blocks extracted — aborting pipeline.")
        return {
            "files_processed": 0,
            "blocks_extracted": 0,
            "chunks_created": 0,
            "tables_preserved": 0,
            "time_seconds": round(time.perf_counter() - start, 2),
        }

    # --- Stage 2: Chunk blocks ---
    print(f"\n[Stage 2] Chunking blocks (mode={chunking_mode!r}) ...")
    chunks = chunk_blocks(blocks, mode=chunking_mode)
    chunks_created = len(chunks)

    tables_preserved = sum(
        1 for c in chunks if c["metadata"].get("block_type") == "table"
    )
    print(f"  Chunks created   : {chunks_created}")
    print(f"  Tables preserved : {tables_preserved}")

    # --- Stage 3: Write to vector store ---
    print("\n[Stage 3] Writing to vector store (ChromaDB) ...")
    added = vector_store.add_chunks(chunks)
    vs_count = vector_store.count
    print(f"  Chunks submitted : {added}")
    print(f"  Collection count : {vs_count}")

    # --- Stage 4: Write to BM25 index ---
    print("\n[Stage 4] Building BM25 index ...")
    bm25_store.build(chunks)
    bm25_size = CONFIG.bm25_index_path.stat().st_size if CONFIG.bm25_index_path.exists() else 0
    print(f"  BM25 index saved : {CONFIG.bm25_index_path}")
    print(f"  Index size       : {bm25_size / 1024:.1f} KB")

    # --- Stage 5: Build entity knowledge graph ---
    print("\n[Stage 5] Building entity knowledge graph ...")
    engine = GraphEngine()
    if clear_existing:
        engine.clear()
        if CONFIG.graph_path.exists():
            CONFIG.graph_path.unlink()
            print(f"  Cleared graph at {CONFIG.graph_path}")

    extractor = EntityExtractor()
    entities_total = 0
    headings_skipped = 0

    for i, chunk in enumerate(chunks):
        if chunk["metadata"].get("block_type") == "heading":
            headings_skipped += 1
            continue
        extraction = extractor.extract(chunk)
        extraction["chunk_content"] = chunk["content"]
        extraction["chunk_metadata"] = chunk["metadata"]
        engine.add_extraction(extraction)
        entities_total += len(extraction.get("entities", []))
        if (i + 1) % 25 == 0:
            print(f"  Graph: processed {i + 1}/{len(chunks)} chunks ...")

    print(f"  Heading chunks skipped: {headings_skipped}")

    engine.save()
    graph_stats = engine.stats
    graph_size = CONFIG.graph_path.stat().st_size if CONFIG.graph_path.exists() else 0
    print(f"  Graph built   : {graph_stats}")
    print(f"  Graph size    : {graph_size / 1024:.1f} KB")
    print(f"  Entities total: {entities_total}")

    elapsed = round(time.perf_counter() - start, 2)
    summary = {
        "files_processed": files_processed,
        "blocks_extracted": blocks_extracted,
        "chunks_created": chunks_created,
        "tables_preserved": tables_preserved,
        "vector_store_count": vs_count,
        "bm25_ready": bm25_store.is_ready,
        "graph_nodes": graph_stats["nodes"],
        "graph_edges": graph_stats["edges"],
        "entities_extracted": entities_total,
        "time_seconds": elapsed,
    }

    print(f"\n{'='*60}")
    print("Ingestion Pipeline Complete")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    asyncio.run(run_ingestion_pipeline())
