"""Chunking strategies for ingested document blocks.

Provides two modes:
- ``template``: Fixed-size splitting via RecursiveCharacterTextSplitter.
- ``semantic``: Embedding-based grouping that keeps semantically coherent
  sentences together.

Tables are never split in either mode.
"""

from __future__ import annotations

import copy
import re
from typing import Literal

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import CONFIG

# Module-level lazy cache so the embedding model is loaded at most once per
# process when semantic chunking is used.
_st_model: SentenceTransformer | None = None


def _get_st_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(CONFIG.embedding_model)
    return _st_model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _template_chunk_block(block: dict) -> list[dict]:
    """Split one block using RecursiveCharacterTextSplitter.

    Tables are returned as-is regardless of size.

    Args:
        block: A block dict with ``content`` and ``metadata`` keys.

    Returns:
        List of chunk dicts with the same metadata as the input block.
    """
    if block["metadata"].get("block_type") == "table":
        return [block]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.template_chunk_size,
        chunk_overlap=CONFIG.template_chunk_overlap,
        length_function=len,
    )
    sub_texts = splitter.split_text(block["content"])
    return [
        {"content": t, "metadata": copy.deepcopy(block["metadata"])}
        for t in sub_texts
        if t.strip()
    ]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ', '! ', or '? ' boundaries.

    Args:
        text: Input text string.

    Returns:
        List of non-empty sentence strings.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _semantic_chunk_block(block: dict) -> list[dict]:
    """Split one block into semantically coherent chunks.

    Sentences are embedded and grouped while cosine similarity between the
    new sentence and the running chunk centroid remains above
    ``CONFIG.semantic_similarity_threshold``. Chunks that exceed
    ``CONFIG.semantic_max_chunk_size`` are flushed immediately. Chunks
    shorter than ``CONFIG.semantic_min_chunk_size`` are merged into the
    preceding chunk.

    Tables are returned as-is regardless of size.

    Args:
        block: A block dict with ``content`` and ``metadata`` keys.

    Returns:
        List of chunk dicts with the same metadata as the input block.
    """
    if block["metadata"].get("block_type") == "table":
        return [block]

    sentences = _split_sentences(block["content"])
    if not sentences:
        return []

    model = _get_st_model()
    embeddings: list[np.ndarray] = model.encode(sentences, convert_to_numpy=True)

    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_embeddings: list[np.ndarray] = []

    def _flush() -> None:
        nonlocal current_sentences, current_embeddings
        if not current_sentences:
            return
        text = " ".join(current_sentences)
        chunks.append(
            {"content": text, "metadata": copy.deepcopy(block["metadata"])}
        )
        current_sentences = []
        current_embeddings = []

    for sentence, emb in zip(sentences, embeddings):
        if not current_sentences:
            current_sentences.append(sentence)
            current_embeddings.append(emb)
            continue

        # Centroid of current chunk
        centroid = np.mean(current_embeddings, axis=0)
        sim = _cosine_similarity(emb, centroid)

        current_text_len = sum(len(s) for s in current_sentences)
        would_exceed = current_text_len + len(sentence) > CONFIG.semantic_max_chunk_size

        if sim >= CONFIG.semantic_similarity_threshold and not would_exceed:
            current_sentences.append(sentence)
            current_embeddings.append(emb)
        else:
            _flush()
            current_sentences.append(sentence)
            current_embeddings.append(emb)

    _flush()

    # Merge chunks that are too short into the preceding chunk
    merged: list[dict] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk["content"]) < CONFIG.semantic_min_chunk_size
        ):
            merged[-1]["content"] = merged[-1]["content"] + " " + chunk["content"]
        else:
            merged.append(chunk)

    return merged


def chunk_blocks(
    blocks: list[dict],
    mode: Literal["template", "semantic"] = "template",
) -> list[dict]:
    """Chunk a list of document blocks using the specified strategy.

    Args:
        blocks: List of block dicts produced by the parser, each with
            ``content`` and ``metadata`` keys.
        mode: Chunking strategy — ``"template"`` (fixed-size with overlap)
            or ``"semantic"`` (embedding-based sentence grouping).

    Returns:
        Flat list of chunk dicts. Each chunk has a ``content`` key and a
        ``metadata`` dict that preserves all keys from the source block.

    Raises:
        ValueError: If an unknown mode is specified.
    """
    if mode not in ("template", "semantic"):
        raise ValueError(f"Unknown chunking mode: {mode!r}. Choose 'template' or 'semantic'.")

    all_chunks: list[dict] = []
    table_count = 0

    for block in blocks:
        is_table = block["metadata"].get("block_type") == "table"

        if mode == "template":
            result = _template_chunk_block(block)
        else:
            result = _semantic_chunk_block(block)

        if is_table and result:
            table_count += 1

        all_chunks.extend(result)

    return all_chunks
