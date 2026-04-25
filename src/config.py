from pathlib import Path
from dataclasses import dataclass

# Project root
ROOT_DIR = Path(__file__).parent.parent

@dataclass(frozen=True)
class Config:
    # Paths
    data_dir: Path = ROOT_DIR / "data"
    vector_db_dir: Path = ROOT_DIR / "vector_db"
    bm25_index_path: Path = ROOT_DIR / "vector_db" / "bm25_index.pkl"
    graph_path: Path = ROOT_DIR / "vector_db" / "entity_graph.pkl"

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking - template mode
    template_chunk_size: int = 1000
    template_chunk_overlap: int = 200

    # Chunking - semantic mode
    semantic_similarity_threshold: float = 0.75
    semantic_max_chunk_size: int = 1200
    semantic_min_chunk_size: int = 150

    # Retrieval
    retrieval_k_candidate: int = 20   # candidates before reranking
    retrieval_k_final: int = 5        # final chunks passed to LLM
    bm25_weight: float = 0.4          # weight in RRF fusion
    dense_weight: float = 0.6         # weight in RRF fusion
    rrf_k: int = 60                   # RRF constant

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.0

    # Entity graph
    entity_similarity_threshold: float = 0.80
    graph_hop_depth: int = 2
    graph_max_entities_per_chunk: int = 8
    graph_entity_types: tuple = ("MODEL", "TECHNIQUE", "DATASET",
                                  "CONCEPT", "ORGANIZATION")
    graph_retrieval_top_k: int = 3

    # Query understanding
    hyde_enabled: bool = True
    router_web_fallback_enabled: bool = True
    retrieval_confidence_threshold: float = 0.35
    max_multihop_steps: int = 3
    web_search_max_results: int = 5

    # Evaluation
    eval_faithfulness_threshold: float = 0.70
    eval_relevancy_threshold: float = 0.70
    eval_context_precision_threshold: float = 0.60
    eval_context_recall_threshold: float = 0.60

CONFIG = Config()
