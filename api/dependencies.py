
import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import faiss

from src.embeddings import load_model
from src.llm import check_ollama_availability, get_available_models
from src.search import SearchEngine

logger = logging.getLogger(__name__)

"""
Shared dependencies for FastAPI endpoints.

This module centralizes loading and caching of heavyweight resources used by
the RAG API, such as:

- artifacts directory
- FAISS index
- chunk metadata
- embedding model
- SearchEngine instance

The goal is to avoid reloading these resources on every request.
"""

@lru_cache(maxsize=1)
def get_artifacts_dir() -> Path:
    """Resolve and validate the artifacts directory."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    logger.info("Using artifacts directory: %s", artifacts_dir)
    return artifacts_dir


@lru_cache(maxsize=1)
def get_faiss_index() -> faiss.Index:
    """Load and cache the FAISS index."""
    artifacts_dir = get_artifacts_dir()
    index_path = artifacts_dir / "faiss_index.bin"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            f"Please run build_index.py first."
        )

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(str(index_path))

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    logger.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index


@lru_cache(maxsize=1)
def get_chunks() -> List[Dict[str, Any]]:
    """Load and cache chunk metadata."""
    artifacts_dir = get_artifacts_dir()
    chunks_path = artifacts_dir / "chunks.pkl"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}\n"
            f"Please run doc_process.py first."
        )

    logger.info("Loading chunks from %s", chunks_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    if not isinstance(chunks, list):
        raise ValueError("Loaded chunks object is not a list.")

    logger.info("Loaded %d chunks", len(chunks))
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the embedding model."""
    logger.info("Loading embedding model")
    model = load_model()
    logger.info("Embedding model loaded successfully")
    return model


def validate_retrieval_artifacts() -> None:
    """
    Validate consistency between FAISS index and chunk metadata.

    Raises:
        ValueError: If the number of FAISS vectors does not match the number of chunks.
    """
    index = get_faiss_index()
    chunks = get_chunks()

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Mismatch between FAISS index vectors ({index.ntotal}) "
            f"and chunks ({len(chunks)})."
        )


@lru_cache(maxsize=1)
def get_search_engine() -> SearchEngine:
    """
    Load and cache a SearchEngine instance.

    This is the preferred dependency for retrieval endpoints because it
    encapsulates the model, FAISS index, chunks, and retrieval logic.
    """
    logger.info("Initializing SearchEngine")
    search_engine = SearchEngine()

    # Optional consistency check
    if search_engine.index.ntotal != len(search_engine.chunks):
        raise ValueError(
            f"Mismatch between SearchEngine index vectors ({search_engine.index.ntotal}) "
            f"and chunks ({len(search_engine.chunks)})."
        )

    logger.info("SearchEngine initialized successfully")
    return search_engine


def check_faiss_availability() -> bool:
    """Return True if FAISS artifacts are available and valid."""
    try:
        validate_retrieval_artifacts()
        return True
    except (FileNotFoundError, ValueError):
        return False


def check_ollama_status() -> bool:
    """Return True if the Ollama service is reachable."""
    return check_ollama_availability()


def get_available_ollama_models() -> List[str]:
    """Return the list of available Ollama models."""
    return get_available_models()


def clear_cache() -> None:
    """
    Clear all cached resources.

    Useful for:
    - tests
    - development reloads
    - forcing artifact/model refresh without restarting the process
    """
    get_artifacts_dir.cache_clear()
    get_faiss_index.cache_clear()
    get_chunks.cache_clear()
    get_embedding_model.cache_clear()
    get_search_engine.cache_clear()
    logger.info("All dependency caches cleared")