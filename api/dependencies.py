"""
Shared dependencies for FastAPI endpoints.

Handles loading and caching of FAISS index, chunks, and embedding model
to avoid reloading on every request.
"""

import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from functools import lru_cache

from src.embeddings import load_model
from src.llm import check_ollama_availability, get_available_models


# Cache paths
_artifacts_dir: Optional[Path] = None
_faiss_index: Optional[faiss.Index] = None
_chunks: Optional[List[Dict]] = None
_embedding_model = None


def get_artifacts_dir() -> Path:
    """Get and validate artifacts directory."""
    global _artifacts_dir
    if _artifacts_dir is None:
        _artifacts_dir = Path(__file__).parent.parent / "artifacts"
        if not _artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {_artifacts_dir}")
    return _artifacts_dir


def get_faiss_index() -> faiss.Index:
    """Load and cache FAISS index."""
    global _faiss_index
    if _faiss_index is None:
        artifacts_dir = get_artifacts_dir()
        index_path = artifacts_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                f"Please run build_index.py first."
            )
        _faiss_index = faiss.read_index(str(index_path))
    return _faiss_index


def get_chunks() -> List[Dict]:
    """Load and cache chunks."""
    global _chunks
    if _chunks is None:
        artifacts_dir = get_artifacts_dir()
        chunks_path = artifacts_dir / "chunks.pkl"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}\n"
                f"Please run doc_process.py first."
            )
        with open(chunks_path, "rb") as f:
            _chunks = pickle.load(f)
    return _chunks


def get_embedding_model():
    """Load and cache embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_model()
    return _embedding_model


def check_faiss_availability() -> bool:
    """Check if FAISS index and chunks are available."""
    try:
        get_faiss_index()
        get_chunks()
        return True
    except FileNotFoundError:
        return False


def check_ollama_status() -> bool:
    """Check if Ollama service is running."""
    return check_ollama_availability()


def get_available_ollama_models() -> List[str]:
    """Get list of available Ollama models."""
    return get_available_models()


def clear_cache():
    """Clear all cached resources. Useful for testing or reloading."""
    global _faiss_index, _chunks, _embedding_model, _artifacts_dir
    _faiss_index = None
    _chunks = None
    _embedding_model = None
    _artifacts_dir = None
