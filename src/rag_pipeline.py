
"""
RAG (Retrieval-Augmented Generation) pipeline compatibility module.

This module provides backward compatibility by re-exporting functions from
the new modular structure. New code should import directly from the submodules:
- from src.prompt import build_prompt, format_context_for_prompt
- from src.retrieval import check_context_quality
- from src.llm import call_ollama, select_model, get_available_models, check_ollama_availability, list_models_info
- from src.pipeline import rag_generate
- from src.config import MODELS_CONFIG, MAX_RETRIES, etc.
"""

# Re-export for backward compatibility
from .config import (
    OLLAMA_API_URL,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    MODELS_CONFIG,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    MIN_CONTEXT_QUALITY,
)

from .prompt import build_prompt, format_context_for_prompt
from .retrieval import check_context_quality
from .llm import (
    check_ollama_availability,
    get_available_models,
    select_model,
    call_ollama,
    list_models_info,
)
from .pipeline import rag_generate, retrieve_context

__all__ = [
    # Config
    "OLLAMA_API_URL",
    "OLLAMA_CHAT_ENDPOINT",
    "OLLAMA_TAGS_ENDPOINT",
    "MODELS_CONFIG",
    "MAX_RETRIES",
    "RETRY_DELAY",
    "REQUEST_TIMEOUT",
    "MIN_CONTEXT_QUALITY",
    # Prompt
    "build_prompt",
    "format_context_for_prompt",
    # Retrieval
    "check_context_quality",
    "retrieve_context",
    # LLM
    "check_ollama_availability",
    "get_available_models",
    "select_model",
    "call_ollama",
    "list_models_info",
    # Pipeline
    "rag_generate",
]
