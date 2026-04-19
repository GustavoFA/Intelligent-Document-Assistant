"""
Intelligent Document Assistant for Brazilian Portuguese legal documents.

RAG system for income tax (IRPF) document retrieval and question answering.
"""

from .prompt import build_prompt, format_context_for_prompt
from .llm import (
    check_ollama_availability,
    get_available_models,
    select_model,
    call_ollama,
    list_models_info,
)
from .pipeline import rag_generate
from .search import SearchEngine

__all__ = [
    "rag_generate",
    "SearchEngine",
    "build_prompt",
    "format_context_for_prompt",
    "check_ollama_availability",
    "get_available_models",
    "select_model",
    "call_ollama",
    "list_models_info",
]