"""
Intelligent Document Assistant for Brazilian Portuguese Legal Documents.

RAG system for income tax (IRPF) document retrieval and question answering.
"""

from .config import *
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
    # Pipeline
    "rag_generate",
    "retrieve_context",
    # Prompt
    "build_prompt",
    "format_context_for_prompt",
    # Retrieval
    "check_context_quality",
    # LLM
    "check_ollama_availability",
    "get_available_models",
    "select_model",
    "call_ollama",
    "list_models_info",
]
