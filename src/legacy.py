"""
Compatibility module for legacy imports.

Prefer direct imports from:
- src.prompt
- src.search
- src.llm
- src.pipeline
- src.config
"""

from .config import (
    OLLAMA_API_URL,
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    MODELS_CONFIG,
    MAX_RETRIES,
    RETRY_DELAY,
    REQUEST_TIMEOUT,
    MAX_CONTEXT_DISTANCE,
)

from .prompt import build_prompt, format_context_for_prompt
from .search import SearchEngine
from .llm import (
    check_ollama_availability,
    get_available_models,
    select_model,
    call_ollama,
    list_models_info,
)
from .pipeline import rag_generate

__all__ = [
    "SearchEngine",
    "rag_generate",
    "build_prompt",
    "format_context_for_prompt",
    "check_ollama_availability",
    "get_available_models",
    "select_model",
    "call_ollama",
    "list_models_info",
    "MAX_CONTEXT_DISTANCE",
]