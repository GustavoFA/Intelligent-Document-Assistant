"""
Configuration constants for RAG pipeline.
"""

# Ollama configuration
OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_API_URL}/chat"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_API_URL}/tags"

# Model configurations for different resource constraints
MODELS_CONFIG = {
    "high_resource": {
        "models": ["llama3", "llama2:13b", "mistral"],
        "description": "For systems with GPU (8GB+ VRAM) or high CPU resources",
    },
    "medium_resource": {
        "models": ["llama2:7b", "neural-chat", "mistral"],
        "description": "For systems with GPU (4-8GB VRAM) or moderate CPU",
    },
    "low_resource": {
        "models": ["orca-mini", "neural-chat:latest", "phi"],
        "description": "For systems with limited resources (CPU only or <4GB VRAM)",
    },
}

# Timeout configuration
MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds
REQUEST_TIMEOUT = 300  # 5 minutes
MIN_CONTEXT_QUALITY = 0.4  # Minimum distance threshold for context relevance
