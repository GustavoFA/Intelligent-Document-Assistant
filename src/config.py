"""
Configuration constants for the RAG pipeline.
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

# Request / retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 2
REQUEST_TIMEOUT = 300

# Retrieval configuration
TOP_K = 5
MAX_CONTEXT_DISTANCE = 0.4

# Embedding configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 32

# Chunking configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

SEPARATORS = [
    r"\n\s*Art\.\s+\d+[A-Za-zº°-]*",
    r"\n\s*§\s*\d+[A-Za-zº°-]*",
    r"\n\s*Parágrafo único",
    r"\n\s*Inciso\s+[IVXLCDM]+",
    r"\n\s*\n",
    r"\n",
    r"\.\s+",
    r";\s+",
    r",\s+",
    r"\s+",
]

# Dataset configuration
HF_DATASET_NAME = "unicamp-dl/rag-rfb"
HF_DATA_FILE = "referred_legal_documents_QA_2024_v1.1.json"