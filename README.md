# Intelligent-Document-Assistant

An end-to-end intelligent document assistant for Portuguese-Brazilian (PT-BR) legal documents. This system leverages **Retrieval-Augmented Generation (RAG)**, **semantic search**, and **FAISS indexing** to provide accurate, grounded, and explainable answers to questions about legal documents.

## Project Overview

This project implements a production-ready pipeline for:
- **Document Processing**: Chunk PT-BR legal documents respecting legal structure (Articles, Paragraphs, Inciso)
- **Semantic Embeddings**: Generate multilingual embeddings optimized for PT-BR legal content
- **Vector Search**: Build and query FAISS index for efficient similarity search
- **Retrieval**: Retrieve relevant legal document chunks based on semantic similarity

The system currently focuses on Brazilian income tax (IRPF) legal documents from the [unicamp-dl/rag-rfb](https://huggingface.co/datasets/unicamp-dl/rag-rfb) Hugging Face dataset.

## Future Improvements

- Improve the RAG system 
  - Chat memory
  - Citations
  - Anti-hallucination

- Improve the logging system

- Change pickle to JSON in chunk saving.

- Change index method for IndexFlatL2 (with normalization)

## Workflow

### Step 1: Document Processing (`doc_process.py`)

Loads and chunks documents from Hugging Face dataset:

```bash
python src/doc_process.py
```

**What it does:**
- Loads legal documents from `unicamp-dl/rag-rfb` dataset
- Cleans text (removes glyphs, normalizes whitespace)
- Chunks documents with legal structure awareness:
  - Separators: Articles, Paragraphs, Inciso, etc.
  - Chunk size: 700 tokens
  - Overlap: 100 tokens
- Saves chunks to `artifacts/chunks.pkl`

**Output:**
- `artifacts/chunks.pkl` - Serialized chunks with metadata (doc_id, filename, text, position)

### Step 2: Build FAISS Index (`build_index.py`)

Generates embeddings and builds vector index:

```bash
python src/build_index.py
```

**What it does:**
- Loads pre-processed chunks from `artifacts/chunks.pkl`
- Uses `embeddings.py` to generate embeddings:
  - Model: `intfloat/multilingual-e5-base`
  - Multilingual model optimized for semantic similarity
  - Batch processing (32 chunks per batch)
  - Generates 768-dimensional embeddings
- Creates FAISS index:
  - IndexFlatL2 for L2 distance-based search
  - Supports efficient similarity search
- Saves index and metadata to artifacts

**Output:**
- `artifacts/faiss_index.bin` - Serialized FAISS index
- `artifacts/chunks_metadata.json` - Chunk metadata for reference
- `artifacts/index_summary.txt` - Index statistics and configuration

### Embedding Management (`embeddings.py`)

Centralized module for embedding operations used by other scripts:

**Configuration (from `config.py`):**
- `EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"` - Multilingual embedding model
- `EMBEDDING_BATCH_SIZE = 32` - Batch size for processing

**Functions:**
- `load_model(model_name)` - Loads pretrained multilingual embedding model
- `generate_embeddings(chunks, model, batch_size)` - Generates embeddings for document chunks
- `encode_query(query, model)` - Encodes query strings into embedding vectors

Used by: `build_index.py` and `search.py`

### Step 3: Search (`search.py`)

Query the indexed documents using the `SearchEngine` class:

```bash
# Interactive CLI
python -m src.search

# Programmatic usage
from src.search import SearchEngine

engine = SearchEngine()
results = engine.retrieve("Como calcular o imposto de renda?", top_k=5)
```

**SearchEngine Class:**
- `__init__()` - Initializes search engine (loads model, index, and chunks)
- `retrieve(query, top_k)` - Retrieves top-k relevant chunks for a query
- `load_index()` - Loads FAISS index from disk
- `load_chunks()` - Loads chunk metadata from disk
- `get_artifacts_dir()` - Resolves and validates artifacts directory

**Features:**
- SearchEngine class encapsulates retrieval logic and resources
- Interactive command-line interface for searching
- Programmatic search interface for integration
- Returns ranked results with:
  - Rank and score (L2 distance)
  - Document metadata (filename, doc_id, chunk_id)
  - Full chunk text
- Input validation and error handling

### Step 4: RAG Generation (`pipeline.py`)

Generate LLM responses using end-to-end Retrieval-Augmented Generation (RAG):

```python
from src.pipeline import rag_generate

# Option 1: Automatic retrieval (simplest)
response = rag_generate(
    user_prompt="Como calcular o imposto de renda?"
)
print(response)

# Option 2: With pre-retrieved context
from src.pipeline import retrieve_context
context = retrieve_context("Como calcular o imposto de renda?", top_k=5)
response = rag_generate(
    user_prompt="Como calcular o imposto de renda?",
    context=context
)
print(response)
```

**Pipeline Flow:**
```
User Query
   ↓
retrieve_context (FAISS)
   ↓
check_context_quality (distance < 0.4)
   ↓
build_prompt (system + user messages)
   ↓
call_ollama (with retry logic)
   ↓
Response
```

**Features:**
- **Automatic Retrieval**: Retrieves relevant chunks from FAISS index using semantic search
- **Context Validation**: Checks if retrieved context meets quality threshold
- **Structured Prompts**: Builds prompts with separate system and user roles
- **LLM Generation**: Calls Ollama with automatic retry on timeout/connection errors
- **Resource-Aware**: Auto-selects model based on available resources:
  - **High Resource**: llama3, llama2:13b (8GB+ VRAM)
  - **Medium Resource**: llama2:7b, neural-chat (4-8GB VRAM)
  - **Low Resource**: orca-mini, phi (CPU or <4GB VRAM)
- **Configurable**: Supports temperature, top_p, top_k tuning

**Requirements:**
- FAISS index and chunks created (`python src/build_index.py`)
- Ollama running locally (`ollama serve`)
- At least one language model installed
- Retrieved context with minimum quality threshold (distance < 0.4)

**Model Selection:**
```bash
# Install Ollama models
ollama pull llama2          # Balanced (7B)
ollama pull mistral         # Fast and efficient
ollama pull neural-chat     # Lightweight
```

**Functions:**
- `retrieve_context(query, top_k=5)` - Retrieves context from FAISS using semantic search
- `rag_generate(user_prompt, context=None, ...)` - Complete RAG pipeline with automatic retrieval
- `build_prompt()` - Constructs prompt with system and user messages
- `call_ollama()` - Calls Ollama API with retry logic
- `check_context_quality()` - Validates context relevance
- `select_model()` - Selects model based on resource constraints

**Configuration:**
- `MAX_RETRIES = 2` - Number of retries on timeout/connection errors
- `RETRY_DELAY = 2` - Seconds to wait between retries
- `REQUEST_TIMEOUT = 300` - Seconds timeout for API requests
- `MIN_CONTEXT_QUALITY = 0.4` - Minimum distance threshold for context (lower is better)
- `top_k = 5` - Default number of context chunks to retrieve

## Module Architecture

The RAG pipeline is organized into modular components for better maintainability and reusability:

### `config.py`
Centralized configuration constants for the entire pipeline:
- Embedding model and batch size
- Chunk size and overlap settings
- Hugging Face dataset information
- Ollama API endpoints and model configurations
- Timeout and retry settings
- Quality thresholds

```python
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS,
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE,
    TOP_K, MAX_CONTEXT_DISTANCE,
    OLLAMA_API_URL, MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT,
    HF_DATASET_NAME, HF_DATA_FILE,
    MODELS_CONFIG
)
```

### `embeddings.py`
Centralized module for embedding operations:
```python
from src.embeddings import load_model, generate_embeddings, encode_query
```
Uses `EMBEDDING_MODEL_NAME` and `EMBEDDING_BATCH_SIZE` from config.

### `doc_process.py`
Document loading, cleaning, and chunking:
```python
from src.doc_process import chunk_documents, clean_legal_text, load_documents
```
Uses `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEPARATORS`, `HF_DATASET_NAME`, `HF_DATA_FILE` from config.

### `build_index.py`
FAISS index creation from embeddings:
```python
from src.build_index import create_faiss_index, save_index, load_chunks
```
Uses `TOP_K` and `EMBEDDING_MODEL_NAME` from config.

### `search.py`
SearchEngine class for semantic document retrieval:
```python
from src.search import SearchEngine, format_results

engine = SearchEngine()  # Loads model, index, and chunks
results = engine.retrieve(query, top_k=5)
formatted = format_results(query, results)
```
Uses `TOP_K` and `MAX_CONTEXT_DISTANCE` from config.

### `prompt.py`
Utilities for building structured prompts with system and user roles:
```python
from src.prompt import build_prompt, format_context_for_prompt

system_msg, user_msg = build_prompt(user_prompt, context=results)
```

### `retrieval.py`
Context validation and quality checking:
```python
from src.retrieval import check_context_quality

is_valid = check_context_quality(context)
```

### `llm.py`
Ollama LLM integration with retry logic:
```python
from src.llm import call_ollama, select_model, check_ollama_availability

model = select_model()
response = call_ollama(system_message, user_message)
```

### `pipeline.py`
Main RAG pipeline orchestration with integrated retrieval:
```python
from src.pipeline import rag_generate, retrieve_context

# Complete end-to-end RAG pipeline
response = rag_generate("Como calcular o imposto de renda?")

# Or retrieve context separately
context = retrieve_context("Como calcular o imposto de renda?", top_k=5)
response = rag_generate("Como calcular o imposto de renda?", context=context)
```

### Backward Compatibility
The original `rag_pipeline.py` re-exports all functions from the modular structure, maintaining backward compatibility with existing code.

## Project Structure

```
Intelligent-Document-Assistant/
├── README.md
├── LICENSE
├── requirements.in
├── requirements.txt
├── run_api.sh
├── notebooks/
│   └── retrieval.ipynb          # Exploratory notebook with examples
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── schemas.py               # Request/response schemas
│   └── dependencies.py          # Dependency injection
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Centralized configuration constants
│   ├── doc_process.py           # Document processing and chunking
│   ├── embeddings.py            # Embedding generation and management
│   ├── build_index.py           # FAISS index creation
│   ├── search.py                # SearchEngine class for semantic search
│   ├── prompt.py                # Prompt building utilities
│   ├── retrieval.py             # Context validation utilities
│   ├── llm.py                   # Ollama LLM integration
│   ├── pipeline.py              # RAG pipeline orchestration
│   ├── rag_pipeline.py          # Backward compatibility re-exports
│   └── legacy.py                # Legacy functions (if any)
└── artifacts/                   # Generated artifacts (indexed files)
    ├── chunks.pkl               # Processed chunks
    ├── faiss_index.bin          # FAISS index
    ├── chunks_metadata.json     # Metadata
    └── index_summary.txt        # Summary
```

## Quick Start

**Full retrieval pipeline:**
```bash
# 1. Process documents
python -m src.doc_process

# 2. Build FAISS index
python -m src.build_index

# 3. Start searching
python -m src.search
```

**Programmatic search:**
```python
from src.search import SearchEngine

# Initialize engine (loads model, index, and chunks)
engine = SearchEngine()

# Retrieve relevant chunks
results = engine.retrieve("Como calcular o imposto de renda?", top_k=5)

# Format and display results
from src.search import format_results
print(format_results("Como calcular o imposto de renda?", results))
```

**Full RAG pipeline (with LLM generation):**
```bash
# Install and start Ollama
ollama serve

# In another terminal, pull a model
ollama pull llama2  # or mistral, neural-chat, etc.

# Run the RAG pipeline
python -c "
from src.search import SearchEngine
from src.pipeline import rag_generate

engine = SearchEngine()
context = engine.retrieve('Como calcular o imposto de renda?', top_k=5)
response = rag_generate('Como calcular o imposto de renda?', context=context)
print(response)
"
```

**Example queries (PT-BR):**
- "Como calcular o imposto de renda?"
- "Qual é a alíquota do IRPF?"
- "Como fazer dedução fiscal?"

## Configuration

All configuration is centralized in `config.py`. Key parameters:

**Document Processing:**
```python
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, HF_DATASET_NAME, HF_DATA_FILE
```
- `CHUNK_SIZE = 700` - Token size for each chunk
- `CHUNK_OVERLAP = 100` - Token overlap between chunks
- `SEPARATORS` - Legal structure patterns (Articles, Paragraphs, etc.)
- `HF_DATASET_NAME = "unicamp-dl/rag-rfb"` - Hugging Face dataset
- `HF_DATA_FILE = "referred_legal_documents_QA_2024_v1.1.json"` - Dataset file

**Embeddings:**
```python
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE
```
- `EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"` - Embedding model
- `EMBEDDING_BATCH_SIZE = 32` - Batch size for embedding generation

**Retrieval & Search:**
```python
from src.config import TOP_K, MAX_CONTEXT_DISTANCE
```
- `TOP_K = 5` - Default number of results
- `MAX_CONTEXT_DISTANCE = 0.4` - Maximum distance threshold for context quality

**Ollama Configuration:**
```python
from src.config import OLLAMA_API_URL, MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT, MODELS_CONFIG
```
- `OLLAMA_API_URL = "http://localhost:11434/api"` - Ollama server URL
- `MAX_RETRIES = 2` - Retry attempts on timeout/connection errors
- `RETRY_DELAY = 2` - Seconds between retry attempts
- `REQUEST_TIMEOUT = 300` - Request timeout in seconds
- `MODELS_CONFIG` - Resource-aware model configurations (high/medium/low resource)

## REST API (FastAPI)

The system provides a production-ready REST API for semantic search and RAG generation.

### Starting the API Server

```bash
# Option 1: Using provided script
bash run_api.sh

# Option 2: Direct command
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**API will be available at:**
- Main API: `http://localhost:8000`
- Interactive Docs (Swagger): `http://localhost:8000/docs`
- Alternative Docs (ReDoc): `http://localhost:8000/redoc`

### API Flow

```
User → HTTP Request → FastAPI → rag_generate() → Response JSON
```

### API Endpoints

#### 1. Health Check
```bash
GET /health
```
Check system status and component availability.

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "faiss_index_available": true,
  "chunking_data_available": true
}
```

#### 2. List Available Models
```bash
GET /models
```
Get available Ollama models and resource configurations.

**Response:**
```json
{
  "high_resource": ["llama3", "llama2:13b", "mistral"],
  "medium_resource": ["llama2:7b", "neural-chat", "mistral"],
  "low_resource": ["orca-mini", "neural-chat:latest", "phi"],
  "available_models": ["llama2:latest", "mistral:latest"]
}
```

#### 3. Semantic Search
```bash
POST /search
Content-Type: application/json

{
  "query": "Como calcular o imposto de renda?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "Como calcular o imposto de renda?",
  "results": [
    {
      "rank": 1,
      "distance": 0.2393,
      "filename": "Lei nº 11.482.txt",
      "text": "O imposto de renda incidente sobre os rendimentos..."
    }
  ],
  "count": 5
}
```

#### 4. RAG Generation (with pre-retrieved context)
```bash
POST /rag/generate
Content-Type: application/json

{
  "user_prompt": "Como calcular o imposto de renda?",
  "context": [
    {
      "rank": 1,
      "distance": 0.2393,
      "doc_id": "doc_1",
      "chunk_id": 5,
      "filename": "Lei nº 11.482.txt",
      "text": "O imposto de renda incidente sobre os rendimentos..."
    }
  ],
  "model": "llama2",
  "resource_level": "medium_resource",
  "temperature": 0.7,
  "top_k": 5
}
```

**Response:**
```json
{
  "user_prompt": "Como calcular o imposto de renda?",
  "response": "O imposto de renda é calculado de acordo com a legislação... [LLM response]",
  "context_count": 1,
  "model_used": "llama2"
}
```

#### 5. RAG Generation (with automatic context retrieval)
```bash
POST /rag/generate/auto?query=Como+calcular+o+imposto+de+renda?&top_k=5&model=llama2&temperature=0.7
```

**Response:**
```json
{
  "user_prompt": "Como calcular o imposto de renda?",
  "response": "O imposto de renda é calculado de acordo com a legislação... [LLM response]",
  "context_count": null,
  "model_used": "llama2"
}
```

### Example Requests

**Using curl (Automatic RAG):**
```bash
curl -X POST "http://localhost:8000/rag/generate/auto" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Como calcular o imposto de renda?"
  }'
```

**Using Python (with requests):**
```python
import requests

# Search for context
search_response = requests.post(
    "http://localhost:8000/search",
    json={"query": "Como calcular o imposto de renda?", "top_k": 5}
)
context = search_response.json()["results"]

# Generate response with context
rag_response = requests.post(
    "http://localhost:8000/rag/generate",
    json={
        "user_prompt": "Como calcular o imposto de renda?",
        "context": context,
        "model": "llama2"
    }
)
print(rag_response.json()["response"])
```

**Using Python (automatic retrieval):**
```python
import requests

response = requests.post(
    "http://localhost:8000/rag/generate/auto",
    params={
        "query": "Como calcular o imposto de renda?",
        "top_k": 5,
        "model": "llama2",
        "temperature": 0.7
    }
)
print(response.json()["response"])
```

### API Configuration

**api/schemas.py:**
- Pydantic models for request/response validation
- Type checking and documentation generation

**api/main.py:**
- FastAPI application with all endpoints
- CORS middleware enabled
- Automatic documentation (Swagger, ReDoc)

**api/dependencies.py:**
- Cached loading of FAISS index, chunks, and embedding model
- Health check functions
- Resource management

### Performance Tips

1. **Pre-compute context**: For better performance, retrieve context separately and reuse:
   ```python
   # Slow: context retrieved on every request
   POST /rag/generate/auto
   
   # Faster: context retrieved once, reused for multiple queries
   POST /search → GET context
   POST /rag/generate → Use same context
   ```

2. **Resource Optimization**: Choose resource level based on your hardware:
   ```bash
   # For limited resources
   "resource_level": "low_resource"
   
   # For typical setup
   "resource_level": "medium_resource"
   
   # For high-end GPU
   "resource_level": "high_resource"
   ```

3. **Context Caching**: FAISS index and chunks are cached in memory after first load

## Installation

### Option 1: Using requirements.txt (Recommended for reproducibility)

```bash
pip install -r requirements.txt
```

This installs all dependencies with pinned versions, ensuring reproducible environments across different machines and setups.

### Option 2: Using requirements.in

```bash
pip install -r requirements.in
```

This installs core dependencies with flexible version constraints, suitable for development environments.

### Manual Installation

```bash
pip install faiss-cpu sentence-transformers langchain-text-splitters datasets numpy pandas tqdm
```

Or for GPU support (requires CUDA):
```bash
pip install faiss-gpu sentence-transformers langchain-text-splitters datasets numpy pandas tqdm
```

## Core Dependencies

**RAG Pipeline:**
- `faiss-cpu` or `faiss-gpu` - Vector similarity search
- `sentence-transformers` - Multilingual embeddings
- `langchain-text-splitters` - Document chunking
- `datasets` - Hugging Face dataset loading
- `requests` - HTTP calls to Ollama API

**Data Processing:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `tqdm` - Progress bars

**Web API (FastAPI):**
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation using Python type annotations

## License

See [LICENSE](LICENSE) file for details.
